import sys
from copy import deepcopy
from pathlib import Path

import torch.nn as nn

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.network.actor_critic import ActorCritic
from solution_methods.L2D.src.agent_utils import eval_actions
from solution_methods.L2D.src.mb_agg import *

base_path = Path(__file__).resolve().parents[3]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/L2D.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameters"]
model_parameters = parameters["network_parameters"]
train_parameters = parameters["train_parameters"]
test_parameters = parameters["test_parameters"]

device = torch.device(env_parameters["device"])


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []
        self.weights = None  # Added to store job weights

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]
        self.weights = None


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.n_jobs = n_j  # Store number of jobs for reward calculation

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)

        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=train_parameters["decay_step_size"],
                                                         gamma=train_parameters["decay_ratio"])

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):

        vloss_coef = train_parameters["vloss_coef"]
        ploss_coef = train_parameters["ploss_coef"]
        entloss_coef = train_parameters["entloss_coef"]

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []

        # store data for all env
        for i in range(len(memories)):
            # Calculate returns handling weighted case
            returns = self.calculate_returns(
                rewards=memories[i].r_mb,
                dones=memories[i].done_mb,
                weights=memories[i].weights
            )
            
            rewards_all_env.append(returns)

            # BEFORE: rewards = []
            #discounted_reward = 0
            #for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
            #    if is_terminal:
            #        discounted_reward = 0
            #    discounted_reward = reward + (self.gamma * discounted_reward)
            #    rewards.insert(0, discounted_reward)
            #rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            #rewards_all_env.append(rewards)
            
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        # get batch argument for net forwarding: mb_g_pool is same for all env
        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                # Evaluate actions and calculate losses
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                # Calculate advantages
                if rewards_all_env[i].dim() > 1:
                    # Weighted instance case
                    weighted_rewards = rewards_all_env[i].sum(dim=1)  # Sum across jobs
                else:
                    # Unweighted instance case
                    weighted_rewards = rewards_all_env[i]
                # BEFORE: advantages = rewards_all_env[i] - vals.view(-1).detach()
                advantages = weighted_rewards - vals.view(-1).detach()
                # PPO losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # Value loss
                # BEFORE: v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                v_loss = self.V_loss_2(vals.squeeze(), weighted_rewards)
                # Policy loss
                p_loss = - torch.min(surr1, surr2).mean()
                # Entropy loss
                ent_loss = - ent_loss.clone()
                # Combined loss
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss

                loss_sum += loss
                vloss_sum += v_loss
                
            # Update networks
            self.optimizer.zero_grad()
            #print('loss_sum', loss_sum)
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Update learning rate if needed
        if train_parameters["decayflag"]:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()

    def calculate_returns(self, rewards, dones, weights=None):
        """
        Calculate returns considering weighted completion times if weights provided
        
        Args:
            rewards: List of rewards from environment (changes in weighted completion time)
            dones: List of done flags
            weights: Optional tensor of job weights (n_jobs,)
            
        Returns:
            torch.tensor: Calculated returns, properly scaled and normalized
        """
        returns = []
        discounted_reward = 0
        
        if weights is None:
            # Standard makespan case - original logic
            for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
        
        else:
            # Weighted completion time case - scale rewards by job weights
            normalized_weights = weights / weights.sum() # Normalize weights to prevent scaling issues
            
            for i, (reward, is_terminal) in enumerate(zip(reversed(rewards), reversed(dones))):
                if is_terminal:
                    discounted_reward = 0
                
                # Scale reward by normalized weight of affected job
                job_idx = (len(rewards) - 1 - i) % self.n_jobs # Index of job associated with reward
                scaled_reward = reward * normalized_weights[job_idx]
                
                discounted_reward = scaled_reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
        
        # Ensure returns is a list before converting to tensor
        if not isinstance(returns, list):
            returns = list(returns)
            
        returns = torch.tensor(returns, dtype=torch.float).to(device) # Convert to tensor
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # Normalize returns
        
        return returns