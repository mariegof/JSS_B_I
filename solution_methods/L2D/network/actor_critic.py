import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.network.graphCNN import GraphCNN
from solution_methods.L2D.network.mlp import MLPActor, MLPCritic

# from solutions.save.Params import configs

class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim, # Will be 2 for unweighted, 3 for weighted
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        """
        Actor-Critic network for job shop scheduling with optional job weights.
        
        Features:
        - CLB (Critical Lower Bound): Time-based feature
        - I(O,st): Binary indicator for scheduled operations
        - Weight: Job priority (only present if input_dim == 3)
        """
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        self.weighted = (input_dim == 3)  # Track if we're using weights

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim, # 2 or 3 based on env config
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        
        # Actor network - input dimension varies based on weighted/unweighted
        #actor_input_dim = hidden_dim * 2 + (1 if self.weighted else 0)  # 129 or 128
        #self.actor = MLPActor(num_mlp_layers_actor, actor_input_dim, hidden_dim_actor, 1).to(device)
        # Actor network - uses GNN embeddings which already incorporate weight information
        self.actor = MLPActor(
            num_mlp_layers_actor, 
            hidden_dim * 2,  # Concatenated operation and graph embeddings
            hidden_dim_actor, 
            1
        ).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,
                ):
        """Forward pass handling both weighted and unweighted cases.
        
        Args:
            x: Node features tensor (batch_size * n_nodes, feature_dim)
                - Unweighted case: [CLB, I(O,st)]
                - Weighted case: [CLB, I(O,st), weight] (weights only on terminal ops)
            graph_pool: Graph pooling matrix
            padded_nei: Padded neighbors information
            adj: Adjacency matrix
            candidate: Candidate operations
            mask: Mask for invalid operations
            
        Returns:
            tuple: (pi, v)
                - pi: Action probabilities incorporating job weights through GNN
                - v: State value estimation
        """
        # 1. Extract features using GNN - weight information (if present) is automatically
        # incorporated into embeddings through message passing
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        
        # prepare policy feature: concat omega feature with global feature
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        # Repeat graph-level embedding for each candidate
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)

        '''# prepare policy feature: concat row work remaining feature
        durfea2mat = x[:, 1].reshape(shape=(-1, self.n_j, self.n_m))
        mask_right_half = torch.zeros_like(durfea2mat)
        mask_right_half.put_(omega, torch.ones_like(omega, dtype=torch.float))
        mask_right_half = torch.cumsum(mask_right_half, dim=-1)
        # calculate work remaining and normalize it with job size
        wkr = (mask_right_half * durfea2mat).sum(dim=-1, keepdim=True)/self.n_ops_perjob'''

        # concatenate feature
        # concateFea = torch.cat((wkr, candidate_feature, h_pooled_repeated), dim=-1)
        # Combine features based on weighted/unweighted case
        '''if self.weighted:
            # Extract weights from input features and reshape properly
            weights = x[:, -1]  # Get weights from last column
            # First reshape to [batch_size, n_m, n_j]
            weights = weights.reshape(-1, self.n_m, self.n_j)
            # Take the first row for each job (weights are same for all operations of a job)
            weights = weights[:, 0, :]
            # Reshape to [batch_size, n_j, 1] for concatenation
            weights = weights.unsqueeze(-1)
            # Concatenate all features including weights
            concateFea = torch.cat(
                (candidate_feature, h_pooled_repeated, weights),
                dim=-1
            )
        else:
            # Original unweighted concatenation
            concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        '''
        # Concatenate features - embeddings already contain weight information from GNN
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        # Pass through actor network (may need to adjust input dimensions)
        candidate_scores = self.actor(concateFea)

        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')

        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled)
        return pi, v


if __name__ == '__main__':
    base_path = Path(__file__).resolve().parents[3]
    sys.path.append(str(base_path))

    param_file = str(base_path) + "/configs/L2D.toml"
    parameters = load_parameters(param_file)
    env_parameters = parameters["env_parameters"]
    model_parameters = parameters["network_parameters"]
    train_parameters = parameters["train_parameters"]
    test_parameters = parameters["test_parameters"]
    ac = ActorCritic(
              n_j=env_parameters["n_j"],
              n_m=env_parameters["n_m"],
              num_layers=model_parameters["num_layers"],
              neighbor_pooling_type=model_parameters["neighbor_pooling_type"],
              input_dim=model_parameters["input_dim"],
              hidden_dim=model_parameters["hidden_dim"],
              num_mlp_layers_feature_extract=model_parameters["num_mlp_layers_feature_extract"],
              num_mlp_layers_actor=model_parameters["num_mlp_layers_actor"],
              hidden_dim_actor=model_parameters["hidden_dim_actor"],
              num_mlp_layers_critic=model_parameters["num_mlp_layers_critic"],
              hidden_dim_critic=model_parameters["hidden_dim_critic"],
              device=torch.device(env_parameters["device"]),
        learn_eps=False
    )
    print(ac)
    print('?? Go home')