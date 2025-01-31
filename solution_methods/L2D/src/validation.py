import sys
from pathlib import Path

import numpy as np
import torch

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.src.agent_utils import greedy_select_action
from solution_methods.L2D.src.JSSP_Env import SJSSP
from solution_methods.L2D.src.mb_agg import g_pool_cal

base_path = Path(__file__).resolve().parents[3]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/L2D.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameters"]
model_parameters = parameters["network_parameters"]
train_parameters = parameters["train_parameters"]


def validate(vali_set, model):
    """
    Validate model performance on validation set.
    
    Args:
        vali_set: List of validation instances
        model: Policy model to evaluate
        weights: Optional job weights, if None uses makespan objective
    
    Returns:
        np.array: Objectives (makespan or weighted completion times) for each instance
    """
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]
    # Check if we have weighted instances
    if len(vali_set[0]) == 3:  # If first instance has 3 elements, it's weighted
        weights = vali_set[0][2][:, -1]  # Extract last column from weights matrix
    
    # BEFORE: env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES)
    # Initialize environment with optional weights
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, weights=weights)
    device = torch.device(env_parameters["device"])
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    # BEFORE: make_spans = []
    # Track objectives for each instance
    objectives = []
    print("\nStarting validation loop")
    
    # rollout using model
    #for data in vali_set:
    for i, data in enumerate(vali_set):
        adj, fea, candidate, mask = env.reset(data)
        rewards = - env.initQuality
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = model(x=fea_tensor,
                              graph_pool=g_pool_step,
                              padded_nei=None,
                              adj=adj_tensor,
                              candidate=candidate_tensor.unsqueeze(0),
                              mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask = env.step(action.item())
            rewards += reward
            if done:
                break
        # BEFORE: make_spans.append(rewards - env.posRewards)
        final_objective = rewards - env.posRewards
        objectives.append(final_objective)
        # print(rewards - env.posRewards)
    # BEFORE: return np.array(make_spans)
    result = np.array(objectives)
    return result
    #return np.array(objectives)




