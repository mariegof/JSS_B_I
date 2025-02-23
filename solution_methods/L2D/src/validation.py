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
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]
    
    # Determine instance type and configure environment
    is_weighted = len(vali_set[0]) == 3
    if is_weighted:
        weights = vali_set[0][2][:, -1]
        print(f"Validating on weighted instances with {N_JOBS} jobs and {N_MACHINES} machines")
    else:
        weights = None
        print(f"Validating on unweighted instances with {N_JOBS} jobs and {N_MACHINES} machines")

    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, weights=weights)
    device = torch.device(env_parameters["device"])
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    objectives = []
    # rollout using model
    for data in vali_set:
        # Reset environment with current instance
        adj, fea, candidate, mask = env.reset(data)
        rewards = - env.initQuality
        # Run episode until completion
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            
            # Get action from model
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
        objectives.append(rewards - env.posRewards)
        # print(rewards - env.posRewards)
    return np.array(objectives)



