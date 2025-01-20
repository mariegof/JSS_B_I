"""
Training Environment for Job Shop Scheduling Problem (JSSP)

Goals:
- Provide the main training environment for learning JSSP policies
- Implement core JSSP mechanics including state transitions and reward computation
- Support flexible job shop configurations and constraints
- Enable collection of training experiences for reinforcement learning

Inputs:
- n_j (int): Number of jobs in the problem instance
- n_m (int): Number of machines in the problem instance
- data (tuple): Contains duration matrix and machine assignment matrix

Outputs:
- adj (np.array): Adjacency matrix representing the current state's disjunctive graph
- fea (np.array): Feature matrix containing normalized processing times and completion status
- reward (float): Reward signal based on makespan improvement
- done (bool): Flag indicating if all operations are scheduled
- omega (np.array): Array of currently schedulable operations
- mask (np.array): Boolean mask indicating which jobs are completed
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils import EzPickle

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.src.permissibleLS import permissibleLeftShift
from solution_methods.L2D.data.instance_generator import override
from solution_methods.L2D.src.updateAdjMat import getActionNbghs
from solution_methods.L2D.src.updateEntTimeLB import calEndTimeLB

base_path = Path(__file__).resolve().parents[3]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/L2D.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameters"]
model_parameters = parameters["network_parameters"]
train_parameters = parameters["train_parameters"]
test_parameters = parameters["test_parameters"]


class SJSSP(gym.Env, EzPickle):
    def __init__(self, n_j, n_m, weights=None):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        
        # Add job weights - if not provided, use equal weights
        self.weights = weights if weights is not None else np.ones(self.number_of_jobs)
        
        # Keep track of job completion times
        self.job_completion_times = np.zeros(self.number_of_jobs)
        
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        
        # Current job being processed
        job_id = action // self.number_of_machines
        
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:

            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                job_id = action // self.number_of_machines
                self.job_completion_times[job_id] = startTime_a + dur_a
                self.completed_jobs.add(job_id)
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0

        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1)/env_parameters["et_normalize_coef"],
                              self.finished_mark.reshape(-1, 1)), axis=1)
        
        # Calculate new weighted completion time
        current_weighted_completion = self.calculate_weighted_completion_time()
        
        # BEFORE: reward = - (self.LBs.max() - self.max_endTime)
        # Reward is negative change in weighted completion time
        reward = -(current_weighted_completion - self.max_endTime)
        if reward == 0:
            reward = env_parameters["rewardscale"]
            self.posRewards += reward
        # BEFORE: self.max_endTime = self.LBs.max()
        # Update max end time for next iteration
        self.max_endTime = current_weighted_completion

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):

        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        
        # Reset job completion tracking
        self.job_completion_times = np.zeros(self.number_of_jobs)
        self.completed_jobs = set()
        
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not env_parameters["init_quality_flag"] else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1)/env_parameters["et_normalize_coef"],
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # start time of operations on machines
        self.mchsStartTimes = -env_parameters["high"] * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        return self.adj, fea, self.omega, self.mask
    
    def calculate_weighted_completion_time(self):
        """Calculate the weighted sum of completion times for all completed jobs"""
        return sum(self.weights[i] * self.job_completion_times[i] 
                  for i in range(self.number_of_jobs) 
                  if i in self.completed_jobs)
