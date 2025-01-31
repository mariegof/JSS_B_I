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
        self.finished_mark = np.zeros((n_j, n_m))
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        self.weights = weights #if weights is not None else np.ones(n_j)  # Default weights = 1
        self.job_completion_times = np.zeros(self.number_of_jobs, dtype=np.float32)  # Track job completion times for weighted objective
        self.completed_jobs = set()  # Tracks completed jobs

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # Calculate objective before action
        prev_objective = (self.calculate_weighted_completion_time() if self.weights is not None else self.LBs.max())
        
        # Current job being processed
        job_id = action // self.number_of_machines
        
        # Execute action if not already processed, redundant action makes no effect
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
                self.job_completion_times[job_id] = startTime_a + dur_a  # Track completion time
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

        fea = np.concatenate((
            self.LBs.reshape(-1, 1)/env_parameters["et_normalize_coef"],  # CLB
            self.finished_mark.reshape(-1, 1),  # I(O, st)
        ), axis=1)
                        
        if self.weights is not None:
            # Create weight features with zeros and only set weights for terminal operations
            weight_features = np.zeros((self.number_of_tasks, 1), dtype=np.float32)  # Shape (n_jobs * n_machines, 1)
            weight_features[self.last_col] = self.weights.reshape(-1, 1)
            # Concatenate features
            fea = np.concatenate((fea, weight_features), axis=1)
        
        # Calculate new objective and reward
        current_objective = self.calculate_weighted_completion_time()
        
        # BEFORE: reward = - (self.LBs.max() - self.max_endTime)
        # Calculate reward as improvement in objective
        reward = -(current_objective - prev_objective)
        if reward == 0:
            reward = env_parameters["rewardscale"]
            self.posRewards += reward
        # BEFORE: self.max_endTime = self.LBs.max()
        self.max_endTime = (self.calculate_weighted_completion_time() if self.weights is not None else self.LBs.max())

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):

        self.step_count = 0 
        # BEFORE: self.m = data[-1] # machine assignment matrix
        self.m = data[1] # machine assignment matrix
        self.dur = data[0].astype(np.single)  # Duration matrix
        if len(data) == 3:  # Weighted case
            # Update weights from data - take the last value of each row
            self.weights = data[2][:, -1]  # This will give us a 1D array of shape (nb_job,)
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
        # BEFORE: self.initQuality = self.LBs.max() if not env_parameters["init_quality_flag"] else 0
        self.initQuality = (self.calculate_weighted_completion_time() 
                   if self.weights is not None and not env_parameters["init_quality_flag"]
                   else self.LBs.max() if not env_parameters["init_quality_flag"]
                   else 0)
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1)/env_parameters["et_normalize_coef"],
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        
        if self.weights is not None:
            # Create weight features with zeros and only set weights for terminal operations
            weight_features = np.zeros((self.number_of_tasks, 1), dtype=np.float32)  # Shape (n_jobs * n_machines, 1)
            weight_features[self.last_col] = self.weights.reshape(-1, 1)
            # Concatenate features
            fea = np.concatenate((fea, weight_features), axis=1)
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
        """
        Calculate weighted sum of completion times: Σ(w_j * C_j)
        
        Uses three-phase approach:
        1. Calculate raw completion times
        2. Apply workload-based scaling
        3. Weight the scaled completion times
        
        Returns:
            float: Weighted sum of completion times if weights present, 
                otherwise returns makespan
        """
        if self.weights is None:
            return self.LBs.max()  # Return makespan for unweighted case
        # Phase 1: Calculate raw completion times
        completion_times = np.zeros(self.number_of_jobs)
        #total_workload = 0
        #max_workload_per_job = 0
        
        for i in range(self.number_of_jobs):
            # Calculate total workload for scaling reference
            #job_workload = np.sum(self.dur[i, :]) # Sum of all operations duration
            #total_workload += job_workload # Sum of all jobs' workload
            #max_workload_per_job = max(max_workload_per_job, job_workload)
            
            # Calculate completion time for each job
            if i in self.completed_jobs:
                completion_times[i] = self.job_completion_times[i]
            # If job is not completed, estimate completion time
            else:
                #job_ops = range(i * self.number_of_machines, (i + 1) * self.number_of_machines)
                #scheduled_ops = [op for op in job_ops if self.finished_mark[op // self.number_of_machines, op % self.number_of_machines] == 1]
                last_scheduled_op = max((op for op in range(i * self.number_of_machines, (i + 1) * self.number_of_machines) if self.finished_mark[i, op % self.number_of_machines] == 1), default=None)
                
                if last_scheduled_op is not None:
                    last_completion = self.temp1[i, last_scheduled_op % self.number_of_machines]
                    remaining_time = sum(self.dur[i, j] for j in range((last_scheduled_op % self.number_of_machines) + 1, self.number_of_machines))
                    completion_times[i] = last_completion + remaining_time
                # If no operations are scheduled, estimate completion time as sum of last scheduled operation and remaining operations
                #if scheduled_ops:
                #    last_op = max(scheduled_ops)
                #    last_completion = self.temp1[last_op // self.number_of_machines, last_op % self.number_of_machines]
                #    remaining_time = sum(self.dur[i, j] for j in range((last_op % self.number_of_machines) + 1, self.number_of_machines))
                #    completion_times[i] = last_completion + remaining_time
                # Else, estimate completion time as sum of all operations duration
                else:
                    completion_times[i] = sum(self.dur[i, :])
                    
        # Phase 2: Scale completion times based on workload
        #avg_workload = total_workload / self.number_of_jobs
        #scaling_factor = self.LBs.max() / max(completion_times.max(), 1)  # Use makespan for scaling
        
        # Phase 3: Apply weights and scale
        #if self.weights is None:
        #    # For unweighted case, return makespan
        #    return self.LBs.max()
        
        # Calculate weighted sum with scaling
        #weighted_sum = 0
        #for i in range(self.number_of_jobs):
        #    # Scale completion time relative to job's workload proportion
        #    workload_proportion = np.sum(self.dur[i, :]) / avg_workload
        #    scaled_completion = completion_times[i] * scaling_factor * workload_proportion
        #    weighted_sum += self.weights[i] * scaled_completion
            
        # Normalize weights once
        normalized_weights = self.weights / np.sum(self.weights)
        # Use normalized weights throughout calculation
        weighted_sum = np.sum(normalized_weights * completion_times)
        
        return weighted_sum

