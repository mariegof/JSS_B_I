import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils import EzPickle

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.src.permissibleLS import permissibleLeftShift
from solution_methods.L2D.training_data.instance_generator import override
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
    def __init__(self,
                 n_j,
                 n_m, 
                 weights=None):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        self.weights = weights
        self.max_normalized_weights = (weights / np.max(weights) if weights is not None else None) # for NN features because we want them on a similar scale as our other features (which are also typically in [0, 1])
        self.job_completion_times = np.zeros(self.number_of_jobs, dtype=np.float32)  # Track job completion times for weighted objective
        self.completed_jobs = set()  # Tracks completed jobs
        # Initialize reward function

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # Execute action if not already processed, redundant action makes no effect
        if action not in self.partial_sol_sequeence:

            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col] # duration of the operation
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift allows the operation to be rescheduled to earlier time
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                # Case 1: Operation is not the last one in its job
                self.omega[action // self.number_of_machines] += 1
            else:
                # Case 2: Operation is the last one in its job
                if self.weights is not None:
                    self.job_completion_times[row] = startTime_a + dur_a # Track completion time
                    self.completed_jobs.add(row) # Mark job as completed (for weighted objective)
                self.mask[action // self.number_of_machines] = 1 # Update mask, setting job as completed (for GNN)

            # Update operation times
            self.temp1[row, col] = startTime_a + dur_a # record the end time of the operation

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp) # update lower bound of end time

            # adj matrix / disjunctive graph updates
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0

        # prepare for return: feature and reward calculation
        # the feature combines normalized lower bounds (completion time estimates) and finished mark
        fea = np.concatenate((self.LBs.reshape(-1, 1)/env_parameters["et_normalize_coef"], # CLB
                              self.finished_mark.reshape(-1, 1)), axis=1) # I(O, st)
        # Add updated operation priorities if this is a weighted instance
        if self.weights is not None:
            # We reuse the same priority calculation as in reset
            weight_features = self._calculate_operation_priorities()
            fea = np.concatenate((fea, weight_features), axis=1)

        # Calculate reward based on instance type
        if self.weights is not None:
            current_objective = self.calculate_objective()
            reward = -(current_objective - self.best_weighted_objective)
            if reward == 0:
                reward = env_parameters["rewardscale"]
                self.posRewards += reward
            self.best_weighted_objective = current_objective
        else:
            current_makespan = self.LBs.max()
            reward = -(current_makespan - self.max_endTime)
            if reward == 0:
                reward = env_parameters["rewardscale"]
                self.posRewards += reward
            self.max_endTime = current_makespan
       
        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):

        self.step_count = 0
        self.m = data[1] # instead of -1, we use 1 in case we receive weighted data (3D)
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
            
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0
        # Reset job completion tracking
        self.job_completion_times = np.zeros(self.number_of_jobs)
        self.completed_jobs = set()

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
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)
        
        # Handle weighted vs unweighted instances differently
        if len(data) == 3:  # Weighted instance
            self.weights = data[2][:, -1]
            self.max_normalized_weights = self.weights / np.max(self.weights)
            
            # For weighted instances, initQuality is the theoretical minimum weighted sum
            # We use the last column of LBs which contains the minimum completion times
            if not env_parameters["init_quality_flag"]:
                minimum_completion_times = self.LBs[:, -1]  # Get completion time for each job
                self.initQuality = np.dot(self.weights, minimum_completion_times)
            else:
                self.initQuality = 0
                
            self.best_weighted_objective = self.initQuality
        else:  # Unweighted instance
            self.weights = None
            self.max_normalized_weights = None
            
            # For unweighted instances, initQuality is the theoretical minimum makespan
            self.initQuality = self.LBs.max() if not env_parameters["init_quality_flag"] else 0
            self.max_endTime = self.initQuality

        fea = np.concatenate((self.LBs.reshape(-1, 1)/env_parameters["et_normalize_coef"],
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        if self.weights is not None:
            # Add operation priorities based on weights and processing times
            weight_features = self._calculate_operation_priorities()
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
    
    def _calculate_operation_priorities(self):
        """
        Calculate priority features using simple NumPy operations.
        Each job's priority is its normalized weight divided by total processing time.
        """
        # Calculate priorities for each job
        total_processing_times = np.sum(self.dur, axis=1)
        job_priorities = self.max_normalized_weights / total_processing_times
        
        # Repeat each priority for all operations of the corresponding job
        return np.repeat(job_priorities, self.number_of_machines)[:, np.newaxis].astype(np.float32)
    
    def calculate_objective(self):
        """
        Calculate weighted sum of completion times: Σ(w_j * C_j)
        
        Returns:
            float: Weighted sum of completion times if weights present, 
                otherwise returns makespan
        """
        # Calculate raw completion times
        completion_times = np.zeros(self.number_of_jobs)

        for i in range(self.number_of_jobs):
            if i in self.completed_jobs:
                # Case 1: Job is complete - use actual completion time
                completion_times[i] = self.job_completion_times[i]
            else:
                # Case 2: Job is incomplete - need to estimate completion time
                # Find last scheduled operation
                scheduled_ops = self.finished_mark[i, :]
                last_scheduled_idx = np.where(scheduled_ops == 1)[0]

                if len(last_scheduled_idx) > 0:
                    # Some operations are scheduled
                    last_op_idx = last_scheduled_idx[-1]
                    last_completion = self.temp1[i, last_op_idx]
                    
                    # Calculate remaining time using vectorized sum
                    remaining_ops = slice(last_op_idx + 1, self.number_of_machines)
                    remaining_time = np.sum(self.dur[i, remaining_ops])
                    
                    completion_times[i] = last_completion + remaining_time
                else:
                    # No operations scheduled - use sum of all durations
                    completion_times[i] = np.sum(self.dur[i, :])

        # Calculate weighted sum
        weighted_sum = np.dot(self.weights, completion_times)

        return weighted_sum