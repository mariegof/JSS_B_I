"""
Test Environment for Job Shop Scheduling Problem (JSSP)

Goals:
- Provide a testing environment for evaluating trained JSSP policies
- Manage state transitions and reward calculations during testing/evaluation
- Track job completion and machine utilization metrics

Inputs:
- n_j (int): Number of jobs in the problem instance
- n_m (int): Number of machines in the problem instance
- JSM_env (JobShop): JobShop environment instance containing job and machine definitions

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

import numpy as np

from scheduling_environment.jobShop import JobShop
from solution_methods.helper_functions import load_parameters

base_path = Path(__file__).resolve().parents[3]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/L2D.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameters"]
model_parameters = parameters["network_parameters"]
train_parameters = parameters["train_parameters"]
test_parameters = parameters["test_parameters"]


class NipsJSPEnv_test():
    def __init__(self, n_j: int, n_m: int, weights: np.ndarray = None):

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.JobShopModule: JobShop = None
        # Track if we're using weights and completion times
        self.job_completion_times = np.zeros(self.number_of_jobs)
        self.completed_jobs = set()
        self.weights = weights
        # If weights are provided, ensure they're the correct shape
        if self.weights is not None:
            assert self.weights.shape == (n_j,), "Weights array must have shape (n_j,)"

    def reset(self, JSM_env: JobShop):
        self.JobShopModule = JSM_env
        self.JobShopModule.reset()

        self.step_count = 0
        
        # Reset job completion tracking
        self.job_completion_times = np.zeros(self.number_of_jobs)
        self.completed_jobs = set()

        # record action history
        self.partial_sol_sequeence = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.JSM_adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.JSM_LBs = np.zeros((len(self.JobShopModule.jobs), len(self.JobShopModule.machines)), dtype=np.single)
        for i in range(len(self.JobShopModule.jobs)):
            for j in range(len(self.JobShopModule.machines)):
                if j == 0:
                    self.JSM_LBs[i, j] = list(self.JobShopModule.jobs[i].operations[j].processing_times.values())[0]
                else:
                    self.JSM_LBs[i, j] = self.JSM_LBs[i, j-1] + list(self.JobShopModule.jobs[i].operations[j].processing_times.values())[0]

        self.JSM_max_endTime = self.JSM_LBs.max() if not env_parameters["init_quality_flag"] else 0
        self.JSM_finished_mark = np.zeros_like(self.JSM_LBs, dtype=np.single)
        self.initQuality = self.JSM_LBs.max() if not env_parameters["init_quality_flag"] else 0
        fea = np.concatenate((self.JSM_LBs.reshape(-1, 1) / env_parameters["et_normalize_coef"], self.JSM_finished_mark.reshape(-1, 1)), axis=1)
        
        if self.JobShopModule.is_weighted:
            # Initialize weight features with zeros
            weight_features = np.zeros((self.number_of_tasks, 1), dtype=np.float32)
            # Assign weights only to terminal operations
            for op in self.JobShopModule.get_last_operations():
                weight_features[op.operation_id] = op.job.weight
            # Add weight features as a new column in the feature matrix
            fea = np.concatenate((fea, weight_features), axis=1)
            
        # initialize feasible omega
        self.JSM_omega = self.first_col.astype(np.int64)
        # initialize mask
        self.JSM_mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        return self.JSM_adj, fea, self.JSM_omega, self.JSM_mask

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        # Calculate objective before action
        prev_objective = (self.calculate_weighted_completion_time() 
                     if self.JobShopModule.is_weighted 
                     else self.JSM_LBs.max())
        
        ope_to_schedule = self.JobShopModule.get_operation(action)

        if len(ope_to_schedule.scheduling_information) == 0:
            self.partial_sol_sequeence.append(action)
            self.step_count += 1

            assigned_mach = list(ope_to_schedule.processing_times.keys())[0]
            process_time = list(ope_to_schedule.processing_times.values())[0]
            self.JobShopModule.schedule_operation_on_machine(ope_to_schedule, assigned_mach, process_time)
            job_id = ope_to_schedule.job_id
            ope_idx_in_job = ope_to_schedule.job.operations.index(ope_to_schedule)
            self.JSM_finished_mark[job_id, ope_idx_in_job] = 1

            self.JSM_adj[ope_to_schedule.operation_id] = 0
            self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id] = 1
            if ope_idx_in_job != 0:
                self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id-1] = 1
            machine = self.JobShopModule.get_machine(assigned_mach)
            ope_idx_in_machine = machine.scheduled_operations.index(ope_to_schedule)
            if ope_idx_in_machine > 0:
                prede_ope_id = machine.scheduled_operations[ope_idx_in_machine - 1].operation_id
                self.JSM_adj[ope_to_schedule.operation_id, prede_ope_id] = 1
            if ope_idx_in_machine < len(machine.scheduled_operations) - 1:
                succe_ope_id = machine.scheduled_operations[ope_idx_in_machine + 1].operation_id
                self.JSM_adj[succe_ope_id, ope_to_schedule.operation_id] = 1
                if ope_idx_in_machine > 0:
                    self.JSM_adj[succe_ope_id, prede_ope_id] = 0

            if action not in self.last_col:
                self.JSM_omega[job_id] += 1
            else:
                self.job_completion_times[job_id] = ope_to_schedule.scheduling_information.get('end_time')
                self.completed_jobs.add(job_id)
                self.JSM_mask[job_id] = 1

            self.JSM_LBs[job_id, ope_idx_in_job] = ope_to_schedule.scheduling_information.get('end_time')
            for i in range(ope_idx_in_job + 1, len(ope_to_schedule.job.operations)):
                next_ope = ope_to_schedule.job.operations[i]
                pure_process_time = list(next_ope.processing_times.values())[0]
                self.JSM_LBs[job_id, i] = self.JSM_LBs[job_id, i-1] + pure_process_time

        # prepare for return
        feature_JSM = np.concatenate((
            self.JSM_LBs.reshape(-1, 1) / env_parameters["et_normalize_coef"], # CLB
            self.JSM_finished_mark.reshape(-1, 1) # I(O, st)
            ), axis=1)
        
        if self.JobShopModule.is_weighted:
            weight_features = np.zeros((self.number_of_tasks, 1), dtype=np.float32)
            last_ops = self.JobShopModule.get_last_operations()
            for op in last_ops:
                weight_features[op.operation_id] = op.job.weight
                
            feature_JSM = np.concatenate((feature_JSM, weight_features), axis=1) # wi when weights exist
        
        # Calculate new objective and reward
        current_objective = self.calculate_weighted_completion_time()
        # BEFORE: reward_JSM = - (self.JSM_LBs.max() - self.JSM_max_endTime)
        reward_JSM = -(current_objective - prev_objective)

        if reward_JSM == 0:
            reward_JSM = env_parameters["rewardscale"]
            self.posRewards += reward_JSM

        # BEFORE: self.JSM_max_endTime = self.JSM_LBs.max()
        # Update max end time based on objective
        self.JSM_max_endTime = current_objective

        return self.JSM_adj, feature_JSM, reward_JSM, self.done(), self.JSM_omega, self.JSM_mask
    
    def calculate_weighted_completion_time(self):
        """
        Calculate weighted sum of completion times: Σ(w_j * C_j)
        
        For the testing environment, we can leverage the JobShop environment
        to get precise completion times for scheduled operations and make
        informed estimates for unscheduled operations.
        
        Returns:
            float: Weighted sum of completion times if the instance is weighted, otherwise returns makespan
        """
        # Check if this is a weighted instance by looking at job weights
        if not self.JobShopModule.is_weighted:
            return self.JSM_LBs.max()
        
        total_weight = 0  # Initialize total weight
        weighted_sum = 0  # Initialize weighted sum
            
        for job_id, job in enumerate(self.JobShopModule.jobs):
            scheduled_ops = job.scheduled_operations
            
            if len(scheduled_ops) == len(job.operations):
                # All operations scheduled: use actual completion time
                completion_time = max(op.scheduled_end_time for op in scheduled_ops)
            elif scheduled_ops:
                # Partially scheduled: use last scheduled operation plus estimates
                last_scheduled = max(scheduled_ops, key=lambda op: op.scheduled_end_time)
                last_scheduled_idx = job.operations.index(last_scheduled)
                
                # Start from last known completion time
                completion_time = last_scheduled.scheduled_end_time
                
                # Estimate remaining operations
                for operation in job.operations[last_scheduled_idx + 1:]:
                    # Use minimum processing time among available machines
                    completion_time += min(operation.processing_times.values())
                            
            else:
                # If no operations scheduled, use lower bound from JSM_LBs
                completion_time = self.JSM_LBs[job_id, -1]
                
            # Update weighted sum and total weight
            total_weight += job.weight
            weighted_sum += job.weight * completion_time
        
        # Normalize at the end (equivalent to normalizing per job)
        return weighted_sum / total_weight if total_weight > 0 else 0