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
        self.weights = weights
        self.max_normalized_weights = (weights / np.max(weights) if weights is not None else None)
        self.job_completion_times = np.zeros(self.number_of_jobs, dtype=np.float32)
        self.completed_jobs = set()

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

        # Initialize quality measures based on instance type
        if self.weights is not None:
            # For weighted instances: initQuality = Σ(w_j * LB_j)
            if not env_parameters["init_quality_flag"]:
                minimum_completion_times = self.JSM_LBs[:, -1]  # Last column contains completion times
                self.initQuality = np.dot(self.weights, minimum_completion_times)
                self.best_weighted_objective = self.initQuality
            else:
                self.initQuality = 0
                self.best_weighted_objective = 0
        else:
            # For makespan instances - original behavior
            self.initQuality = self.JSM_LBs.max() if not env_parameters["init_quality_flag"] else 0
            self.JSM_max_endTime = self.initQuality
        
        self.JSM_finished_mark = np.zeros_like(self.JSM_LBs, dtype=np.single)
        fea = np.concatenate((self.JSM_LBs.reshape(-1, 1) / env_parameters["et_normalize_coef"],
                              self.JSM_finished_mark.reshape(-1, 1)), axis=1)
        if self.weights is not None:
            # Add weight features using same method as JSSP_Env
            weight_features = self._calculate_operation_priorities()
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
        # Get operation to schedule based on action
        ope_to_schedule = self.JobShopModule.get_operation(action)

        if len(ope_to_schedule.scheduling_information) == 0: # If not already scheduled
            self.partial_sol_sequeence.append(action)
            self.step_count += 1
            
            # Schedule operation on machine
            assigned_mach = list(ope_to_schedule.processing_times.keys())[0]
            process_time = list(ope_to_schedule.processing_times.values())[0]
            self.JobShopModule.schedule_operation_on_machine(ope_to_schedule, assigned_mach, process_time)
            # Get job and operation indices
            job_id = ope_to_schedule.job_id
            ope_idx_in_job = ope_to_schedule.job.operations.index(ope_to_schedule)
            # Mark operation as finished
            self.JSM_finished_mark[job_id, ope_idx_in_job] = 1

            # Update graph connectivity (disjunctive graph)
            self.JSM_adj[ope_to_schedule.operation_id] = 0
            self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id] = 1
            if ope_idx_in_job != 0:
                self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id-1] = 1
                
            # Update machine connections
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

            # Update job tracking
            if action not in self.last_col:
                self.JSM_omega[job_id] += 1
            else:
                # For weighted instances, track job completion when last operation is scheduled
                if self.weights is not None:
                    self.job_completion_times[job_id] = ope_to_schedule.scheduling_information.get('end_time')
                    self.completed_jobs.add(job_id)
                self.JSM_mask[job_id] = 1

            # Update time estimates for current and remaining operations
            self.JSM_LBs[job_id, ope_idx_in_job] = ope_to_schedule.scheduling_information.get('end_time')
            for i in range(ope_idx_in_job + 1, len(ope_to_schedule.job.operations)):
                next_ope = ope_to_schedule.job.operations[i]
                pure_process_time = list(next_ope.processing_times.values())[0]
                self.JSM_LBs[job_id, i] = self.JSM_LBs[job_id, i-1] + pure_process_time

        # Prepare feature matrix for return
        feature_JSM = np.concatenate((self.JSM_LBs.reshape(-1, 1) / env_parameters["et_normalize_coef"],
                              self.JSM_finished_mark.reshape(-1, 1)), axis=1)
        
        # Add weight features for weighted instances
        if self.weights is not None:
            weight_features = self._calculate_operation_priorities()
            feature_JSM = np.concatenate((feature_JSM, weight_features), axis=1)
        
        # Calculate reward based on instance type
        if self.weights is not None:
            current_objective = self.calculate_objective()
            reward_JSM = -(current_objective - self.best_weighted_objective)
            if reward_JSM == 0:
                reward_JSM = env_parameters["rewardscale"]
                self.posRewards += reward_JSM
            self.best_weighted_objective = current_objective
        else:
            # Original makespan reward calculation
            reward_JSM = -(self.JSM_LBs.max() - self.JSM_max_endTime)
            if reward_JSM == 0:
                reward_JSM = env_parameters["rewardscale"]
                self.posRewards += reward_JSM
            self.JSM_max_endTime = self.JSM_LBs.max()

        return self.JSM_adj, feature_JSM, reward_JSM, self.done(), self.JSM_omega, self.JSM_mask
    
    def _calculate_operation_priorities(self):
        """
        Calculate priority features using simple NumPy operations, adapted for test environment.
        Each job's priority is its normalized weight divided by total processing time.
        Works with JSM environment structure for testing.
        """
        # For each job, get total processing time from its operations
        total_processing_times = np.array([
            sum(min(op.processing_times.values()) for op in self.JobShopModule.jobs[i].operations)
            for i in range(self.number_of_jobs)
        ])
        
        # Calculate priorities using same logic as training environment
        job_priorities = self.max_normalized_weights / total_processing_times
        
        # Repeat each priority for all operations of the corresponding job
        return np.repeat(job_priorities, self.number_of_machines)[:, np.newaxis].astype(np.float32)

    def calculate_objective(self):
        """Calculate the current objective value (makespan or weighted completion time).
    
        The objective is our unified measure of schedule quality:
        - For unweighted instances: returns makespan (maximum completion time)
        - For weighted instances: returns weighted sum of completion times
        
        Returns:
            float: Current objective value
        """
        # Calculate raw completion times
        completion_times = np.zeros(self.number_of_jobs)

        for i in range(self.number_of_jobs):
            if i in self.completed_jobs:
                # Case 1: Job is complete - use actual completion time
                completion_times[i] = self.job_completion_times[i]
            else:
                # Case 2: Job is incomplete - need to estimate completion time
                # Find last scheduled operation more efficiently
                scheduled_ops = self.JSM_finished_mark[i, :]
                last_scheduled_idx = np.where(scheduled_ops == 1)[0]

                if len(last_scheduled_idx) > 0:
                    # Some operations are scheduled
                    last_op_idx = last_scheduled_idx[-1]
                    last_completion = self.JSM_LBs[i, last_op_idx]
                    
                    # Calculate remaining time using vectorized sum
                    remaining_ops = slice(last_op_idx + 1, self.number_of_machines)
                    remaining_time = sum(
                        min(op.processing_times.values())  # Use minimum processing time
                        for op in self.JobShopModule.jobs[i].operations[remaining_ops]
                    )
                    
                    completion_times[i] = last_completion + remaining_time
                else:
                    # No operations scheduled - use lower bound
                    completion_times[i] = self.JSM_LBs[i, -1]

        # Calculate weighted sum using job weights
        weighted_sum = np.dot(self.weights, completion_times)
        
        return weighted_sum