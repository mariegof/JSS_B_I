import json
import sys
from pathlib import Path
# Add project root to Python path
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))
from plotting.drawer import create_colormap
from matplotlib import pyplot as plt
import numpy as np
import logging

class SimpleOperation:
    def __init__(self, job_id, op_id, machine_id, processing_time):
        self.job_id = job_id
        self.op_id = op_id
        self.machine_id = machine_id
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None

class SimpleJob:
    def __init__(self, job_id, weight, operations):
        self.job_id = job_id
        self.weight = weight
        self.operations = operations

class SimpleMachine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.schedule = []

    def add_operation(self, operation, start_time):
        operation.start_time = start_time
        operation.end_time = start_time + operation.processing_time
        self.schedule.append(operation)

class SimpleJobShop:
    def __init__(self, jobs, machines):
        self.jobs = jobs
        self.machines = machines
        
    # PDR 1: Choose the job with the highest weight
    def schedule_weighted_jobs(self):
        # Sort jobs by weight (descending)
        sorted_jobs = sorted(self.jobs, key=lambda x: x.weight, reverse=True)
        
        for job in sorted_jobs:
            for op in job.operations:
                machine = self.machines[op.machine_id]
                start_time = self.find_earliest_start_time(machine, op)
                machine.add_operation(op, start_time)
                
    # PDR 2: Choose the job with the highest weight and the shortest remaining processing time (WSRPT)
    def schedule_weighted_jobs_wsrpt(self):
        remaining_operations = [job.operations.copy() for job in self.jobs]
        
        while any(remaining_operations):
            # Calculate priority for each job
            priorities = []
            for job, ops in zip(self.jobs, remaining_operations):
                if ops:
                    remaining_time = sum(op.processing_time for op in ops)
                    priority = job.weight * remaining_time
                    priorities.append((job, ops[0], priority))
            
            # Sort by priority (descending)
            priorities.sort(key=lambda x: x[2], reverse=True)
            
            # Schedule the highest priority operation
            job, operation, _ = priorities[0]
            machine = self.machines[operation.machine_id]
            start_time = self.find_earliest_start_time(machine, operation)
            machine.add_operation(operation, start_time)
            
            # Remove the scheduled operation from remaining_operations
            remaining_operations[self.jobs.index(job)].pop(0)

    def find_earliest_start_time(self, machine, operation):
        if not machine.schedule:
            return 0
        last_end_time = max(op.end_time for op in machine.schedule)
        return max(last_end_time, self.get_previous_operation_end_time(operation))

    def get_previous_operation_end_time(self, operation):
        job = next(job for job in self.jobs if job.job_id == operation.job_id)
        op_index = job.operations.index(operation)
        if op_index == 0:
            return 0
        previous_op = job.operations[op_index - 1]
        return previous_op.end_time if previous_op.end_time is not None else 0

    def calculate_weighted_completion_time(self):
        total = 0
        for job in self.jobs:
            completion_time = max(op.end_time for op in job.operations)
            total += job.weight * completion_time
        return total

def draw_gantt_chart(job_shop, save=False, filename=None):
    fig, ax = plt.subplots()
    colormap = create_colormap()

    for machine_id, machine in job_shop.machines.items():
        machine_operations = sorted(machine.schedule, key=lambda op: op.start_time)
        for operation in machine_operations:
            operation_start = operation.start_time
            operation_end = operation.end_time
            operation_duration = operation_end - operation_start
            operation_label = f"{operation.op_id}"

            # Set color based on job ID
            color_index = int(operation.job_id[1:]) - 1  # Assuming job_id is like "J1", "J2", etc.
            color = colormap(color_index % colormap.N)

            ax.broken_barh(
                [(operation_start, operation_duration)],
                (int(machine_id[1:]) - 0.4, 0.8),  # Assuming machine_id is like "M1", "M2", etc.
                facecolors=color,
                edgecolor='black'
            )

            middle_of_operation = operation_start + operation_duration / 2
            ax.text(
                middle_of_operation,
                int(machine_id[1:]),
                operation_label,
                ha='center',
                va='center',
                fontsize=8
            )

    fig = ax.figure
    fig.set_size_inches(16, 8)

    ax.set_yticks(range(1, len(job_shop.machines) + 1))
    ax.set_yticklabels([f'M{i}' for i in range(1, len(job_shop.machines) + 1)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Job Shop Scheduling Gantt Chart')
    ax.grid(True)

    if save:
        plt.savefig(filename or "job_shop_schedule.png")
    
    return plt

def create_control_instance(weights):
    processing_times = np.array([
        [4, 5, 3, 2, 4, 1],
        [3, 4, 2, 3, 2, 3],
        [5, 2, 4, 3, 3, 2],
        [2, 3, 3, 4, 2, 3],
        [4, 2, 3, 2, 4, 3],
        [3, 3, 2, 4, 3, 2]
    ])
    
    machine_order = np.array([
        [3, 1, 2, 4, 6, 5],
        [2, 3, 5, 1, 4, 6],
        [1, 2, 4, 5, 3, 6],
        [4, 1, 5, 2, 6, 3],
        [5, 6, 1, 3, 2, 4],
        [6, 4, 2, 1, 3, 5]
    ])
    
    #weights = np.array([3.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    
    jobs = []
    for job_idx in range(6):
        operations = []
        for op_idx in range(6):
            machine_id = f"M{machine_order[job_idx][op_idx]}"
            proc_time = processing_times[job_idx][op_idx]
            operations.append(SimpleOperation(f"J{job_idx+1}", f"O{job_idx*6+op_idx+1}", machine_id, proc_time))
        jobs.append(SimpleJob(f"J{job_idx+1}", weights[job_idx], operations))
    
    machines = {f"M{i+1}": SimpleMachine(f"M{i+1}") for i in range(6)}
    
    return SimpleJobShop(jobs, machines)

def solve_weighted_instance(weights, name):
    results = {}
    for method in ['HWF', 'WSRPT']:
        job_shop = create_control_instance(weights)
        
        if method == 'HWF':
            job_shop.schedule_weighted_jobs()
        else:
            job_shop.schedule_weighted_jobs_wsrpt()
        
        stats = {"job_stats": [], "overall_stats": {}}
        
        logging.info(f"\nSolving instance with weights: {weights} using {method}")
        logging.info("Job completion times:")
        for job in job_shop.jobs:
            completion_time = max(op.end_time for op in job.operations)
            weighted_completion = job.weight * completion_time
            log_msg = f"Job {job.job_id}: Weight = {job.weight}, Completion Time = {completion_time}, Weighted Completion = {weighted_completion}"
            logging.info(log_msg)
            stats["job_stats"].append({
                "job_id": job.job_id,
                "weight": float(job.weight),
                "completion_time": int(completion_time),
                "weighted_completion": float(weighted_completion)
            })
        
        weighted_completion_time = job_shop.calculate_weighted_completion_time()
        logging.info(f"Total weighted completion time: {weighted_completion_time}")
        
        makespan = max(max(op.end_time for op in job.operations) for job in job_shop.jobs)
        logging.info(f"Makespan: {makespan}")
        
        total_processing_time = sum(op.processing_time for job in job_shop.jobs for op in job.operations)
        avg_utilization = total_processing_time / (len(job_shop.machines) * makespan) * 100
        logging.info(f"Average machine utilization: {avg_utilization:.2f}%")
        
        stats["overall_stats"] = {
            "total_weighted_completion_time": float(weighted_completion_time),
            "makespan": int(makespan),
            "avg_machine_utilization": float(avg_utilization)
        }
        
        # Save stats to a JSON file
        charts_dir = Path("control_charts")
        charts_dir.mkdir(exist_ok=True)
        with open(charts_dir / f"{name}_{method}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        # Generate and save Gantt chart
        plt = draw_gantt_chart(job_shop, save=True, filename=charts_dir / f"{name}_{method}_gantt.png")
        plt.close()
        
        results[method] = stats
    
    return results

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    weight_sets = {
        "original": [3.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        "same": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "uniform": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }
    
    all_results = {}
    for name, weights in weight_sets.items():
        all_results[name] = solve_weighted_instance(weights, name)
