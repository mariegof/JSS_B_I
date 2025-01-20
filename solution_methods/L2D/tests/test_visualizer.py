import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from pathlib import Path
import json

class TestVisualizer:
    def __init__(self):
        # Create results directory in L2D/tests/test_results
        current_dir = Path(__file__).resolve().parent
        self.output_dir = current_dir / 'test_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = self.output_dir / 'test_results.log'
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logging.info(f"Test results will be saved to {self.output_dir}")

    def plot_completion_times(self, completion_times, weights, title="Job Completion Times"):
        """Plot job completion times with their weights"""
        plt.figure(figsize=(10, 6))
        jobs = range(len(completion_times))
        
        # Create bar chart
        plt.bar(jobs, completion_times, alpha=0.5, label='Completion Time', color='blue')
        plt.bar(jobs, completion_times * weights, alpha=0.5, label='Weighted Completion Time', color='red')
        
        plt.xlabel('Job ID')
        plt.ylabel('Time')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.output_dir / 'completion_times.png'
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved completion times plot to {plot_path}")

    def save_test_statistics(self, env, test_name="weighted_jssp_test"):
        """Save test statistics to JSON"""
        stats = {
            "test_name": test_name,
            "number_of_jobs": env.number_of_jobs,
            "number_of_machines": env.number_of_machines,
            "weights": env.weights.tolist(),
            "job_completion_times": env.job_completion_times.tolist(),
            "weighted_completion_times": (env.weights * env.job_completion_times).tolist(),
            "total_weighted_completion_time": float(np.sum(env.weights * env.job_completion_times)),
        }
        
        # Log statistics
        logging.info(f"\nTest Statistics for {test_name}:")
        logging.info(f"Total Weighted Completion Time: {stats['total_weighted_completion_time']:.2f}")
        logging.info("Individual Job Statistics:")
        for i in range(env.number_of_jobs):
            logging.info(f"Job {i}: CT={stats['job_completion_times'][i]:.2f}, "
                        f"Weight={stats['weights'][i]:.2f}, "
                        f"Weighted CT={stats['weighted_completion_times'][i]:.2f}")
        
        # Save to JSON
        json_path = self.output_dir / 'test_statistics.json'
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logging.info(f"Saved test statistics to {json_path}")
        
        return stats