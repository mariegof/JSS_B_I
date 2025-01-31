"""
Comprehensive validation metrics and analysis for JSSP solutions.
Handles both makespan and weighted completion time objectives.
"""
import logging
from pathlib import Path
import numpy as np
import torch

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.src.validation import validate

class ValidationMetrics:
    def __init__(self, env_params=None):
        """Initialize validation metrics tracker"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Load parameters if not provided
        if env_params is None:
            base_path = Path(__file__).resolve().parents[3]
            param_file = str(base_path / "configs/L2D.toml")
            parameters = load_parameters(param_file)
            self.env_params = parameters["env_parameters"]
        else:
            self.env_params = env_params

    def analyze_makespan(self, objectives, instance_data):
        """Analyze makespan-specific metrics"""
        return {
            'makespan_mean': np.mean(objectives),
            'makespan_std': np.std(objectives),
            'makespan_min': np.min(objectives),
            'makespan_max': np.max(objectives),
            'critical_path_ratio': self._calculate_critical_path_ratio(objectives, instance_data)
        }
    
    def analyze_weighted_completion(self, objectives, weights, instance_data):
        """
        Analyze weighted completion time metrics
        
        Args:
            objectives: Array of final objectives for each instance
            weights: Array of job weights
            instance_data: List of validation instances
            
        Returns:
            dict: Dictionary of computed metrics
        """
        return {
            'weighted_completion_mean': np.mean(objectives),
            'weighted_completion_std': np.std(objectives),
            'weight_completion_correlation': self._calculate_weight_completion_correlation(
                objectives, weights, instance_data
            ),
            'priority_satisfaction_rate': self._calculate_priority_satisfaction(
                objectives, weights, instance_data
            )
        }
    
    def evaluate_model(self, model, validation_set, weights=None):
        """
        Comprehensive model evaluation with detailed metrics
        
        Args:
            model: Policy model to evaluate
            validation_set: List of validation instances
            weights: Optional job weights for weighted completion time objective
            
        Returns:
            dict: Dictionary of computed metrics
        """
        # Run validation
        objectives = validate(validation_set, model.policy, weights)
        
        # Basic statistics
        metrics = {
            'mean_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'min_objective': np.min(objectives),
            'max_objective': np.max(objectives)
        }
        
        # Add objective-specific metrics
        if weights is None:
            metrics.update(self.analyze_makespan(objectives, validation_set))
            logging.info("Computed makespan metrics")
        else:
            metrics.update(self.analyze_weighted_completion(objectives, weights, validation_set))
            logging.info("Computed weighted completion time metrics")
        
        return metrics

    def _calculate_critical_path_ratio(self, objectives, instance_data):
        """Calculate ratio of makespan to critical path lower bound"""
        critical_paths = []
        for instance in instance_data:
            # Calculate simple critical path (longest path of processing times)
            critical_path = np.max(np.sum(instance[0], axis=1))
            critical_paths.append(critical_path)
        
        ratios = objectives / np.array(critical_paths)
        return np.mean(ratios)

    def _calculate_weight_completion_correlation(self, objectives, weights, instance_data):
        """
        Calculate correlation between weights and completion times
        
        Args:
            objectives: Array of completion times for each instance
            weights: Array of job weights
            instance_data: List of problem instances
            
        Returns:
            float: Average correlation coefficient (-1 to 1)
        """
        correlations = []
        
        for instance_idx, data in enumerate(instance_data):
            n_jobs = data[0].shape[0]
            job_completion_times = np.zeros(n_jobs)
            
            # Calculate completion time for each job
            for job_id in range(n_jobs):
                job_ops = data[0][job_id, :]
                job_completion_times[job_id] = np.sum(job_ops)
            
            # Normalize values
            norm_completion_times = job_completion_times / np.mean(job_completion_times)
            norm_weights = weights / np.mean(weights)
            
            # Calculate correlation if there's variance in both arrays
            if np.std(norm_completion_times) > 0 and np.std(norm_weights) > 0:
                correlation = np.corrcoef(norm_weights, norm_completion_times)[0,1]
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0

    def _calculate_priority_satisfaction(self, objectives, weights, instance_data):
        """
        Calculate how well priority (weight) order is respected
        
        Args:
            objectives: Array of completion times for each instance
            weights: Array of job weights
            instance_data: List of problem instances
            
        Returns:
            float: Average priority satisfaction rate (0-1)
        """
        if len(weights) <= 1:
            return 1.0  # Perfect satisfaction for single job
            
        satisfaction_rates = []
        weight_order = np.argsort(-weights)  # Descending order of weights
        
        for instance_idx, data in enumerate(instance_data):
            # Get job completion times for this instance
            n_jobs = data[0].shape[0]  # Number of jobs from duration matrix shape
            job_completion_times = np.zeros(n_jobs)
            
            # Calculate completion time for each job
            for job_id in range(n_jobs):
                job_ops = data[0][job_id, :]  # Processing times for this job
                job_completion_times[job_id] = np.sum(job_ops)  # Simple sum for now
            
            correct_pairs = 0
            total_pairs = 0
            
            # Compare all pairs of jobs
            for i in range(len(weight_order)):
                for j in range(i + 1, len(weight_order)):
                    if weights[weight_order[i]] > weights[weight_order[j]]:
                        total_pairs += 1
                        # Check if higher weight job completes earlier
                        if job_completion_times[weight_order[i]] < job_completion_times[weight_order[j]]:
                            correct_pairs += 1
            
            if total_pairs > 0:
                satisfaction_rates.append(correct_pairs / total_pairs)
                
        return np.mean(satisfaction_rates) if satisfaction_rates else 1.0