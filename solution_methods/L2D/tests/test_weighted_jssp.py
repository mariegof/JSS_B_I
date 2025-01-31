"""
Test module for evaluating JSSP implementation with and without weights.
Provides comprehensive comparison of makespan vs weighted completion time objectives.
"""
import unittest
import logging
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime

from solution_methods.L2D.src.JSSP_Env import SJSSP
from solution_methods.L2D.data.instance_generator import uniform_instance_generator
from solution_methods.L2D.src.validation import validate
from solution_methods.L2D.tests.validation_metrics import ValidationMetrics
from solution_methods.L2D.src.PPO_model import PPO
from solution_methods.helper_functions import load_parameters

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

class TestJSSPObjectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test environment, logging, and result storage"""
        # Setup logging with detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create results directory
        cls.results_dir = Path(__file__).parent / 'test_results'
        cls.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load parameters
        base_path = Path(__file__).resolve().parents[3]
        cls.parameters = load_parameters(str(base_path / "configs/L2D.toml"))
        
        # Test configurations
        cls.configs = {
            'small': {'n_j': 6, 'n_m': 4},
            'medium': {'n_j': 10, 'n_m': 10},
            'large': {'n_j': 15, 'n_m': 15}
        }
        
        # Define weight scenarios for testing
        cls.weight_scenarios = {
            'uniform': lambda n: np.ones(n),
            'linear': lambda n: np.linspace(1, 2, n),
            'exponential': lambda n: 2.0 ** np.arange(n)
        }
        
        # Number of instances per configuration
        cls.n_instances = 5
        
        # Initialize validation metrics tracker
        cls.metrics_tracker = ValidationMetrics(cls.parameters["env_parameters"])
        
        # Store results
        cls.test_results = {}

    def generate_test_instances(self, size_config):
        """Generate test instances with proper dimensions"""
        n_j = size_config['n_j']
        n_m = size_config['n_m']
        
        instances = []
        for _ in range(self.n_instances):
            dur_matrix = np.random.randint(1, 10, size=(n_j, n_m)).astype(np.float32)
            machine_matrix = np.tile(np.arange(1, n_m + 1), (n_j, 1))
            instances.append((dur_matrix, machine_matrix))
            
        return instances

    def init_model(self, n_j, n_m):
        """Initialize PPO model with correct dimensions for testing"""
        model = PPO(
            lr=self.parameters["train_parameters"]["lr"],
            gamma=self.parameters["train_parameters"]["gamma"],
            k_epochs=self.parameters["train_parameters"]["k_epochs"],
            eps_clip=self.parameters["train_parameters"]["eps_clip"],
            n_j=n_j,
            n_m=n_m,
            num_layers=self.parameters["network_parameters"]["num_layers"],
            neighbor_pooling_type=self.parameters["network_parameters"]["neighbor_pooling_type"],
            input_dim=2,  # Original input dimension: [completion_time, finished_flag]
            hidden_dim=self.parameters["network_parameters"]["hidden_dim"],
            num_mlp_layers_feature_extract=self.parameters["network_parameters"]["num_mlp_layers_feature_extract"],
            num_mlp_layers_actor=self.parameters["network_parameters"]["num_mlp_layers_actor"],
            hidden_dim_actor=self.parameters["network_parameters"]["hidden_dim_actor"],
            num_mlp_layers_critic=self.parameters["network_parameters"]["num_mlp_layers_critic"],
            hidden_dim_critic=self.parameters["network_parameters"]["hidden_dim_critic"]
        )
        
        # Set model to evaluation mode
        model.policy.eval()
        model.policy_old.eval()
        
        return model

    def test_makespan_baseline(self):
        """Test baseline makespan optimization without weights"""
        logging.info("Testing makespan optimization (baseline)")
        
        for size_name, config in self.configs.items():
            logging.info(f"Testing {size_name} configuration: {config['n_j']}x{config['n_m']}")
            
            test_instances = self.generate_test_instances(config)
            model = self.init_model(config['n_j'], config['n_m'])
            metrics = self.metrics_tracker.evaluate_model(model, test_instances)
            
            self.test_results[f'makespan_{size_name}'] = metrics
            
            # Log results
            logging.info(f"Makespan metrics for {size_name}:")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.3f}")
            
            # Updated assertions for positive objectives
            self.assertGreater(metrics['mean_objective'], 0, 
                            "Mean objective should be positive")
            self.assertGreaterEqual(metrics['critical_path_ratio'], 1.0, 
                                "Critical path ratio should be >= 1")
            self.assertGreaterEqual(metrics['min_objective'], 0, 
                                "Minimum objective should be non-negative")

    def test_weighted_completion(self):
        """Test weighted completion time optimization with balanced variation checks"""
        logging.info("Testing weighted completion time optimization")
        
        for size_name, config in self.configs.items():
            logging.info(f"Testing {size_name} configuration with weights")
            test_instances = self.generate_test_instances(config)
            
            for weight_type, weight_fn in self.weight_scenarios.items():
                logging.info(f"Testing {weight_type} weights")
                weights = weight_fn(config['n_j'])
                
                model = self.init_model(config['n_j'], config['n_m'])
                metrics = self.metrics_tracker.evaluate_model(
                    model, test_instances, weights=weights)
                
                self.test_results[f'weighted_{size_name}_{weight_type}'] = metrics
                
                logging.info(f"Weighted metrics for {size_name} ({weight_type}):")
                for metric, value in metrics.items():
                    logging.info(f"{metric}: {value:.3f}")
                
                if weight_type == 'uniform':
                    makespan_metrics = self.test_results[f'makespan_{size_name}']
                    
                    # Calculate relative difference as percentage
                    relative_diff = abs(metrics['mean_objective'] - makespan_metrics['mean_objective']) / makespan_metrics['mean_objective'] * 100
                    
                    # Allow for larger variation in smaller problems
                    allowed_percentage = 40 if config['n_j'] <= 6 else 30
                    
                    logging.info(f"Relative difference: {relative_diff:.1f}% (allowed: {allowed_percentage}%)")
                    
                    self.assertLess(
                        relative_diff,
                        allowed_percentage,
                        f"Relative difference ({relative_diff:.1f}%) exceeds allowed threshold ({allowed_percentage}%)"
                    )
                    
                    # Check standard deviation ratio with balanced thresholds
                    std_ratio = metrics['std_objective'] / makespan_metrics['std_objective']
                    min_std_ratio = 0.4 if config['n_j'] <= 6 else 0.5
                    max_std_ratio = 2.5 if config['n_j'] <= 6 else 2.0
                    
                    logging.info(f"Standard deviation ratio: {std_ratio:.2f} (allowed: {min_std_ratio}-{max_std_ratio})")
                    
                    self.assertGreater(std_ratio, min_std_ratio,
                                    f"Variation in weighted completion times should be at least {min_std_ratio} times makespan std dev")
                    self.assertLess(std_ratio, max_std_ratio,
                                f"Variation in weighted completion times should be at most {max_std_ratio} times makespan std dev")
    
    def tearDown(self):
        """Save test results and generate report"""
        # Convert results to JSON-serializable format
        serializable_results = convert_to_serializable(self.test_results)
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f'test_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logging.info(f"Test results saved to {results_file}")

if __name__ == '__main__':
    unittest.main(verbosity=2)