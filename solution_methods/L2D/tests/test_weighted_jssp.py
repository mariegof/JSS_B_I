import unittest
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from solution_methods.L2D.tests.test_visualizer import TestVisualizer
from src.JSSP_Env import SJSSP
from src.PPO_model import PPO, Memory
from data.instance_generator import uniform_instance_generator

class TestWeightedJSSP(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.n_j = 3  # Small problem for testing
        self.n_m = 2
        self.weights = np.array([1.0, 2.0, 0.5])  # Test different priorities
        self.env = SJSSP(n_j=self.n_j, n_m=self.n_m, weights=self.weights)
        
        # Create a simple test instance
        self.test_instance = uniform_instance_generator(
            n_j=self.n_j,
            n_m=self.n_m,
            low=1,
            high=10
        )

    def test_env_initialization(self):
        """Test environment initialization with weights"""
        self.assertEqual(len(self.env.weights), self.n_j)
        self.assertEqual(self.env.weights[1], 2.0)  # Check high priority job
        self.assertEqual(self.env.number_of_jobs, self.n_j)
        self.assertEqual(self.env.number_of_machines, self.n_m)

    def test_job_completion_tracking(self):
        """Test job completion time tracking"""
        adj, fea, omega, mask = self.env.reset(self.test_instance)
        
        # Complete one job
        for _ in range(self.n_m):  # Complete all operations of first job
            action = omega[0]  # Take first available action
            _, _, reward, done, omega, mask = self.env.step(action)
        
        # Check if job completion time was recorded
        self.assertTrue(0 in self.env.completed_jobs)
        self.assertTrue(self.env.job_completion_times[0] > 0)

    def test_weighted_reward_calculation(self):
        """Test weighted completion time reward calculation"""
        adj, fea, omega, mask = self.env.reset(self.test_instance)
        
        # Track completion times
        completion_times = {}
        
        # Complete high priority job (weight = 2.0)
        for _ in range(self.n_m):
            action = omega[1]  # Actions for second job (index 1)
            _, _, reward, done, omega, mask = self.env.step(action)
            if done or mask[1]:  # If job 1 is completed
                completion_times[1] = self.env.job_completion_times[1]
        
        # Complete low priority job (weight = 0.5)
        for _ in range(self.n_m):
            action = omega[2]  # Actions for third job (index 2)
            _, _, reward, done, omega, mask = self.env.step(action)
            if done or mask[2]:  # If job 2 is completed
                completion_times[2] = self.env.job_completion_times[2]
        
        # Verify weighted completion times
        weighted_completion_high = completion_times[1] * self.weights[1]  # Weight = 2.0
        weighted_completion_low = completion_times[2] * self.weights[2]   # Weight = 0.5
        
        # The weighted completion time of the high-priority job should have
        # more impact on the objective
        impact_ratio = weighted_completion_high / weighted_completion_low
        self.assertGreater(impact_ratio, 1.0, 
            f"High priority job (weighted completion={weighted_completion_high}) should have "
            f"more impact than low priority job (weighted completion={weighted_completion_low})")
        
        # Verify that jobs have different completion times recorded
        self.assertNotEqual(completion_times[1], completion_times[2], 
            "Jobs should have different completion times")
        
        # Verify weights are being applied correctly
        self.env.weights = np.array([1.0, 1.0, 1.0])  # Equal weights
        weighted_sum_equal = sum(self.env.job_completion_times * self.env.weights)
        
        self.env.weights = self.weights  # Original weights
        weighted_sum_original = sum(self.env.job_completion_times * self.env.weights)
        
        self.assertNotEqual(weighted_sum_equal, weighted_sum_original,
            "Different weight distributions should result in different weighted sums")

    def test_ppo_memory_weights(self):
        """Test PPO memory handling of weights"""
        memory = Memory()
        memory.weights = torch.tensor(self.weights)
        
        # Add some test data
        memory.r_mb.append(1.0)
        memory.done_mb.append(False)
        
        # Check weight storage
        self.assertTrue(torch.allclose(memory.weights, torch.tensor(self.weights)))
        
        # Test memory clearing
        memory.clear_memory()
        self.assertEqual(len(memory.r_mb), 0)
        self.assertIsNone(memory.weights)

    def test_ppo_return_calculation(self):
        """Test PPO return calculation with weights"""
        ppo = PPO(
            lr=0.0001,
            gamma=0.99,
            k_epochs=4,
            eps_clip=0.2,
            n_j=self.n_j,
            n_m=self.n_m,
            num_layers=2,
            neighbor_pooling_type="sum",
            input_dim=2,
            hidden_dim=64,
            num_mlp_layers_feature_extract=2,
            num_mlp_layers_actor=2,
            hidden_dim_actor=32,
            num_mlp_layers_critic=2,
            hidden_dim_critic=32
        )
        
        # Test data
        rewards = [1.0, 2.0, -1.0]
        dones = [False, False, True]
        weights = torch.tensor(self.weights)
        
        # Calculate returns
        returns = ppo.calculate_returns(rewards, dones, weights)
        
        # Check return shape and values
        self.assertEqual(len(returns), len(rewards))
        self.assertTrue(torch.all(returns[:-1] >= returns[1:]))  # Returns should be descending due to discount

    def test_complete_episode(self):
        """Test complete episode with weighted completion times and visualization"""
        visualizer = TestVisualizer()
        
        adj, fea, omega, mask = self.env.reset(self.test_instance)
        total_reward = 0
        step_count = 0
        
        while not self.env.done():
            # Take random action from available ones
            valid_actions = omega[~mask]
            action = np.random.choice(valid_actions)
            _, _, reward, done, omega, mask = self.env.step(action)
            total_reward += reward
            step_count += 1
                
        # Verify episode completion
        self.assertEqual(step_count, self.n_j * self.n_m)
        self.assertEqual(len(self.env.completed_jobs), self.n_j)
        
        # Generate visualizations and statistics
        visualizer.plot_completion_times(
            self.env.job_completion_times,
            self.env.weights,
            "Test Episode Job Completion Times"
        )
        stats = visualizer.save_test_statistics(self.env)
        
        # Additional assertions using statistics
        self.assertTrue(all(ct > 0 for ct in stats['job_completion_times']),
                    "All jobs should have positive completion times")
        self.assertTrue(stats['total_weighted_completion_time'] > 0,
                    "Total weighted completion time should be positive")

if __name__ == '__main__':
    unittest.main()