"""
Performance Analysis Framework for Job Shop Scheduling Methods.
Uses main() functions of existing scheduling methods for better integration.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import tomli
import tomli_w
import subprocess
from tqdm import tqdm
print(Path.cwd())
# Set up project root path and add it to sys.path
base_path = Path(__file__).resolve().parent  # perf_profile.py is at project root
sys.path.append(str(base_path))
logging.basicConfig(level=logging.INFO)

from solution_methods.helper_functions import load_parameters
from plotting.perfprof import create_performance_profile

class PerformanceAnalyzer:
    """Manages the complete performance analysis pipeline."""
    
    def __init__(self, config_path):
        """Initialize the analyzer with experiment configuration."""
        self.base_path = base_path
        self.exp_config = load_parameters(config_path)
        self.output_dir = self._setup_output_directory()
        
        # Initialize results structure
        self.results = {
            'metadata': {
                'instance_type': self.exp_config['global_parameters']['instance_type'],
                'n_instances': self.exp_config['global_parameters']['n_instances'],
                'timestamp': datetime.now().isoformat(),
                'parameters': self.exp_config['global_parameters']
            },
            'methods': {
                'L2D': {'objectives': [], 'runtimes': []},
                'SPT': {'objectives': [], 'runtimes': []},
            }
        }
        
        if self.exp_config['global_parameters']['instance_type'] != "unweighted":
            self.results['methods']['WSPT'] = {'objectives': [], 'runtimes': []}
            
        # Define paths relative to project root
        self.l2d_config_path = self.base_path / "configs" / "L2D.toml"
        self.dr_config_path = self.base_path / "configs" / "dispatching_rules.toml"
        self.intermediate_path = self.output_dir / "intermediate_results.json"
        self.progress_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

        self._backup_configs()
    
    def _setup_output_directory(self):
        """Create and return the output directory path."""
        base_dir = Path(self.exp_config['output_parameters']['base_dir'])
        exp_name = self.exp_config['output_parameters']['experiment_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = base_dir / f"{exp_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_intermediate_results(self):
        """Save current progress to avoid losing work."""
        with open(self.intermediate_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def load_intermediate_results(self, path):
        """Load previously saved results."""
        with open(path, 'r') as f:
            self.results = json.load(f)
    
    def _backup_configs(self):
        """Create backups of original configuration files."""
        shutil.copy2(self.l2d_config_path, self.l2d_config_path.with_suffix('.bak'))
        shutil.copy2(self.dr_config_path, self.dr_config_path.with_suffix('.bak'))
    
    def _restore_configs(self):
        """Restore original configuration files."""
        shutil.move(self.l2d_config_path.with_suffix('.bak'), self.l2d_config_path)
        shutil.move(self.dr_config_path.with_suffix('.bak'), self.dr_config_path)
    
    def _update_specific_config_fields(self, config_path, updates):
        """Updates configuration fields while preserving existing structure."""
        
        # Read existing configuration
        with open(config_path, 'rb') as f:
            config = tomli.load(f)
        #print(f"Original config: {config}")
        
        # Process and apply updates
        for section, fields in updates.items():
            if section not in config:
                print(f"Section {section} not found in config!")
                continue
                
            # Create a copy of fields to modify
            processed_fields = fields.copy()
            
            # Handle problem instance path
            if 'problem_instance' in processed_fields:
                # Convert to the expected format with /exp/ prefix
                instance_name = Path(processed_fields['problem_instance']).name
                processed_fields['problem_instance'] = f"/exp/{instance_name}"
                #print(f"Processed problem_instance: {processed_fields['problem_instance']}")
            
            # Update the configuration section
            config[section].update(processed_fields)
        
        #print(f"Updated config: {config}")
        
        # Write updated configuration
        with open(config_path, 'wb') as f:
            tomli_w.dump(config, f)
        
        # Verify write
        with open(config_path, 'rb') as f:
            written_config = tomli.load(f)
        #print(f"Verified written config: {written_config}")
    
    def generate_instance(self, instance_idx, output_path):
        """Generate a job shop instance with specified parameters."""
        print(f"Generating instance with path: {output_path}")
        params = self.exp_config['global_parameters']
        seed = params['seed'] + instance_idx
        
        if params['instance_type'] == "unweighted":
            from data.weighted_instance_generator import uniform_instance_gen
            instance_str = uniform_instance_gen(
                n_j=params['n_j'],
                n_m=params['n_m'],
                low=params['low'],
                high=params['high'],
                seed=seed
            )
        else:
            from data.weighted_instance_generator import weighted_instance_gen
            weight_type = "uniform" if params['instance_type'] == "uniform" else "differentiated"
            instance_str = weighted_instance_gen(
                n_j=params['n_j'],
                n_m=params['n_m'],
                low=params['low'],
                high=params['high'],
                weight_type=weight_type,
                seed=seed
            )
        
        with output_path.open('w') as f:
            f.write(instance_str)
    
    def _run_method(self, method_name, instance_path):
        """Run a scheduling method with standardized result handling."""
        start_time = datetime.now()
        instance_name = instance_path.stem
        exp_name = f"exp_{instance_name}"  # Consistent naming without timestamps
        
        try:
            if method_name == 'L2D':
                # Configure L2D
                base_dir = "./results/L2D"  # Relative path for config
                config_updates = {
                    'test_parameters': {
                        'problem_instance': str(instance_path),
                        'show_gantt': False,
                        'save_gantt': False,
                        'show_precedences': False,
                        'save_results': True,
                        'folder_name': base_dir,
                        'experiment_name': exp_name,
                        'is_weighted': self.exp_config['global_parameters']['instance_type'] != "unweighted"
                    }
                }
                self._update_specific_config_fields(self.l2d_config_path, config_updates)
                script_path = str(self.base_path / "solution_methods" / "L2D" / "run_L2D.py")
                # Absolute path for finding results - match the method's structure
                output_dir = self.base_path / "results" / "L2D" / exp_name
                results_file = "L2D_results.json"
                
            else:  # Dispatching rules
                # Configure dispatching rules
                base_dir = "./results/dispatching_rules"  # Relative path for config
                config_updates = {
                    'instance': {
                        'problem_instance': str(instance_path),
                        'dispatching_rule': method_name,
                        'machine_assignment_rule': 'SPT',
                        'online_arrivals': False
                    },
                    'output': {
                        'logbook': False,
                        'show_gantt': False,
                        'save_gantt': False,
                        'show_precedences': False,
                        'save_results': True,
                        'exp_name': exp_name,
                        'folder': base_dir
                    }
                }
                self._update_specific_config_fields(self.dr_config_path, config_updates)
                script_path = str(self.base_path / "solution_methods" / "dispatching_rules" / "run_dispatching_rules.py")
                # Absolute path for finding results - match the method's structure
                output_dir = self.base_path / "results" / "dispatching_rules" / exp_name
                results_file = f"{method_name}_results.json"

            # Create base directory if it doesn't exist
            (output_dir.parent).mkdir(parents=True, exist_ok=True)
            
            # Run method with absolute paths
            subprocess.run([sys.executable, script_path], check=True)
            
            # Look for results file in output directory
            results_path = output_dir / results_file
            if not results_path.exists():
                raise FileNotFoundError(f"Results file not found at {results_path}")
            
            # Read and process results
            with open(results_path, 'r') as f:
                run_results = json.load(f)
                
            runtime = (datetime.now() - start_time).total_seconds()
            objective = run_results['objective']
            
            logging.info(f"{method_name} completed: objective={objective:.2f}, runtime={runtime:.2f}s")
            
            # Clean up
            if output_dir.exists():
                shutil.rmtree(output_dir)
                
            return objective, runtime
                
        except Exception as e:
            logging.error(f"Error running {method_name} on {instance_name}: {str(e)}")
            if 'output_dir' in locals() and output_dir.exists():
                shutil.rmtree(output_dir) # Clean up in case of error
            return None, None
        
    def run_analysis(self):
        """Execute the complete analysis pipeline."""
        n_instances = self.results['metadata']['n_instances']
        #print(f"Base path: {base_path}")
        # Create instance directory in data/exp relative to project root
        instance_dir = self.base_path / "data" / "exp"
        #print(f"Creating instance directory at: {instance_dir}")
        instance_dir.mkdir(exist_ok=True, parents=True)    
        # Check for existing results
        if self.intermediate_path.exists():
            user_input = input(f"Found existing results at {self.intermediate_path}. Load them? (y/n): ")
            if user_input.lower() == 'y':
                self.load_intermediate_results(self.intermediate_path)
                print("Loaded previous results. Continuing from where we left off...")
                # Find the last completed instance
                completed_instances = max(
                    len(method_data['objectives']) 
                    for method_data in self.results['methods'].values()
                )
                start_idx = completed_instances
            else:
                start_idx = 0
        else:
            start_idx = 0
                
        try:
            # Main processing loop with progress bar
            with tqdm(
                total=n_instances,
                initial=start_idx,
                desc="Processing instances",
                bar_format=self.progress_format
            ) as pbar:
                for idx in range(start_idx, n_instances):
                    instance_path = instance_dir / f"instance_{idx}.txt"
                    
                    # Generate instance
                    self.generate_instance(idx, instance_path)
                    print(f"\nProcessing instance {idx + 1}/{n_instances}")
                    
                    # Run each method with status updates
                    print("Collected objectives:")
                    for method_name in self.results['methods']:
                        print(f"  Running {method_name}...", end='', flush=True)
                        objective, runtime = self._run_method(method_name, instance_path)
                        
                        if objective is not None:
                            self.results['methods'][method_name]['objectives'].append(objective)
                            self.results['methods'][method_name]['runtimes'].append(runtime)
                            print(f" Done (objective: {objective:.2f}, runtime: {runtime:.2f}s)")
                        else: 
                            print(" Failed!")
                            
                    # Clean up instance file
                    instance_path.unlink()
                    # Save intermediate results after each instance
                    self.save_intermediate_results()
                    # Update progress bar
                    pbar.update(1)
                    # Print summary after each instance
                    print("\nCurrent statistics:")
                    for method in self.results['methods']:
                        objectives = self.results['methods'][method]['objectives']
                        if objectives:
                            print(f"  {method}: avg={np.mean(objectives):.2f}, "
                                  f"min={np.min(objectives):.2f}, "
                                  f"max={np.max(objectives):.2f}")
                
                # Generate final results
                print("\nGenerating final performance profile...")
                self.create_performance_profile()
                
                # Save final results
                final_results_path = self.output_dir / "final_results.json"
                with open(final_results_path, 'w') as f:
                    json.dump(self.results, f, indent=2)
                print(f"Final results saved to {final_results_path}")
            
        finally:
            if instance_dir.exists():
                shutil.rmtree(instance_dir) # Clean up instance directory
            self._restore_configs() # Restore original configuration files
    
    def create_performance_profile(self):
        """Generate performance profile with detailed debugging."""
        methods = list(self.results['methods'].keys())
        print("\nAnalyzing results for methods:", methods)
        
        # Print raw data for each method
        print("\nRaw Objective Values:")
        raw_objectives = []
        for method in methods:
            objectives = self.results['methods'][method]['objectives']
            print(f"\n{method}:")
            print(f"Number of results: {len(objectives)}")
            print(f"First 5 values: {objectives[:5]}")
            print(f"Statistics: min={np.min(objectives):.2f}, max={np.max(objectives):.2f}, "
                f"mean={np.mean(objectives):.2f}")
            raw_objectives.append(objectives)
        
        # Convert to array and show shape
        objectives = np.array(raw_objectives).T
        print(f"\nObjectives array shape: {objectives.shape}")
        
        # Show performance ratio calculations
        best_per_instance = np.min(objectives, axis=1)
        ratios = objectives / best_per_instance[:, np.newaxis]
        
        print("\nPerformance Ratios (first 5 instances):")
        print("Instance | " + " | ".join(f"{m:^10}" for m in methods))
        print("-" * (10 + 13 * len(methods)))
        for i in range(min(5, len(best_per_instance))):
            ratio_str = " | ".join(f"{ratios[i,j]:^10.3f}" for j in range(len(methods)))
            print(f"{i:^8} | {ratio_str}")
        
        # Create the profile plot
        fig, ax = create_performance_profile(
            data=objectives,
            method_names=methods,
            title=f'Performance Profile\n{self.results["metadata"]["instance_type"]} instances',
            max_ratio=1.4,
            styles=['-o', '-s', '-^']
        )
        
        # Customize plot
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(np.arange(1.0, 1.5, 0.1))
        
        plt.savefig(self.output_dir / "performance_profile.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
def main():
    parser = argparse.ArgumentParser(description="Run performance analysis")
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        default="configs/experiment.toml",
        help="path to experiment config file",
    )
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.config_file)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()