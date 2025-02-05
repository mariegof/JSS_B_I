import argparse
import logging
import os
from pathlib import Path
import sys
import numpy as np

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
from solution_methods.helper_functions import (
    load_parameters, load_job_shop_env, initialize_device, set_seeds
)
from plotting.drawer import draw_gantt_chart, draw_precedence_relations
from utils import output_dir_exp_name, results_saving

# Use base_path to create absolute path to config file
PARAM_FILE = str(base_path / "configs" / "L2D.toml")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def solve_control(jobShopEnv, method="HWF", **parameters):
    """
    Solve using control methods, matching run_L2D.py's interface exactly.
    Returns same output format for consistency.
    """
    set_seeds(parameters["test_parameters"]["seed"])
    
    # Reset environment
    jobShopEnv.reset()
    # Extract weights (ensure they exist)
    weights = np.array([job.weight if job.weight is not None else 0 for job in jobShopEnv.jobs])
    
    if method == "HWF":
        job_order = np.argsort(-weights) # Highest Weight First
    else:  # WSPT
        processing_times = np.array([
            sum(sum(op.processing_times.values()) for op in job.operations)
            for job in jobShopEnv.jobs
        ])
        ratios = weights / processing_times
        job_order = np.argsort(-ratios)
    
    # Schedule operations
    for job_idx in job_order:
        job = jobShopEnv.get_job(job_idx)
        for operation in job.operations:
            machine_id = list(operation.processing_times.keys())[0]
            duration = list(operation.processing_times.values())[0]
            jobShopEnv.schedule_operation_on_machine(operation, machine_id, duration)
    
    # Calculate objective matching L2D
    objective = (
        jobShopEnv.weighted_completion_time 
        if jobShopEnv.is_weighted 
        else jobShopEnv.makespan
    )
    
    return objective, jobShopEnv

def main(param_file=PARAM_FILE):
    # Load parameters
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return
    
    # Load instance
    jobShopEnv = load_job_shop_env(parameters['test_parameters'].get('problem_instance'))
    
    # Solve with both methods
    for method in ["HWF", "WSPT"]:
        # Create method-specific copy of parameters
        method_params = parameters.copy()
        method_params['method_name'] = method
        
        # Solve instance with current method
        objective, jobShopEnv = solve_control(jobShopEnv, method, **method_params)
        
        if objective is not None:
            # Check output configuration and prepare output paths if needed
            output_config = method_params['test_parameters']
            save_gantt = output_config.get('save_gantt')
            save_results = output_config.get('save_results')
            show_gantt = output_config.get('show_gantt')

            if save_gantt or save_results:
                output_dir, exp_name = output_dir_exp_name(method_params)
                output_dir = os.path.join(output_dir, f"{exp_name}")
                os.makedirs(output_dir, exist_ok=True)

            # Plot Gantt chart if required
            if show_gantt or save_gantt:
                plt = draw_gantt_chart(jobShopEnv)
                if save_gantt:
                    plt.savefig(os.path.join(output_dir, "gantt.png"))
                    logging.info(f"Gantt chart saved to {output_dir}")
                if show_gantt:
                    plt.show()
                plt.close()
                
            if save_results:
                results_saving(
                    objective=objective,
                    path=output_dir,
                    parameters=method_params,
                    makespan=jobShopEnv.makespan,
                    max_flowtime=jobShopEnv.max_flowtime,
                )
                logging.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run control methods")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )
    args = parser.parse_args()
    main(param_file=args.config_file)