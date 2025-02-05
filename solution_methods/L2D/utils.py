import os
import datetime
import json
from pathlib import Path

DEFAULT_RESULTS_ROOT = os.getcwd() + "/results/L2D" # i.e. /home/runner/work/Job-Shop-Scheduling/Job-Shop-Scheduling/results/L2D


def output_dir_exp_name(parameters):
    if 'experiment_name' in parameters['test_parameters'] is not None:
        exp_name = parameters['test_parameters']['experiment_name']
    else:
        instance_name = parameters['test_parameters']['problem_instance'].replace('/', '_')[1:]
        network = parameters['test_parameters']['trained_policy'].split('/')[-1].split('.')[0]
        if parameters['test_parameters']['sample'] == False:
            type = 'greedy'
        else:
            type = 'sample'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{instance_name}_network_{network}_{type}_{timestamp}"

    if 'folder_name' in parameters['test_parameters'] is not None:
        output_dir = parameters['test_parameters']['folder_name']
    else:
        output_dir = DEFAULT_RESULTS_ROOT
    return output_dir, exp_name
    
    '''
    base_dir = Path(DEFAULT_RESULTS_ROOT)
    # Get instance name from the problem instance path
    if parameters.get('method_name'):
        # For control solver methods
        instance_name = Path(parameters['test_parameters']['problem_instance']).stem
        output_dir = base_dir / parameters['method_name']
    else:
        # For L2D
        instance_name = Path(parameters['test_parameters']['problem_instance']).stem
        output_dir = base_dir / "L2D"
    
    # Default method is L2D if not specified
    method_name = parameters.get('method_name', 'L2D')
    
    return str(output_dir), instance_name
    '''

def results_saving(objective, path, parameters, **kwargs):
    """
    Save the L2D results to a JSON file.
    """
    # Base results dictionary
    # BEFORE: results = {
    #    "instance": parameters["test_parameters"]["problem_instance"],
    #    "makespan": makespan,
    #    "trained_policy" : parameters['test_parameters']['trained_policy'],
    #    "sample": parameters['test_parameters']['sample'],
    #    "seed": parameters['test_parameters']['seed']
    #}
    results = {
        "instance": parameters["test_parameters"]["problem_instance"],
        "objective": objective,
        "objective_type": "weighted_completion_time" if parameters["test_parameters"].get("is_weighted", False) else "makespan",
        "trained_policy": parameters["test_parameters"]["trained_policy"],
        "sample": parameters["test_parameters"]["sample"],
        "seed": parameters["test_parameters"]["seed"]
    }
    
    # Add additional fields dynamically from kwargs
    results.update(kwargs)

    # Generate a default experiment name based on instance and solve time if not provided
    os.makedirs(path, exist_ok=True)

    # Save results to JSON
    file_path = os.path.join(path, "L2D_results.json")
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)