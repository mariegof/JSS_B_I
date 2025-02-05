import os
import json
import datetime

from scheduling_environment.simulationEnv import SimulationEnv
from solution_methods.helper_functions import load_job_shop_env

DEFAULT_RESULTS_ROOT = os.getcwd() + "/results/dispatching_rules/"


def configure_simulation_env(jobShopEnv, **parameters):
    """Configure the simulation environment based on parameters."""
    simulationEnv = SimulationEnv(online_arrivals=parameters['instance']['online_arrivals'])
    simulationEnv.jobShopEnv = jobShopEnv

    if parameters['instance']['online_arrivals']:
        simulationEnv.set_online_arrival_details(parameters['online_arrival_details'])
        simulationEnv.jobShopEnv.set_nr_of_machines(parameters['online_arrival_details']['number_total_machines'])

    return simulationEnv


def output_dir_exp_name(parameters):
    if 'exp_name' in parameters['output'] is not None:
        exp_name = parameters['output']['exp_name']
    else:
        if parameters['instance']['online_arrivals']:
            instance_name = 'online_arrival_config'
        else:
            instance_name = parameters['instance']['problem_instance'].replace('/', '_')[1:]
            instance_name = instance_name.split('.')[0] if '.' in instance_name else instance_name
        dispatching_rule = parameters['instance']['dispatching_rule']
        machine_assignment_rule = parameters['instance']['machine_assignment_rule']
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{instance_name}_{dispatching_rule}_{machine_assignment_rule}_{timestamp}"

    if 'folder' in parameters['output'] is not None:
        output_dir = parameters['output']['folder']
    else:
        output_dir = DEFAULT_RESULTS_ROOT
    return output_dir, exp_name


def results_saving(objective, path, parameters, **kwargs):
    """
    Save the dispatching rules scheduling results to a JSON file.
    """
    if parameters['instance']['online_arrivals']:
        results = {
            "instance": parameters["instance"]["problem_instance"],
            # "makespan": makespan,
            "objective": objective,
            "objective_type": "weighted_completion_time" if parameters["instance"].get("is_weighted", False) else "makespan",
            "dispatching_rule": parameters["instance"]["dispatching_rule"],
            "machine_assignment_rule": parameters["instance"]["machine_assignment_rule"],
            "number_total_machines": parameters["online_arrival_details"]["number_total_machines"],
            "inter_arrival_time": parameters["online_arrival_details"]["inter_arrival_time"],
            "simulation_time": parameters["online_arrival_details"]["simulation_time"],
            "min_nr_operations_per_job": parameters["online_arrival_details"]["min_nr_operations_per_job"],
            "max_nr_operations_per_job": parameters["online_arrival_details"]["max_nr_operations_per_job"],
            "min_duration_per_operation": parameters["online_arrival_details"]["min_duration_per_operation"],
            "max_duration_per_operation": parameters["online_arrival_details"]["max_duration_per_operation"]
        }

    else:
        results = {
            "instance": parameters["instance"]["problem_instance"],
            # "makespan": makespan,
            "objective": objective,
            "objective_type": "weighted_completion_time" if parameters["instance"].get("is_weighted", False) else "makespan",
            "dispatching_rule": parameters["instance"]["dispatching_rule"],
            "machine_assignment_rule": parameters["instance"]["machine_assignment_rule"]
        }
        
    # Add additional metrics from kwargs
    results.update(kwargs)

    # Generate a default experiment name based on instance and solve time if not provided
    os.makedirs(path, exist_ok=True)

    # Save results to JSON
    # file_path = os.path.join(path, "GA_results.json")
    method_name = parameters["instance"]["dispatching_rule"]
    file_path = os.path.join(path, f"{method_name}_results.json")
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
