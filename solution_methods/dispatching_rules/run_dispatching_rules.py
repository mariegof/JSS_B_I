import argparse
import logging
import os
from pathlib import Path
import sys

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

from scheduling_environment.jobShop import JobShop
from plotting.drawer import plot_gantt_chart, draw_precedence_relations
from solution_methods.dispatching_rules.utils import configure_simulation_env, output_dir_exp_name, results_saving
from solution_methods.helper_functions import load_parameters, load_job_shop_env
from solution_methods.dispatching_rules.src.scheduling_functions import scheduler

logging.basicConfig(level=logging.INFO)
PARAM_FILE = str(base_path / "configs" / "dispatching_rules.toml")


def run_dispatching_rules(jobShopEnv, **kwargs):
    dispatching_rule = kwargs['instance']['dispatching_rule']
    machine_assignment_rule = kwargs['instance']['machine_assignment_rule']

    if dispatching_rule == 'SPT' and machine_assignment_rule != 'SPT':
        raise ValueError("SPT dispatching rule requires SPT machine assignment rule.")

    # Configure simulation environment
    simulationEnv = configure_simulation_env(jobShopEnv, **kwargs)
    simulationEnv.simulator.process(scheduler(simulationEnv, **kwargs))

    # For online arrivals, run the simulation until the configured end time
    if kwargs['instance']['online_arrivals']:
        simulationEnv.simulator.run(until=kwargs['online_arrival_details']['simulation_time'])
    # For static instances, run until all operations are scheduled
    else:
        simulationEnv.simulator.run()
        
    # Calculate objective based on instance type
    if jobShopEnv.is_weighted:
        raw_objective = simulationEnv.jobShopEnv.weighted_completion_time
        total_weight = sum(job.weight for job in jobShopEnv.jobs)
        objective = raw_objective / total_weight  # Normalize to match L2D
        logging.info(f"Normalized Weighted Sum Objective: {objective}")
    else:
        # Use makespan for unweighted instances
        objective = simulationEnv.jobShopEnv.makespan
        logging.info(f"Makespan: {objective}")

    # BEFORE: makespan = simulationEnv.jobShopEnv.makespan
    # logging.info(f"Makespan: {makespan}")

    # return makespan, simulationEnv.jobShopEnv
    return objective, simulationEnv.jobShopEnv


def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    # Configure the simulation environment
    if parameters['instance']['online_arrivals']:
        jobShopEnv = JobShop()
        # BEFORE: makespan, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)
        objective, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)
        logging.warning(f"Makespan/Weighted Sum objective is irrelevant for problems configured with 'online arrivals'.")
    else:
        jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
        # BEFORE: makespan, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)
        objective, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)

    # BEFORE: if makespan is not None:
    if objective is not None:
        # Check output configuration and prepare output paths if needed
        output_config = parameters['output']
        save_gantt = output_config.get('save_gantt')
        save_results = output_config.get('save_results')
        show_gantt = output_config.get('show_gantt')
        show_precedences = output_config.get('show_precedences')

        if save_gantt or save_results:
            output_dir, exp_name = output_dir_exp_name(parameters)
            output_dir = os.path.join(output_dir, f"{exp_name}")
            os.makedirs(output_dir, exist_ok=True)

        # Draw precedence relations if required
        if show_precedences:
            draw_precedence_relations(jobShopEnv)

        # Plot Gantt chart if required
        if show_gantt or save_gantt:
            logging.info("Generating Gantt chart.")
            plt = plot_gantt_chart(jobShopEnv)

            if save_gantt:
                # BEFORE: plt.savefig(output_dir + "/gantt.png")
                plt.savefig(os.path.join(output_dir, "gantt.png"))
                logging.info(f"Gantt chart saved to {output_dir}")

            if show_gantt:
                plt.show()

        # Save results if enabled
        if save_results:
            # BEFORE: results_saving(makespan, output_dir, parameters)
            results_saving(
                objective=objective,
                path=output_dir,
                parameters=parameters,
                makespan=jobShopEnv.makespan,
                max_flowtime=jobShopEnv.max_flowtime,
            )
            logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dispatching Rules.")
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)