import sys
import numpy as np
from pathlib import Path

base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.data.instance_generator import permute_rows

def weighted_instance_gen(n_j, n_m, low, high, seed=None, weight_type="uniform"):
    """
    Generate weighted JSSP instance in text format for parser.
    Returns instance as a formatted string.
    
    Format:
    n_j n_m
    machine_id1 duration1 machine_id2 duration2 ... [weight]
    machine_id1 duration1 machine_id2 duration2 ... [weight]
    ...
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        
    # Generate data
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(n_m), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    
    # Generate weights according to type
    if weight_type == "uniform":
        weights = np.ones(n_j)
    elif weight_type == "differentiated":
        weights = np.random.randint(1, 5, size=n_j)
    else:
        raise ValueError("Only 'uniform' and 'differentiated' weight types are supported")
    
    # Format instance as text
    lines = [f"{n_j} {n_m}"]
    for j in range(n_j):
        # Format each operation's machine and duration
        job_entries = []
        for m in range(n_m):
            job_entries.extend([str(int(machines[j,m])), str(int(times[j,m]))])
        # Add weight at the end
        job_entries.append(str(int(weights[j])))
        lines.append(" ".join(job_entries))
    
    return "\n".join(lines)    

def uniform_instance_gen(n_j, n_m, low=1, high=99, seed=None):
    """
    Generate an unweighted job shop instance in the same style as the weighted generator.
    The only difference is that we don't append weights at the end of each line.
    """
    if seed is not None:
        np.random.seed(seed)
        
    times = np.random.randint(low, high + 1, size=(n_j, n_m))
    machines = np.zeros((n_j, n_m), dtype=np.int32)
    
    # Randomly assign machines for each job's operations
    for i in range(n_j):
        machines[i] = np.random.permutation(n_m)
        
    # Generate instance string
    lines = [f"{n_j} {n_m}"]
    for i in range(n_j):
        job_entries = []
        for j in range(n_m):
            job_entries.extend([str(int(machines[i,j])), str(int(times[i,j]))])
        lines.append(" ".join(job_entries))
        
    return "\n".join(lines)

def generate_benchmark_instances(parameters):
    """
    Generate benchmark instances and save as text files in wjsp folder.
    """
    # Extract parameters from config
    env_params = parameters["env_parameters"]
    n_j = env_params["n_j"]
    n_m = env_params["n_m"]
    low = env_params["low"]
    high = env_params["high"]
    seed = env_params["np_seed_train"]
    
    # Create wjsp directory next to this script
    wjsp_dir = Path(__file__).parent / "wjsp"
    wjsp_dir.mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate instances for both weight types
    num_instances = 2  # Generate 2 instances of each type
    for weight_type in ["uniform", "differentiated"]:
        for i in range(num_instances):
            # Generate instance text
            instance = weighted_instance_gen(
                n_j=n_j,
                n_m=n_m,
                low=low,
                high=high,
                weight_type=weight_type
            )
            
            # Save to text file
            filename = f"wjssp_{n_j}x{n_m}_{weight_type}_{i}.txt"
            with open(wjsp_dir / filename, "w") as f:
                f.write(instance)
                
        print(f"Generated {num_instances} {weight_type} instances: {n_j}x{n_m}")

if __name__ == "__main__":
    param_file = str(Path(__file__).parents[1] / "configs/L2D.toml")
    parameters = load_parameters(param_file)
    generate_benchmark_instances(parameters)