import numpy as np
from pathlib import Path


def permute_rows(x):
    # Permutes the rows of a numpy array `x` randomly.
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uniform_instance_generator(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines

def weighted_instance_generator(n_j, n_m, low, high, weight_type="uniform"):
    """Generate a weighted JSSP instance.
    
    Args:
        n_j: Number of jobs
        n_m: Number of machines
        low: Min processing time
        high: Max processing time
        weight_type: "uniform" or "differentiated"
    
    Returns:
        tuple: (times, machines, weights) where weights is a 1D array
    """
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(n_m), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    
    # Create weights as a 1D array (one weight per job)
    if weight_type == "uniform":
        weights = np.ones(n_j, dtype=np.int32)
    elif weight_type == "differentiated":
        weights = np.random.randint(1, 101, size=n_j, dtype=np.int32)
    else:
        raise ValueError("Only 'uniform' and 'differentiated' weight types supported")
    
    # Expand weights to match the shape of times/machines
    weights_matrix = np.zeros((n_j, n_m), dtype=np.int32)
    weights_matrix[:, -1] = weights  # Only last operation has weight
    
    return times, machines, weights_matrix

def generate_training_instances(current_dir, sizes, low=1, high=99, batch_size=100, seed=200):
    """
    Generate benchmark instances for various problem sizes and weight types.
    
    Args:
        current_dir (Path): Directory to save generated instances
        sizes (list): List of (jobs, machines) tuples defining problem sizes
        low (int): Minimum processing time
        high (int): Maximum processing time
        batch_size (int): Number of instances to generate per configuration
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)

    for j, m in sizes:
        # Generate standard (unweighted) instances
        standard_data = np.array([uniform_instance_generator(n_j=j, n_m=m, low=low, high=high) for _ in range(batch_size)])
        np.save(current_dir / f'generatedData{j}_{m}_Seed{seed}.npy', standard_data)
        print(f"Generated standard instances {j}x{m}")

        # Generate both uniform and differentiated weighted instances
        for weight_type in ["uniform", "differentiated"]:
            weighted_data = np.array([weighted_instance_generator(
                n_j=j, n_m=m, low=low, high=high, weight_type=weight_type) 
                for _ in range(batch_size)])
            np.save(current_dir / f'generatedData{j}_{m}_{weight_type}_Seed{seed}.npy', 
                   weighted_data)
            print(f"Generated {weight_type} weighted instances {j}x{m}")

def override(fn):
    """
    override decorator
    """
    return fn


if __name__ == "__main__":
    # Get the directory where this script is located
    current_dir = Path(__file__).parent
    # Set parameters
    #j = 20              # Number of jobs
    #m = 10              # Number of machines
    #l = 1               # Minimum processing time
    #h = 99              # Maximum processing time
    #batch_size = 100    # nr of instances to generate
    #seed = 201          # Random seed

    # Set random seed for reproducibility
    #np.random.seed(seed)
    
    # Define problem sizes to generate (job × machine)
    problem_sizes = [
        (6, 6), (10, 10), (15, 10), (15, 15),
        (20, 10), (20, 15), (20, 20), (30, 15),
        (30, 20), (50, 20), (100, 20), (200, 50)
    ]

    # Generate data for batch_size number of instances
    generate_training_instances(current_dir, problem_sizes)
    print("Data generation completed.")