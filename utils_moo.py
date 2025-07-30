from typing import Dict, List
import wandb
import torch
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import AllChem
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from objective import METRIC_MAXIMIZE
import math

class EvaluatedMolecule:
    def __init__(self, molecule: Chem.Mol, objective_values: torch.Tensor, raw_metrics: Dict[str, float]):
        
        self.molecule = molecule
        self.objective_values = objective_values # lower is better
        
        self.raw_metrics = raw_metrics
    
    def __lt__(self, other):
        return self.objective_values.mean() < other.objective_values.mean()



def calculate_hypervolume(points):
    """
    Calculate hypervolume for multi-objective optimization using pymoo's HV indicator.
    
    Args:
        points: N x K tensor of objective values (lower is better)
    
    Returns:
        hypervolume: Scalar hypervolume value
    """
    # pymoo's HV indicator assumes a minimization problem.
    # We convert our maximization problem to minimization by negating the points.
    points = points.cpu().numpy()
    n_points, n_dims = points.shape
    
    if n_points == 0:
        return 0.0
    
    # For the converted minimization problem, the reference point must be larger
    # than all point coordinates. Since original points are >= 0, a zero vector is a suitable ref point.
    reference_point = np.zeros(n_dims)
    hv_indicator = HV(ref_point=reference_point)
    hypervolume = hv_indicator(points)
    
    return hypervolume


def calculate_molecules_diversity(molecules: List[Chem.Mol]) -> float:
    """
    Calculates the internal molecules of a list of molecules, ensuring they are
    sanitized before fingerprinting.

    Args:
        molecules: A list of RDKit Mol objects.

    Returns:
        A diversity score between 0.0 (all identical) and 1.0 (highly diverse).
    """
    valid_mols = [m for m in molecules if m is not None]

    if len(valid_mols) < 2:
        return 0.0

    # --- NEW: Sanitize molecules before fingerprinting ---
    sanitized_mols = []
    for mol in valid_mols:
        try:
            # This is the crucial step that fixes the error.
            Chem.SanitizeMol(mol)
            sanitized_mols.append(mol)
        except Chem.rdchem.MolSanitizeException:
            # Optionally, you can log or print a warning here.
            # print("Warning: A molecule failed sanitization and will be skipped.")
            pass # Skip molecules that cannot be sanitized

    # If few molecules survive sanitization, diversity is low.
    if len(sanitized_mols) < 2:
        return 0.0

    # 1. Generate Morgan fingerprints from the *sanitized* molecules
    try:
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in sanitized_mols]
    except RuntimeError:
        # This is a fallback in case sanitization passes but fingerprinting still fails.
        return 0.0
        
    # 2. Calculate similarities for all unique pairs
    unique_similarities = []
    for i in range(len(fingerprints)):
        sims = BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
        unique_similarities.extend(sims)
    
    if not unique_similarities:
        return 0.0

    # 3. Calculate the final diversity score
    average_similarity = np.mean(unique_similarities)
    diversity = 1.0 - average_similarity
    
    return diversity

def get_num_pareto_front(objective_values):
    """
    Computes the Pareto front from a set of multi-objective values.

    Args:
        objective_values (torch.Tensor): A tensor of shape (N, K) where N is the
                                         number of points and K is the number of
                                         objectives. Assumes lower values are better.

    Returns:
        torch.Tensor: A tensor containing the points on the Pareto front.
    """
    nds = NonDominatedSorting()
    

    return len(nds.do(objective_values.cpu().numpy())[0])


def log_molecules_objective_values(evaluated_molecules: List[EvaluatedMolecule], objectives_feedbacks, stage, commit=True):

    buffer_molecules = [item.molecule for item in evaluated_molecules]
    buffer_objective_values = torch.stack([item.objective_values for item in evaluated_molecules])
    metrics = list(evaluated_molecules[0].raw_metrics.keys())


    molecules_diversity = calculate_molecules_diversity(buffer_molecules)
    hypervolume = calculate_hypervolume(buffer_objective_values)
    num_pareto_front = get_num_pareto_front(buffer_objective_values)

    log_dict = {
        "objectives_feedbacks": objectives_feedbacks,
        f"{stage}/diversity": molecules_diversity,
        f"{stage}/hypervolume": hypervolume,
        f"{stage}/number_of_pareto": num_pareto_front,
    }
    for metric in metrics:
        metric_tensor = torch.tensor([mol.raw_metrics[metric] for mol in evaluated_molecules])
        metric_tensor = metric_tensor[~metric_tensor.isnan()]
        log_dict[f"{stage}/{metric}_mean"] = metric_tensor.mean()
        log_dict[f"{stage}/{metric}_best"] = metric_tensor.max() if METRIC_MAXIMIZE[metric] else metric_tensor.min()
    wandb.log(log_dict, commit=commit)



def get_frac(x):
    """
    Calculates the fractional part of a number.
    """
    return x - np.floor(x)

def is_prime(num):
    """
    Checks if a number is prime.
    """
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def find_next_prime(num):
    """
    Finds the smallest prime number greater than or equal to the given number.
    """
    prime_candidate = num
    while True:
        if is_prime(prime_candidate):
            return prime_candidate
        prime_candidate += 1

# From paper: [Relationships between Decomposition-based MOEAs and Indicator-based MOEAs], Algorithm 6
def generate_weight_vectors(N, n):
    """
    Implements Algorithm 6 from the paper to generate weight vectors.

    Args:
        N (int): The number of weight vectors to generate.
        n (int): The dimension of the objective space.

    Returns:
        numpy.ndarray: A set of weight vectors W. Note the potential for
                       dimension mismatch due to bugs in the source algorithm.
    """
    # --- Step 1: Find the smallest prime number p ---
    # Find the smallest prime number p that satisfies p >= 2*(n-1)+1
    p = find_next_prime(2 * (n - 1) + 1)
    print(f"Step 1: Using prime number p = {p}")

    # --- Step 2: Construct the generating vector z ---
    # The paper's notation {x} is interpreted as the fractional part (x mod 1),
    # and [x] is interpreted as rounding to the nearest integer.
    z = np.zeros(n - 1)
    z[0] = 1
    for i in range(1, n - 2 + 1): # Corresponds to indices 1 to n-2 in paper
        expr = 2 * np.cos(2 * np.pi * i / p)
        # The paper's notation is ambiguous. We interpret [N * {expr}] as
        # rounding (N times the fractional part of expr).
        fractional_part = get_frac(expr)
        z[i] = np.round(N * fractional_part)
    print(f"Step 2: Generating vector z = {z}")

    # --- Step 3-5: Generate N points T in the (n-1) dimensional unit cube ---
    T = np.zeros((N, n - 1))    

    for j in range(1, N + 1):
        # T_j = {(j * z) / N}, where {...} is the fractional part.
        # This can be calculated efficiently using the modulo operator.
        T[j-1, :] = get_frac(j * z / N)
    print(f"Step 3-5: Generated T matrix of shape {T.shape}")

    U = np.random.rand(1, (n - 1))
    T = np.remainder((T + U), 1)

    # --- Step 6-9: Project T into subspaces Theta and X ---
    q = math.ceil((n - 1) / 2)
    # Theta corresponds to the first q columns of T
    Theta = T[:, 0:q]
    # X corresponds to the remaining columns
    X = T[:, q:n-1]
    # Scale Theta
    Theta = (np.pi / 2) * Theta
    print(f"Step 6-9: Created Theta (shape {Theta.shape}) and X (shape {X.shape})")

    # --- Step 10: Define k ---
    k = math.floor(n / 2)
    print(f"Step 10: k = {k}")

    # --- Step 11-31: Construct Weight Vectors W ---

    if n % 2 == 0:
        # --- Even n case ---
        print("Executing odd n case...")
        # Initialize W. NOTE: The algorithm only seems to fill n-1 components.
        
        
        Y = np.zeros((k+1, N))
        Y[0,:] = 0
        Y[k,:] = 1

        W = np.zeros((n, N))

        for i in range(k-1, 0, -1):
            Y[i] = Y[i+1] * (X[:,i-1] ** (1 / i))

        # Loop for i = 1 to k
        for i in range(1, k + 1):
            # W_{2i-1} = sqrt(Y_i - Y_{i-1}) * cos(Theta_i)
            # W_{2i}   = sqrt(Y_i - Y_{i-1}) * sin(Theta_i)
            # Python indices: W[:, 2*i-2], W[:, 2*i-1], Theta[:, i-1]
            term = np.sqrt(Y[i] - Y[i - 1])
            W[2 * i - 2] = term * np.cos(Theta[:, i - 1])
            W[2 * i - 1] = term * np.sin(Theta[:, i - 1])

    else:
        # --- Odd n case ---

        Y = np.zeros((k+1, N))
        Y[k,:] = 1

        W = np.zeros((n, N))

        # Loop for i = k down to 1
        for i in range(k, 0, -1):
            # Y_i = Y_{i+1} * X_{i,:}^{2/(2i-1)}
            # Paper uses 1-based indexing for X. X_{i,:} -> X[:, i-1]
            Y[i-1] = Y[i] * (X[:, i - 1] ** (2 / (2 * i - 1)))

        W[0] = np.sqrt(Y[0])

        # Loop for i = 1 to k
        for i in range(1, k + 1):
            # W_{2i}   = sqrt(Y_{i+1} - Y_i) * cos(Theta_i)
            # W_{2i+1} = sqrt(Y_{i+1} - Y_i) * sin(Theta_i)
            # Python indices: W[:, 2*i-1], W[:, 2*i], Theta[:, i-1]
            term = np.sqrt(Y[i] - Y[i-1])
            W[2 * i - 1] = term * np.cos(Theta[:, i - 1])
            W[2 * i]     = term * np.sin(Theta[:, i - 1])

    # --- Step 32: Return W ---
    return W
