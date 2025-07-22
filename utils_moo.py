from typing import Dict, List
import wandb
import torch
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import AllChem
from pymoo.indicators.hv import HV

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

def get_pareto_front(objective_values):
    """
    Computes the Pareto front from a set of multi-objective values.

    Args:
        objective_values (torch.Tensor): A tensor of shape (N, K) where N is the
                                         number of points and K is the number of
                                         objectives. Assumes lower values are better.

    Returns:
        torch.Tensor: A tensor containing the points on the Pareto front.
    """
    if objective_values.numel() == 0:
        return torch.empty(0, objective_values.shape[1], device=objective_values.device, dtype=objective_values.dtype)

    is_pareto = torch.ones(objective_values.shape[0], dtype=torch.bool, device=objective_values.device)
    for i in range(objective_values.shape[0]):
        # A point is on the Pareto front if no other point dominates it.
        # A point `j` dominates point `i` if `j` is better or equal in all objectives
        # and strictly better in at least one.
        dominators = (objective_values <= objective_values[i]).all(dim=1) & (objective_values < objective_values[i]).any(dim=1)
        if dominators.any():
            is_pareto[i] = False
            
    return objective_values[is_pareto]


def log_molecules_objective_values(evaluated_molecules: List[EvaluatedMolecule], objectives_feedbacks, stage, commit=True):

    buffer_molecules = [item.molecule for item in evaluated_molecules]
    buffer_objective_values = torch.stack([item.objective_values for item in evaluated_molecules])
    metrics = list(evaluated_molecules[0].raw_metrics.keys())


    molecules_diversity = calculate_molecules_diversity(buffer_molecules)
    hypervolume = calculate_hypervolume(buffer_objective_values)
    pareto_front = get_pareto_front(buffer_objective_values)

    log_dict = {
        "objectives_feedbacks": objectives_feedbacks,
        f"{stage}/diversity": molecules_diversity,
        f"{stage}/hypervolume": hypervolume,
        f"{stage}/number_of_pareto": len(pareto_front),
    }
    for metric in metrics:
        log_dict[f"{stage}/{metric}_mean"] = torch.tensor([mol.raw_metrics[metric] for mol in evaluated_molecules]).mean()
    
    wandb.log(log_dict, commit=commit)