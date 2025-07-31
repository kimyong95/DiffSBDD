import argparse
from pathlib import Path
from typing import List

from utils import seed_everything
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
import pandas as pd
import random
import wandb
from objective import Objective, METRIC_MAXIMIZE
from collections import defaultdict
from torch_scatter import scatter_mean
from openbabel import openbabel
from pymoo.indicators.hv import HV
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from typing import Dict
from lightning_modules import LigandPocketDDPM
from constants import FLOAT_TYPE, INT_TYPE
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.metrics import MoleculeProperties
from sbdd_metrics.metrics import FullEvaluator
from pyMultiobjective.algorithm import s_ii

from utils_moo import EvaluatedMolecule, log_molecules_objective_values

def prepare_from_sdf_files(sdf_files, atom_encoder):

    ligand_coords = []
    atom_one_hot = []
    for file in sdf_files:
        rdmol = Chem.SDMolSupplier(str(file), sanitize=False)[0]
        ligand_coords.append(
            torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        )
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in rdmol.GetAtoms()])
        atom_one_hot.append(
            F.one_hot(types, num_classes=len(atom_encoder))
        )

    return torch.cat(ligand_coords, dim=0), torch.cat(atom_one_hot, dim=0)


def prepare_ligand_from_pdb(biopython_atoms, atom_encoder):

    coord = torch.tensor(np.array([a.get_coord()
                                   for a in biopython_atoms]), dtype=FLOAT_TYPE)
    types = torch.tensor([atom_encoder[a.element.capitalize()]
                          for a in biopython_atoms])
    one_hot = F.one_hot(types, num_classes=len(atom_encoder))

    return coord, one_hot


def prepare_substructure(ref_ligand, fix_atoms, pdb_model):

    if fix_atoms[0].endswith(".sdf"):
        # ligand as sdf file
        coord, one_hot = prepare_from_sdf_files(fix_atoms, model.lig_type_encoder)

    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ref_ligand.split(':')
        ligand = utils.get_residue_with_resi(pdb_model[chain], int(resi))
        fixed_atoms = [a for a in ligand.get_atoms() if a.get_name() in set(fix_atoms)]
        coord, one_hot = prepare_ligand_from_pdb(fixed_atoms, model.lig_type_encoder)

    return coord, one_hot


def spea2(objective_values: torch.Tensor) -> torch.Tensor:
    """
    Calculates the SPEA2 fitness for a given set of objective values.

    Args:
        objective_values (np.array): A Pytorch tensor of shape (N, K), where N is the
                                     number of solutions and K is the number of objectives.
                                     It's assumed that lower values are better (minimization).

    Returns:
        np.array: A Pytorch tensor of shape (N,), containing the calculated fitness
                  value for each of the N solutions.
    """
    device = objective_values.device
    objective_values = objective_values.cpu().numpy()

    if not isinstance(objective_values, np.ndarray) or objective_values.ndim != 2:
        raise ValueError("Input 'objective_values' must be a 2D NumPy array.")

    n_solutions, n_objectives = objective_values.shape
    
    # --------------------------------------------------------------------------
    # 1. Calculate Raw Fitness (R)
    # --------------------------------------------------------------------------
    
    # Step 1.1: Calculate Strength (S) for each solution
    # Strength S(i) = number of solutions that solution i dominates.
    strength = np.zeros(n_solutions)
    for i in range(n_solutions):
        dominates_count = 0
        for j in range(n_solutions):
            if i == j:
                continue
            if s_ii.dominance_function(objective_values[i, :], objective_values[j, :]):
                dominates_count += 1
        strength[i] = dominates_count

    # Step 1.2: Calculate Raw Fitness (R)
    # Raw Fitness R(i) = sum of strengths of all solutions that dominate solution i.
    raw_fitness = np.zeros(n_solutions)
    for i in range(n_solutions):
        dominator_strength_sum = 0
        for j in range(n_solutions):
            if i == j:
                continue
            # Check if solution j dominates solution i
            if s_ii.dominance_function(objective_values[j, :], objective_values[i, :]):
                dominator_strength_sum += strength[j]
        raw_fitness[i] = dominator_strength_sum
        
    # --------------------------------------------------------------------------
    # 2. Calculate Density (D)
    # --------------------------------------------------------------------------
    
    # Step 2.1: Calculate pairwise Euclidean distances in the objective space
    distance_matrix = s_ii.euclidean_distance(objective_values)
    
    # Step 2.2: Calculate k for k-th nearest neighbor
    # Handle case where n_solutions is too small
    k = int(np.sqrt(n_solutions)) -1
    if k < 0:
        k = 0 # If N is 1, 2, or 3, k will be 0, meaning the nearest neighbor.
    
    # Step 2.3: Calculate density D(i) = 1 / (sigma_k + 2)
    density = np.zeros(n_solutions)
    for i in range(n_solutions):
        # Sort distances for the current solution i to all other solutions
        sorted_distances = np.sort(distance_matrix[i, :])
        
        # The first element is always 0 (distance to self), so we look at index k+1
        # However, the original s_ii.py implementation has a slight difference.
        # Let's stick to the logic of finding the k-th nearest *other* point.
        # If we sort all distances, sorted_distances[0] is d(i,i)=0.
        # The nearest neighbor is at sorted_distances[1].
        # The k-th nearest neighbor is at sorted_distances[k].
        # The s_ii.py code is a bit different, but this is the standard interpretation.
        # Let's check the original code's logic again:
        # distance_ordered = (distance[distance[:,i].argsort()]).T
        # fitness[i,0] = raw_fitness[i,0] + 1/(distance_ordered[i,k] + 2)
        # This sorts distances *to* i, then picks the k-th value. It's equivalent.
        
        # Ensure k is a valid index
        if k < len(sorted_distances):
            sigma_k = sorted_distances[k]
        else: # Fallback for very small populations
            sigma_k = sorted_distances[-1]

        density[i] = 1 / (sigma_k + 2)
        
    # --------------------------------------------------------------------------
    # 3. Calculate Final Fitness F = R + D
    # --------------------------------------------------------------------------
    final_fitness = raw_fitness + density

    final_fitness_torch = torch.from_numpy(final_fitness).to(device=device)
    
    return final_fitness_torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--pocket_pdbfile', type=str, default='example/5ndu_pocket.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_C_8V2.sdf')
    parser.add_argument('--objective', type=str, default="sa;qed;vina")
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=64)
    parser.add_argument('--evolution_steps', type=int, default=3000)
    parser.add_argument('--outfile', type=Path, default='output.sdf')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='egd', choices=['egd', 'sbdd-ea-mean', 'sbdd-ea-spea2'])
    parser.add_argument('--sbdd_top_k', type=int, default=4)

    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    name_str = args.mode

    run = wandb.init(
        project=f"sbdd-multi-objective",
        name=f"{name_str}-o={args.objective}-s{seed}",
        config=args,
    )

    pdb_id = Path(args.pdbfile).stem

    device = "cuda"

    sbdd_top_k = args.sbdd_top_k
    population_size = args.population_size
    evolution_steps = args.evolution_steps

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # Prepare ligand + pocket
    # Load PDB
    pdb_model = PDBParser(QUIET=True).get_structure('', args.pdbfile)[0]
    # Define pocket based on reference ligand
    residues = utils.get_pocket_from_ligand(pdb_model, args.ref_ligand)

    metrics = args.objective.split(";")
    objective_fn = Objective(metrics, args.pocket_pdbfile)

    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]

    # Population initialization
    ref_metric, ref_objective_value = objective_fn([ref_mol])
    buffer = [
        EvaluatedMolecule(ref_mol, ref_objective_value[0], ref_metric[0])
    ] * population_size
    
    for generation_idx in range(evolution_steps):

        # 1. Diversify molecules from the buffer
        molecules_to_diversify = [buffer_item.molecule for buffer_item in buffer]
        
        with torch.inference_mode():
            # Ensure the [molecules_to_diversify] is sorted from the best to the worst, the crossover will replace the 2nd half to the 1st half's offspring
            ligands_to_diversify = utils.prepare_ligands_from_mols(molecules_to_diversify, model.lig_type_encoder, device=model.device)
            diversified_molecules = model.generate_ligands(
                pdb_file=args.pdbfile,
                ref_ligand_path=args.ref_ligand,
                ref_ligand=ligands_to_diversify,
                n_samples=len(molecules_to_diversify),
                diversify_from_timestep=args.timesteps,
                sanitize=False,
                relax_iter=(200 if args.relax else 0),
                largest_frag=False,
                crossover=bool(args.mode == 'egd')
            )
        
        # 2. Evaluate, select, and update buffer
        raw_metrics, objective_values = objective_fn(diversified_molecules)

        candicates = [
            EvaluatedMolecule(mol, obj_values, raw_metric)
            for mol, obj_values, raw_metric in zip(diversified_molecules, objective_values, raw_metrics)
        ]

        log_molecules_objective_values(candicates, objectives_feedbacks=objective_fn.objectives_consumption, stage="candidates", commit=False)

        if args.mode == 'egd':
            # 3. Union with family, and select best molecules based on the objective values
            buffer.extend(candicates)
            fitness = spea2(torch.stack([c.objective_values for c in buffer]))
            sorted_indices = torch.argsort(fitness, descending=False)
            buffer = [buffer[i] for i in sorted_indices[:population_size]]
        elif args.mode == 'sbdd-ea-mean':
            # 3. Drop the parents
            buffer = candicates
            aggregated_objective_values = torch.stack([c.objective_values for c in buffer]).mean(dim=1)
            sorted_indices = torch.argsort(aggregated_objective_values, descending=False)
            buffer = [buffer[i] for i in sorted_indices[:sbdd_top_k]]
            while len(buffer) < population_size:
                buffer.extend(buffer)
            buffer = buffer[:population_size]
        elif args.mode == 'sbdd-ea-spea2':
            # 3. Drop the parents
            buffer = candicates
            aggregated_objective_values = spea2(torch.stack([c.objective_values for c in buffer]))
            sorted_indices = torch.argsort(aggregated_objective_values, descending=False)
            buffer = [buffer[i] for i in sorted_indices[:sbdd_top_k]]
            while len(buffer) < population_size:
                buffer.extend(buffer)
            buffer = buffer[:population_size]

        log_molecules_objective_values(candicates, objectives_feedbacks=objective_fn.objectives_consumption, stage="population", commit=True)