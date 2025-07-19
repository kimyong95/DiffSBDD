import argparse
from pathlib import Path
from rdkit import Chem
import einops
import math
from objective import Objective, METRIC_MAXIMIZE, METRIC_RANGE
import wandb
from constants import FLOAT_TYPE, INT_TYPE
import torch.nn.functional as F
from utils import seed_everything
import utils
import torch
from tqdm import tqdm
import einops
from contextlib import contextmanager
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List
from pymoo.indicators.hv import HV
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import AllChem

from torch_scatter import scatter_mean
from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule, process_molecule

#################### reserve fixed 3 GB memory for Gnina ######################
reserve_bytes = 3 * (1024**3)
reserve_fraction = reserve_bytes / torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(reserve_fraction, torch.cuda.current_device())
###############################################################################

from torch import nn
from value_model import ValueModel

def generate_systematic_weights(k: int, H: int) -> torch.Tensor:
    """
    Generates a uniform, grid-like set of weight vectors.

    This function implements the systematic approach by Das and Dennis to create
    structured weights for multi-objective optimization.

    Args:
        k (int): The number of objectives (dimension of the vectors).
        H (int): The number of divisions for the grid. Must be a positive integer.

    Returns:
        torch.Tensor: A tensor of shape (N, k) containing the weight vectors,
                      where N = C(H + k - 1, k - 1). Each vector sums to 1.0.
    """
    # 1. Input validation
    if not isinstance(k, int) or k <= 0 or not isinstance(H, int) or H <= 0:
        raise ValueError("Both k (objectives) and H (divisions) must be positive integers.")

    # 2. Recursive helper to find all lists of 'k' integers that sum to 'H'
    def _find_combinations(dim: int, total: int):
        # Base case: last dimension takes the remainder of the sum
        if dim == 1:
            yield [total]
            return
        
        # Recursive step: iterate through possible values for the current dimension
        for i in range(total + 1):
            for combo in _find_combinations(dim - 1, total - i):
                yield [i] + combo

    # 3. Generate integer combinations and convert to a tensor
    # The .flip() arranges the weights in a more intuitive descending order
    combinations = list(_find_combinations(k, H))
    weights = torch.tensor(combinations, dtype=torch.float32).flip(dims=(1,))

    # 4. Normalize by H to ensure the elements in each vector sum to 1
    return weights / H

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--pocket_pdbfile', type=str, default='example/5ndu_pocket.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_C_8V2.sdf')
    parser.add_argument('--objective', type=str, default='qed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sub_batch_size', type=int, default=4, help="Sub-batch size for sampling, should be smaller than batch_size.") 
    parser.add_argument('--diversify_from_timestep', type=int, default=100, help="diversify the [ref_ligand], lower timestep means closer to [ref_ligand], set -1 for no diversify (no reference ligand used).")
    parser.add_argument('--baseline', action='store_true', help="Use baseline sampling without selection mechanism.")
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    if args.diversify_from_timestep == -1:
        args.diversify_from_timestep = None

    baseline_str = "-(baseline)" if args.baseline else ""

    run = wandb.init(
        project=f"guide-sbdd",
        name=f"sample-and-select{baseline_str}-s={seed}-{args.objective}",
        config=args,
    )

    pdb_id = Path(args.pdbfile).stem

    device = "cuda"
    
    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    batch_size = args.batch_size
    sub_batch_size = args.sub_batch_size
    atom_dim = model.ddpm.n_dims + model.ddpm.atom_nf
    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]
    ref_ligand = utils.prepare_ligands_from_mols([ref_mol]*batch_size, model.lig_type_encoder, device=model.device)
    num_atoms = ref_mol.GetNumAtoms()
    num_nodes_lig = torch.ones(batch_size, dtype=int) * num_atoms
    metrics = args.objective.split(";")
    num_objectives = len(metrics)
    objective_fn = Objective(metrics, args.pocket_pdbfile)
    
    dimension = num_atoms * atom_dim

    concentration = torch.ones(num_objectives)
    dirichlet_dist = torch.distributions.Dirichlet(concentration)
    weights = dirichlet_dist.sample(sample_shape=(batch_size,)).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    @torch.inference_mode()
    def callback_func(pred_z0_lig, s, lig_mask, pocket, xh_pocket):
        """
        Callback function to be used during sampling.
        For each predicted z0, it generates a sub-batch of samples by noising z0 to zt
        and then taking one denoising step to zs.
        A selection mechanism should be implemented to choose one `zs` for each original molecule.
        """
        device = pred_z0_lig.device

        # 1. Expand the batch for sub-sampling
        num_nodes_lig = torch.unique(lig_mask, return_counts=True)[1]
        current_batch_size = len(num_nodes_lig)

        pred_z0_lig_list = utils.batch_to_list(pred_z0_lig, lig_mask)
        xh_pocket_list = utils.batch_to_list(xh_pocket, pocket['mask'])
        num_nodes_pocket = torch.unique(pocket['mask'], return_counts=True)[1]

        # `pocket` is the dictionary of clean pocket data (t=0)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        xh0_pocket_list = utils.batch_to_list(xh0_pocket, pocket['mask'])

        # Repeat each molecule's data `sub_batch_size` times
        expanded_pred_z0_lig = torch.cat([p for p in pred_z0_lig_list for _ in range(sub_batch_size)], dim=0)
        expanded_xh_pocket = torch.cat([p for p in xh_pocket_list for _ in range(sub_batch_size)], dim=0)
        expanded_xh0_pocket = torch.cat([p for p in xh0_pocket_list for _ in range(sub_batch_size)], dim=0)
        expanded_pocket_x = torch.cat([pocket['x'] for _ in range(sub_batch_size)], dim=0)

        # Create new masks for the expanded batch
        new_batch_size = current_batch_size * sub_batch_size
        new_num_nodes_lig = num_nodes_lig.repeat_interleave(sub_batch_size)
        new_lig_mask = utils.num_nodes_to_batch_mask(new_batch_size, new_num_nodes_lig, device=device)
        new_num_nodes_pocket = num_nodes_pocket.repeat_interleave(sub_batch_size)
        new_pocket_mask = utils.num_nodes_to_batch_mask(new_batch_size, new_num_nodes_pocket, device=device)

        # 2. For each predicted z0, create `sub_batch_size` noised versions `zt_given_z0` at time t=s+1
        t_int = s + 1
        t_array = torch.full((new_batch_size, 1), fill_value=t_int, device=device) / model.ddpm.T
        gamma_t = model.ddpm.inflate_batch_array(model.ddpm.gamma(t_array), expanded_pred_z0_lig)

        zt_lig, _, _ = model.ddpm.noised_representation(
            xh_lig=expanded_pred_z0_lig, xh0_pocket=expanded_xh0_pocket,
            lig_mask=new_lig_mask, pocket_mask=new_pocket_mask, gamma_t=gamma_t
        )

        # 3. For each `zt_given_z0`, take one denoising step to obtain `zs`
        s_array = torch.full((new_batch_size, 1), fill_value=s, device=device) / model.ddpm.T

        zs_lig, zs_pocket, pred_z0_given_zs_lig = model.ddpm.sample_p_zs_given_zt(
            s=s_array, t=t_array, zt_lig=zt_lig, xh0_pocket=expanded_xh_pocket,
            ligand_mask=new_lig_mask, pocket_mask=new_pocket_mask
        )

        #############################################
        # Build molecules from pred_z0_given_zs_lig #
        #############################################
        ndims = model.ddpm.n_dims

        pred_z0_given_zs_lig[:, :ndims], zs_pocket[:, :ndims] = \
            model.ddpm.remove_mean_batch(pred_z0_given_zs_lig[:, :ndims],
                                   zs_pocket[:, :ndims],
                                   new_lig_mask, new_pocket_mask)

        # Convert pred_z0_given_zs_lig to molecules for inspection
        x_lig, h_lig = model.ddpm.unnormalize(
            pred_z0_given_zs_lig[:, :model.ddpm.n_dims],
            pred_z0_given_zs_lig[:, model.ddpm.n_dims:],
        )
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), model.ddpm.atom_nf)

        model.ddpm.assert_mean_zero_with_mask(x_lig[:, :ndims], new_lig_mask)

        pocket_com_before = scatter_mean(expanded_pocket_x, new_pocket_mask, dim=0)
        pocket_com_after = scatter_mean(zs_pocket[:, :ndims], new_pocket_mask, dim=0)

        zs_pocket[:, :ndims] += \
            (pocket_com_before - pocket_com_after)[new_pocket_mask]
        x_lig += \
            (pocket_com_before - pocket_com_after)[new_lig_mask]

        atom_type = torch.argmax(h_lig, dim=1)

        molecules = []
        for mol_pc in zip(utils.batch_to_list(x_lig.cpu(), new_lig_mask),
                            utils.batch_to_list(atom_type.cpu(), new_lig_mask)):

            mol = build_molecule(mol_pc[0], mol_pc[1], model.dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                    add_hydrogens=False,
                                    sanitize=args.sanitize,
                                    relax_iter=(200 if args.relax else 0),
                                    largest_frag=not args.all_frags)
            if mol is not None:
                molecules.append(mol)
            else:
                molecules.append(None)

        success_indices = []
        success_molecules = []
        for i, mol in enumerate(molecules):
            if mol is not None:
                success_indices.append(i)
                success_molecules.append(mol)
        
        multi_objective_values = torch.full(
            (len(molecules), num_objectives), 
            float('inf'), 
            device=device, 
            dtype=FLOAT_TYPE
        )

        # [successful_objective_values] lower is better
        metrics_breakdown, successful_objective_values = objective_fn(success_molecules)
        multi_objective_values[success_indices, :] = successful_objective_values.to(device)

        select_indices = []
        select_lig_mask = []
        select_pocket_mask = []
        selected_multi_objective_values_list = []
        for i in range(batch_size):
            weight_i = weights[i][None,:]
            aggregated_objective_values = (multi_objective_values * weight_i).sum(1)
            select_index = aggregated_objective_values.argmin().item()
            selected_multi_objective_values_list.append(multi_objective_values[select_index].clone())
            multi_objective_values[select_index, :] = float('inf')  # Mark as used

            select_indices.append(select_index)
            select_lig_mask.append( (new_lig_mask == select_index).nonzero().squeeze(1) )
            select_pocket_mask.append( (new_pocket_mask == select_index).nonzero().squeeze(1) )
        
        select_lig_mask = torch.stack(select_lig_mask).flatten()
        select_pocket_mask = torch.stack(select_pocket_mask).flatten()

        final_zs_lig = zs_lig[select_lig_mask]
        final_zs_pocket = expanded_xh_pocket[select_pocket_mask]

        selected_molecules = [ molecules[i] for i in select_indices ]
        molecules_diversity = calculate_molecules_diversity(selected_molecules)

        selected_multi_objective_values = torch.stack(selected_multi_objective_values_list)

        hypervolume = calculate_hypervolume(selected_multi_objective_values)
        pareto_front = get_pareto_front(selected_multi_objective_values)

        log_dict = {
            "intermediate/hypervolume": hypervolume,
            "intermediate/number_of_pareto": len(pareto_front),
            "intermediate/feasible_mol_rate": len(success_indices) / (batch_size * sub_batch_size),
            "intermediate/diversity": molecules_diversity,
            "objectives_feedbacks": objective_fn.objectives_consumption,
        }

        for k, v in metrics_breakdown.items():
            log_dict[f"intermediate/{k}_mean"] = torch.tensor(v).mean()
            if METRIC_MAXIMIZE[k]:
                best = torch.tensor(v).max()
            else:
                best = torch.tensor(v).min()
            log_dict[f"intermediate/{k}_best"] = best
        wandb.log(log_dict)

        if args.baseline:
            return {}
        else:
            return {"z_lig": final_zs_lig, "xh_pocket": final_zs_pocket}

    with torch.inference_mode():
        molecules, pred_z0_lig_traj = model.generate_ligands(
            args.pdbfile, batch_size, args.resi_list, args.ref_ligand, ref_ligand,
            num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
            relax_iter=(200 if args.relax else 0),
            diversify_from_timestep=args.diversify_from_timestep,
            callback=callback_func,
        )
    assert len(molecules) == batch_size

    success_indices = []
    success_molecules = []
    for i, mol in enumerate(molecules):
        if mol is not None:
            success_indices.append(i)
            success_molecules.append(mol)

    # [objective_values] lower is better
    metrics_breakdown, objective_values = objective_fn(success_molecules)
    molecules_diversity = calculate_molecules_diversity(success_molecules)



    # Plot objectives and calculate hypervolume
    hypervolume = calculate_hypervolume(objective_values)
    pareto_front = get_pareto_front(objective_values)

    # Log final pareto front
    objective_names = list(metrics_breakdown.keys())
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            plot_data = pareto_front[:, [i, j]].cpu().numpy()
            table = wandb.Table(data=plot_data, columns=[objective_names[i], objective_names[j]])
            log_payload = {
                f"final/pareto_front/{objective_names[i]}_vs_{objective_names[j]}": wandb.plot.scatter(
                    table, objective_names[i], objective_names[j],
                    title=f"Pareto Front: {objective_names[i]} vs {objective_names[j]}"
                )
            }
            wandb.log(log_payload, commit=False)

    # wandb log the score lower better (final log)
    log_dict = {
        "final/hypervolume": hypervolume,
        "final/feasible_mol_rate": len(success_indices) / batch_size,
        "final/diversity": molecules_diversity,
        "objectives_feedbacks": objective_fn.objectives_consumption,
        "final/number_of_pareto": len(pareto_front),
    }
    for k, v in metrics_breakdown.items():
        log_dict[f"final/{k}_mean"] = torch.tensor(v).mean()
        if METRIC_MAXIMIZE[k]:
            best = torch.tensor(v).max()
        else:
            best = torch.tensor(v).min()
        log_dict[f"final/{k}_best"] = best
    wandb.log(log_dict)
