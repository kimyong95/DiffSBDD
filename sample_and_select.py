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
from utils_moo import EvaluatedMolecule, log_molecules_objective_values
from rdkit import Chem

from rdkit.Chem import AllChem
from scipy.stats import qmc
from scipy.special import gammaincinv

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

def mu_ts_to_zs(mu_lig, xh0_pocket, lig_mask, pocket_mask, t, s, model):
    
    gamma_s = model.ddpm.gamma(s)
    gamma_t = model.ddpm.gamma(t)

    sigma_s = model.ddpm.sigma(gamma_s, target_tensor=mu_lig)
    sigma_t = model.ddpm.sigma(gamma_t, target_tensor=mu_lig)

    sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
        model.ddpm.sigma_and_alpha_t_given_s(gamma_t, gamma_s, mu_lig)

    # Compute sigma for p(zs | zt).
    sigma = sigma_t_given_s * sigma_s / sigma_t

    # Sample zs given the parameters derived from zt.
    zs_lig, xh0_pocket = model.ddpm.sample_normal_zero_com(mu_lig, xh0_pocket, sigma, lig_mask, pocket_mask)

    # The zs_lig should have mean zero.
    return zs_lig, xh0_pocket

def zt_to_xh(zt_lig, xh_pocket, lig_mask, pocket_mask, t, model, with_noise):

    batch_size = t.shape[0]
    s0 = torch.zeros_like(t)
    noise0 = torch.zeros_like(zt_lig) if not with_noise else None
    
    if t[0].item() > 0:
        z0_lig, xh0_pocket, _ = model.ddpm.sample_p_zs_given_zt(s0, t, zt_lig, xh_pocket, lig_mask, pocket_mask, given_noise=noise0)
        x_lig, h_lig, x_pocket, h_pocket = model.ddpm.sample_p_xh_given_z0(z0_lig, xh0_pocket, lig_mask, pocket_mask, batch_size, given_noise=noise0)
    else:
        x_lig, h_lig, x_pocket, h_pocket = model.ddpm.sample_p_xh_given_z0(zt_lig, xh_pocket , lig_mask, pocket_mask, batch_size, given_noise=noise0)

    model.ddpm.assert_mean_zero_with_mask(x_lig, lig_mask)

    return x_lig, h_lig, x_pocket, h_pocket

def shift_x_lig_back_to_pocket_com_before(
        x_lig, lig_mask,
        original_pocket, new_pocket, pocket_mask
    ):

    pocket_com_before = scatter_mean(original_pocket, pocket_mask, dim=0)
    pocket_com_after = scatter_mean(new_pocket, pocket_mask, dim=0)

    x_lig = x_lig + (pocket_com_before - pocket_com_after)[lig_mask]

    return x_lig

def is_number(s):
  """
  Checks if a string is a number, including negatives and decimals.
  """
  try:
    float(s) # Use float() to handle both integers and decimals
    return True
  except ValueError:
    return False

def aggregate_objectives(multi_objective_values, shift_constants, weight_lambda, mode):
    """
    Aggregate multi-objective values using a specified function.

    Args:
        multi_objective_values (torch.Tensor):
            A tensor of shape (N, K) where N is the number of points and K is the number of objectives.
            Lower is better.

    Returns:
        torch.Tensor: A tensor of aggregated values.
    """

    if mode == "max":
        aggregated_objective_value = torch.max(
            (multi_objective_values - shift_constants[None, :]) / weight_lambda[None, :],
            dim=1
        ).values
    elif mode == "logsumexp":
        aggregated_objective_value = torch.logsumexp(
            (multi_objective_values - shift_constants[None, :]) / weight_lambda[None, :],
            dim=1
        )
    elif mode == "neglogsumexp":
        aggregated_objective_value = - torch.logsumexp(
            - (multi_objective_values - shift_constants[None, :]) / weight_lambda[None, :],
            dim=1
        )

    return aggregated_objective_value

def _generate_weights_recursive(k: int, p: int):
    """
    Private recursive generator for integer combinations.
    Yields all lists of k non-negative integers that sum to p.
    """
    # Base case: if only one objective is left, its value must be the remainder.
    if k == 1:
        yield [p]
        return

    # Recursive step: iterate through possible values for the current objective.
    for i in range(p + 1):
        # Generate combinations for the remaining k-1 objectives with the remaining sum p-i.
        for sub_combination in _generate_weights_recursive(k - 1, p - i):
            yield [i] + sub_combination

def generate_simplex_lattice_weights(
    k: int, 
    p: int, 
    dtype: torch.dtype = torch.float32, 
    device: torch.device = None
) -> torch.Tensor:
    """
    Generates evenly distributed weight vectors using the Simplex-Lattice Design.

    Args:
        k (int): The number of objectives (dimension of the weight vector).
        p (int): The number of divisions for the simplex.
        dtype (torch.dtype, optional): The desired data type of the output tensor. 
                                       Defaults to torch.float32.
        device (torch.device, optional): The desired device of the output tensor. 
                                         Defaults to the default device.

    Returns:
        torch.Tensor: A tensor of shape (n, k) where n is the number of
                      generated weights, and each row is a weight vector 
                      summing to 1.
    """
    if k < 1 or p < 0:
        raise ValueError("k must be >= 1 and p must be >= 0")
    
    # 1. Generate integer combinations that sum to p
    integer_combinations = list(_generate_weights_recursive(k, p))
    
    weights_tensor = torch.tensor(
        integer_combinations, 
        dtype=dtype, 
        device=device
    )

    weights_tensor /= p
    
    return weights_tensor

def check_n_feasibility(n_target: int, k: int) -> tuple[bool, int]:
    """
    Checks if a target number of vectors 'n' is feasible for 'k' objectives.

    Args:
        n_target (int): The desired number of weight vectors.
        k (int): The number of objectives.

    Returns:
        tuple[bool, int]: A tuple containing:
                          - A boolean indicating if n_target is feasible.
                          - The required number of divisions 'p' if feasible, 
                            otherwise -1.
    """
    if n_target < 1 or k < 1:
        return (False, -1)

    p = 0
    while True:
        # Calculate n_calc = C(p + k - 1, k - 1)
        try:
            n_calc = math.comb(p + k - 1, k - 1)
        except ValueError:
            # This can happen if p=0 and k=1, where C(-1,0) is invalid for math.comb
            # but the result should be 1. Let's handle it manually.
            if p + k - 1 < k - 1:
                 n_calc = 0 if n_target != 1 else 1
            else: # Should not happen in normal flow
                return(False,-1)

        if n_calc == n_target:
            return (True, p)
        
        if n_calc > n_target:
            return (False, -1)
        
        p += 1

def get_lambda(batch_size: int, num_objectives: int, mode, device):
    if mode == "l2":
        lambda_ = torch.randn((batch_size, num_objectives), device=device, generator=generator).abs()
        lambda_ = lambda_ / lambda_.norm(dim=1, p=2, keepdim=True)
    elif mode == "l1":
        lambda_ = torch.randn((batch_size, num_objectives), device=device, generator=generator).abs()
        lambda_ = lambda_ / lambda_.norm(dim=1, p=1, keepdim=True)
    elif mode == "simplex":
        feasible, p = check_n_feasibility(batch_size, num_objectives)
        assert feasible, f"Cannot generate {batch_size} weight vectors for {num_objectives} objectives when using simplex."
        lambda_inv = generate_simplex_lattice_weights(num_objectives, p, device=device).clamp(min=1e-7)
        lambda_ = 1.0 / lambda_inv
        lambda_ = lambda_ / lambda_.sum(dim=1, keepdim=True)

    return lambda_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--pocket_pdbfile', type=str, default='example/5ndu_pocket.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_C_8V2.sdf')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--objective', type=str, default="sa;qed;vina")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--sub_batch_size', type=int, default=4, help="Sub-batch size for sampling, should be smaller than batch_size.") 
    parser.add_argument('--diversify_from_timestep', type=int, default=100, help="diversify the [ref_ligand], lower timestep means closer to [ref_ligand], set -1 for no diversify (no reference ligand used).")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shift_constants', type=str, default="0")
    parser.add_argument('--lambda_mode', type=str, default="l2")
    parser.add_argument('--aggre_mode', type=str, default="neglogsumexp")
    parser.add_argument('--ea_optimize_steps', default=0, type=int, help="Number of steps to optimize using evolutionary algorithm. Set to 0 to disable.")
    parser.add_argument('--with_noise', action='store_true', help="Whether to add noise to the generated ligands.")

    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    if args.diversify_from_timestep == -1:
        args.diversify_from_timestep = None

    with_noise_str = "t" if args.with_noise else "f"

    run = wandb.init(
        project=f"sbdd-multi-objective",
        name=f"sample-and-select-ag={args.aggre_mode}-c={args.shift_constants}-n={with_noise_str}-l={args.lambda_mode}-b={args.batch_size}:{args.sub_batch_size}-o={args.objective}-s={seed}",
        config=args,
    )

    pdb_id = Path(args.pdbfile).stem

    device = "cuda"
    
    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    ea_optimize_steps = args.ea_optimize_steps
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

    generator = torch.Generator(device=device).manual_seed(seed)
    
    torch.manual_seed(seed)

    lambda_ = get_lambda(batch_size, num_objectives, args.lambda_mode, device)

    shift_constants = torch.zeros((num_objectives,), dtype=torch.float32, device=device)
    if is_number(args.shift_constants):
        shift_constants[:] = float(args.shift_constants)
    elif args.shift_constants == "best":
        shift_constants[:] = float('inf')
    elif args.shift_constants == "worst":
        shift_constants[:] = float('-inf')
    elif args.shift_constants == "mean":
        shift_constants[:] = 0.0
    
    else:
        raise ValueError(f"Unknown shift_constants: {args.shift_constants}")
    
    @torch.inference_mode()
    def callback_func(mu_ts_lig, xh_pocket_t, s, lig_mask, pocket):
        """
        Callback function to be used during sampling.
        For each predicted z0, it generates a sub-batch of samples by noising z0 to zt
        and then taking one denoising step to zs.
        A selection mechanism should be implemented to choose one `zs` for each original molecule.
        """

        device = mu_ts_lig.device
        
        pocket_mask = pocket["mask"]
        original_pocket = pocket["x"]
        num_nodes_lig = torch.unique(lig_mask, return_counts=True)[1]
        batch_size = len(num_nodes_lig)

        t = s + 1
        t_array = torch.full((batch_size, 1), fill_value=t, device=device) / model.ddpm.T
        s_array = torch.full((batch_size, 1), fill_value=s, device=device) / model.ddpm.T

        
        zs_all = []
        xh_pocket_s_all = []

        xh_given_zs_all = []
        lig_mask_all = []
        pocket_mask_all = []
        for sb in range(sub_batch_size):

            zs, xh_pocket_s = mu_ts_to_zs(mu_ts_lig, xh_pocket_t, lig_mask, pocket_mask, t_array, s_array, model)
            
            # To be selected and returned
            zs_all.append(zs)
            xh_pocket_s_all.append(xh_pocket_s)

            x_lig, h_lig, x_pocket_s, h_pocket_s = zt_to_xh(zs, xh_pocket_s, lig_mask, pocket_mask, s_array, model, with_noise=args.with_noise)
            x_lig = shift_x_lig_back_to_pocket_com_before(
                x_lig = x_lig,
                lig_mask = lig_mask,
                original_pocket = original_pocket,
                new_pocket = x_pocket_s,
                pocket_mask = pocket_mask
            )

            # To be evaluated for selection
            xh = torch.cat([x_lig, h_lig], dim=1)
            xh_given_zs_all.append(xh)
            lig_mask_all.append(lig_mask + batch_size * sb)
            pocket_mask_all.append(pocket_mask + batch_size * sb)
        
        zs_all = torch.cat(zs_all, dim=0)
        xh_pocket_s_all = torch.cat(xh_pocket_s_all, dim=0)
        xh_given_zs_all = torch.cat(xh_given_zs_all, dim=0)
        lig_mask_all = torch.cat(lig_mask_all, dim=0)
        pocket_mask_all = torch.cat(pocket_mask_all, dim=0)

        ############################################################
        # Build molecules from x0_given_zt_lig and h0_given_zt_lig #
        ############################################################

        molecules = []
        for mol_pc in zip(
            utils.batch_to_list(xh_given_zs_all[:,:3].cpu(), lig_mask_all),
            utils.batch_to_list(xh_given_zs_all[:,3:].argmax(dim=1).cpu(), lig_mask_all)
        ):
            mol = build_molecule(mol_pc[0], mol_pc[1], model.dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                    add_hydrogens=False,
                                    sanitize=False,
                                    relax_iter=(200 if args.relax else 0),
                                    largest_frag=False)
            molecules.append(mol)

        # [successful_objective_values] lower is better
        raw_metrics, objective_values = objective_fn(molecules)
        objective_values = objective_values.to(device)

        if args.shift_constants == "best":                
            shift_constants[:] = torch.minimum(shift_constants, objective_values.min(dim=0).values)
        elif args.shift_constants == "worst":
            shift_constants[:] = torch.maximum(shift_constants, objective_values.max(dim=0).values)
        elif args.shift_constants == "mean":
            shift_constants[:] = objective_values[~objective_values.isinf().any(dim=1)].mean(dim=0)
        
        select_indices = []
        select_lig_mask = []
        select_pocket_mask = []
        objective_values_candidates = objective_values.clone()
        for i in range(batch_size):
            lambda_i = lambda_[i, :]
            aggregated_objective_values = aggregate_objectives(objective_values_candidates, shift_constants, lambda_i, mode=args.aggre_mode)
            select_index = aggregated_objective_values.argmin().item()
            objective_values_candidates[select_index, :] = float('inf')  # Mark as used

            select_indices.append(select_index)
            select_lig_mask.append( (lig_mask_all == select_index).nonzero().squeeze(1) )
            select_pocket_mask.append( (pocket_mask_all == select_index).nonzero().squeeze(1) )
        del objective_values_candidates
        select_lig_mask = torch.stack(select_lig_mask).flatten()
        select_pocket_mask = torch.stack(select_pocket_mask).flatten()

        selected_zs_lig = zs_all[select_lig_mask]
        selected_xh_pocket_s = xh_pocket_s_all[select_pocket_mask]

        selected_molecules = [
            EvaluatedMolecule(molecules[i], objective_values[i], raw_metrics[i])
            for i in select_indices
        ]

        log_molecules_objective_values(
            selected_molecules, 
            objectives_feedbacks=objective_fn.objectives_consumption,
            stage=f"intermediate",
        )

        return {"z_lig": selected_zs_lig, "xh_pocket": selected_xh_pocket_s}

    with torch.inference_mode():
        molecules = model.generate_ligands(
            args.pdbfile, batch_size, None, args.ref_ligand, ref_ligand,
            num_nodes_lig, sanitize=False, largest_frag=False,
            relax_iter=(200 if args.relax else 0),
            diversify_from_timestep=args.diversify_from_timestep,
            # callback=callback_func,
        )
    assert len(molecules) == batch_size

    # [objective_values] lower is better
    raw_metrics, objective_values = objective_fn(molecules)
    evaluated_molecules = [
        EvaluatedMolecule(mol, obj_values, raw_metric)
        for mol, obj_values, raw_metric in zip(molecules, objective_values, raw_metrics)
    ]
    
    log_molecules_objective_values(
        evaluated_molecules, 
        objectives_feedbacks=objective_fn.objectives_consumption,
        stage=f"final",
    )