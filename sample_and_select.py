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

def is_number(s):
  """
  Checks if a string is a number, including negatives and decimals.
  """
  try:
    float(s) # Use float() to handle both integers and decimals
    return True
  except ValueError:
    return False

def aggregate_objectives(multi_objective_values, shift_constants, weight_lambda):
    """
    Aggregate multi-objective values using a specified function.

    Args:
        multi_objective_values (torch.Tensor):
            A tensor of shape (N, K) where N is the number of points and K is the number of objectives.
            Lower is better.

    Returns:
        torch.Tensor: A tensor of aggregated values.
    """
    
    aggregated_objective_value = - torch.logsumexp( - (multi_objective_values - shift_constants[None,:]) / weight_lambda, dim=1 )

    return aggregated_objective_value

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
    parser.add_argument('--lambda_normalization', type=str, default="l2")
    parser.add_argument('--ea_optimize_steps', default=0, type=int, help="Number of steps to optimize using evolutionary algorithm. Set to 0 to disable.")


    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    if args.diversify_from_timestep == -1:
        args.diversify_from_timestep = None

    run = wandb.init(
        project=f"sbdd-multi-objective",
        name=f"sample-and-select-s={seed}-c={args.shift_constants}-{args.lambda_normalization}-{args.objective}",
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
    
    lambda_ = torch.randn((batch_size, num_objectives), device=device, generator=generator).abs()
    if args.lambda_normalization == "l2":
        lambda_ = lambda_ / lambda_.norm(dim=1, p=2, keepdim=True)
    elif args.lambda_normalization == "l1":
        lambda_ = lambda_ / lambda_.norm(dim=1, p=1, keepdim=True)

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
                                    sanitize=True,
                                    relax_iter=(200 if args.relax else 0),
                                    largest_frag=False)
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
        successful_raw_metrics, successful_objective_values = objective_fn(success_molecules)
        multi_objective_values[success_indices, :] = successful_objective_values.to(device)
        raw_metrics = [None] * len(molecules)
        for i, m in zip(success_indices, successful_raw_metrics):
            raw_metrics[i] = m

        if args.shift_constants == "best":                
            shift_constants[:] = torch.minimum(shift_constants, multi_objective_values.min(dim=0).values)
        elif args.shift_constants == "worst":
            shift_constants[:] = torch.maximum(shift_constants, multi_objective_values.max(dim=0).values)
        elif args.shift_constants == "mean":
            shift_constants[:] = multi_objective_values.mean(dim=0)

        select_indices = []
        select_lig_mask = []
        select_pocket_mask = []
        multi_objective_values_candidates = multi_objective_values.clone()
        for i in range(batch_size):
            lambda_i = lambda_[i, :]
            aggregated_objective_values = aggregate_objectives(multi_objective_values_candidates, shift_constants, lambda_i)
            select_index = aggregated_objective_values.argmin().item()
            multi_objective_values_candidates[select_index, :] = float('inf')  # Mark as used

            select_indices.append(select_index)
            select_lig_mask.append( (new_lig_mask == select_index).nonzero().squeeze(1) )
            select_pocket_mask.append( (new_pocket_mask == select_index).nonzero().squeeze(1) )
        del multi_objective_values_candidates
        select_lig_mask = torch.stack(select_lig_mask).flatten()
        select_pocket_mask = torch.stack(select_pocket_mask).flatten()

        final_zs_lig = zs_lig[select_lig_mask]
        final_zs_pocket = expanded_xh_pocket[select_pocket_mask]

        selected_molecules = [
            EvaluatedMolecule(molecules[i], multi_objective_values[i], raw_metrics[i])
            for i in select_indices
        ]

        log_molecules_objective_values(
            selected_molecules, 
            objectives_feedbacks=objective_fn.objectives_consumption,
            stage=f"intermediate",
        )

        return {"z_lig": final_zs_lig, "xh_pocket": final_zs_pocket}

    with torch.inference_mode():
        molecules, pred_z0_lig_traj = model.generate_ligands(
            args.pdbfile, batch_size, None, args.ref_ligand, ref_ligand,
            num_nodes_lig, sanitize=True, largest_frag=False,
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
    raw_metrics, objective_values = objective_fn(success_molecules)
    selected_molecules = [
        EvaluatedMolecule(mol, obj_values, raw_metric)
        for mol, obj_values, raw_metric in zip(success_molecules, objective_values, raw_metrics)
    ]
    
    log_molecules_objective_values(
        selected_molecules, 
        objectives_feedbacks=objective_fn.objectives_consumption,
        stage=f"final",
    )