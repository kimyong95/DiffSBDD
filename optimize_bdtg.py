import argparse
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
import pandas as pd
import random
import wandb
from collections import defaultdict
from objective import Objective, METRIC_MAXIMIZE
import einops
from torch_scatter import scatter_mean
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM
from constants import FLOAT_TYPE, INT_TYPE
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.metrics import MoleculeProperties
from sbdd_metrics.metrics import FullEvaluator

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


def prepare_ligands_from_mols(mols, atom_encoder, device='cpu'):

    ligand_coords = []
    atom_one_hots = []
    masks = []
    sizes = []
    for i, mol in enumerate(mols):
        coord = torch.tensor(mol.GetConformer().GetPositions(), dtype=FLOAT_TYPE)
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in mol.GetAtoms()], dtype=INT_TYPE)
        one_hot = F.one_hot(types, num_classes=len(atom_encoder))
        mask = torch.ones(len(types), dtype=INT_TYPE) * i
        ligand_coords.append(coord)
        atom_one_hots.append(one_hot)
        masks.append(mask)
        sizes.append(len(types))

    ligand = {
        'x': torch.cat(ligand_coords, dim=0).to(device),
        'one_hot': torch.cat(atom_one_hots, dim=0).to(device),
        'size': torch.tensor(sizes, dtype=INT_TYPE).to(device),
        'mask': torch.cat(masks, dim=0).to(device),
    }

    return ligand


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


def generate_ligands(model, pocket, mols, given_noise_list,
                    sanitize=False,
                    largest_frag=False,
                    relax_iter=0):
    """
    Diversify ligands for a specified pocket.
    
    Parameters:
        model: The model instance used for diversification.
        pocket: The pocket information including coordinates and types.
        mols: List of RDKit molecule objects to be diversified.
        sanitize: If True, performs molecule sanitization post-generation (default: False).
        largest_frag: If True, only the largest fragment of the generated molecule is returned (default: False).
        relax_iter: Number of iterations for force field relaxation of the generated molecules (default: 0).
    
    Returns:
        A list of diversified RDKit molecule objects.
    """
    batch_size = len(mols)
    ligand = prepare_ligands_from_mols(mols, model.lig_type_encoder, device=model.device)

    pocket_mask = pocket['mask']
    lig_mask = ligand['mask']

    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    out_lig, out_pocket, _, _ = model.ddpm.sample_given_pocket(pocket, ligand["size"], given_noise_list=given_noise_list)

    # Move generated molecule back to the original pocket position
    pocket_com_after = scatter_mean(out_pocket[:, :model.x_dims], pocket_mask, dim=0)

    out_pocket[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[pocket_mask]
    out_lig[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[lig_mask]

    # Build mol objects
    x = out_lig[:, :model.x_dims].detach().cpu()
    atom_type = out_lig[:, model.x_dims:].argmax(1).detach().cpu()

    molecules = []
    for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                      utils.batch_to_list(atom_type, lig_mask)):

        mol = build_molecule(*mol_pc, model.dataset_info, add_coords=True)
        mol = process_molecule(mol,
                               add_hydrogens=False,
                               sanitize=sanitize,
                               relax_iter=relax_iter,
                               largest_frag=largest_frag)
        if mol is not None:
            molecules.append(mol)
        else:
            molecules.append(None)

    return molecules

def update_parameters(mu, sigma, noise, scores):
    ''' minimize score '''

    # noise: (T, B, D)
    # scores: (T, B)

    assert noise.shape[0] == scores.shape[0] == mu.shape[0] == sigma.shape[0]
    assert noise.shape[1] == scores.shape[1]

    T_dim = noise.shape[0]
    B_dim = noise.shape[1]
    D_dim = mu.shape[1]

    lr_mu = math.sqrt(D_dim)
    lr_sigma = 1
    

    mu = mu.clone()
    sigma = sigma.clone()

    for t in range(T_dim):
        
        scores_t = scores[t:].nanmean(0)
        valid_mask = ~torch.isnan(scores_t)
        scores_t = scores_t[valid_mask]
        z_t = noise[t][valid_mask]
        scores_t_normalized = (scores_t - scores_t.mean()) / (scores_t.std() + 1e-8)

        w = torch.exp( - scores_t_normalized) / torch.exp( - scores_t_normalized).sum()

        sigma[t] = 1 / (
            
            1/sigma[t] + lr_sigma / math.sqrt(D_dim) * (
                (1/sigma[t])[None,:] * (z_t - mu[t,None,:]) * (z_t - mu[t,None,:]) * (1/sigma[t])[None,:] * \
                
                w[:,None]

            # sum over N
            ).sum(0)
        )
        
        mu[t] = mu[t] - lr_mu / math.sqrt(D_dim) * (

            (z_t - mu[t][None,:]) * \
            
            scores_t_normalized[:,None]
        
        # mean over N
        ).mean(0)

    return mu, sigma

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--pocket_pdbfile', type=str, default='example/5ndu_pocket.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_C_8V2.sdf')
    parser.add_argument('--objective', type=str, default='qed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimization_steps', type=int, default=300)
    parser.add_argument('--outfile', type=Path, default='output.sdf')
    parser.add_argument('--relax', action='store_true')

    args = parser.parse_args()

    run = wandb.init(
        project="guide-sbdd",
        name=f"bdtg-{args.objective}",
    )

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size
    optimization_steps = args.optimization_steps

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # Prepare ligand + pocket
    # Load PDB
    pdb_model = PDBParser(QUIET=True).get_structure('', args.pdbfile)[0]
    # Define pocket based on reference ligand
    residues = utils.get_pocket_from_ligand(pdb_model, args.ref_ligand)
    pocket = model.prepare_pocket(residues, repeats=batch_size)

    metrics = args.objective.split(";")
    objective_fn = Objective(metrics, args.pocket_pdbfile)

    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]

    # guidance parameters
    num_atoms = len(ref_mol.GetAtoms())
    atom_dim = model.ddpm.n_dims + model.ddpm.atom_nf
    mu = torch.zeros(model.ddpm.T+1, num_atoms * atom_dim, dtype=torch.float32).to(device)
    sigma = torch.ones(model.ddpm.T+1, num_atoms * atom_dim, dtype=torch.float32).to(device)

    ref_objective_value, _ = objective_fn([ref_mol])
    print(f"Reference molecule objective value (lower better): {ref_objective_value[0]}")

    for optimization_idx in range(optimization_steps):

        batch_mu = einops.repeat(mu, 'T D -> T B D', B=batch_size)
        batch_sigma = einops.repeat(sigma, 'T D -> T B D', B=batch_size)
        batch_noise = batch_mu + batch_sigma**0.5 * torch.randn_like(batch_mu)
        batch_noise_norm = batch_noise.norm(dim=-1)
        batch_noise_projected = batch_noise / batch_noise_norm[:,:,None] * batch_noise_norm[:,:,None]
        given_noise_list = einops.rearrange(batch_noise_projected, 'T B (M N) -> T (B M) N', B=batch_size, M=num_atoms, N=atom_dim)
        
        molecules = generate_ligands(
            model,
            pocket,
            [ref_mol] * batch_size,
            given_noise_list=given_noise_list,
            sanitize=True,
            relax_iter=(200 if args.relax else 0),
        )
        success_indices = []
        success_molecules = []
        for i, mol in enumerate(molecules):
            if mol is not None:
                success_indices.append(i)
                success_molecules.append(mol)
        batch_noise = batch_noise[:,success_indices,:]
        
        # Evaluate and save molecules
        objective_values = torch.zeros((model.ddpm.T+1, len(success_indices)), dtype=torch.float32).to(device)
        objective_values_ , metrics_breakdown = objective_fn(success_molecules)
        objective_values[:] = objective_values_
            
        # want to minimize objective_values
        mu, sigma = update_parameters(mu, sigma, batch_noise, objective_values)

        # wandb log the score lower better
        log_dict = {
            "train/score_mean": objective_values[-1].mean(),
            "train/score_best": objective_values[-1].min(),
            "step": optimization_idx,
            "train/feasible_mol_rate": len(success_indices) / batch_size,
            "|mu|_2": mu.norm().item(),
            "|sigma|_2": sigma.sum(-1).mean().item(),
        }
        for k, v in metrics_breakdown.items():
            log_dict[f"train/{k}_mean"] = torch.tensor(v).mean()
            # Maximize the metric
            if METRIC_MAXIMIZE[k]:
                best = torch.tensor(v).max()
            else:
                best = torch.tensor(v).min()
            log_dict[f"train/{k}_best"] = best
        wandb.log(log_dict)

