import argparse
from pathlib import Path
from rdkit import Chem
import einops
import math
from objective import Objective, METRIC_MAXIMIZE
import wandb
from constants import FLOAT_TYPE, INT_TYPE
import torch.nn.functional as F
from utils import seed_everything
import utils
import torch
from tqdm import tqdm
import einops
import pandas as pd
from contextlib import contextmanager
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

from lightning_modules import LigandPocketDDPM

#################### reserve fixed 3 GB memory for Gnina ######################
reserve_bytes = 3 * (1024**3)
reserve_fraction = reserve_bytes / torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(reserve_fraction, torch.cuda.current_device())
###############################################################################

from torch import nn
from value_model import ValueModel

def update_parameters(mu, sigma, noise, scores):
    ''' minimize score '''

    # noise: (T, B, D)
    # scores: (T, B)

    assert noise.shape[0] == scores.shape[0] == mu.shape[0] == sigma.shape[0]
    assert noise.shape[1] == scores.shape[1]

    T_dim = noise.shape[0]
    B_dim = noise.shape[1]
    D_dim = mu.shape[1]

    lr_mu = 1
    lr_sigma = 1/128
    

    mu = mu.clone()
    sigma = sigma.clone()

    for t in range(T_dim):
        
        scores_t = scores[t:].mean(0)
        z_t = noise[t]
        scores_t_normalized = (scores_t - scores_t.mean()) / scores_t.std().clamp(min=1e-8)
        
        scores_t_softmaxed = nn.Softmax(dim=0)(-scores_t_normalized)


        sigma[t] = 1 / (
            
            1/sigma[t] + lr_sigma * (
                (1/sigma[t])[None,:] * (z_t - mu[t,None,:]) * (z_t - mu[t,None,:]) * (1/sigma[t])[None,:] * \
                
                scores_t_softmaxed[:,None]

            # sum over N
            ).sum(0)
        )
        
        mu[t] = mu[t] - lr_mu * (

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
    parser.add_argument('--optimization_steps', type=int, default=1000)
    parser.add_argument('--diversify_from_timestep', type=int, default=100, help="diversify the [ref_ligand], lower timestep means closer to [ref_ligand], set -1 for no diversify (no reference ligand used).")

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

    run = wandb.init(
        project=f"guide-sbdd",
        name=f"bdtg-gp-s{seed}-{args.objective}",
        config=args,
    )

    pdb_id = Path(args.pdbfile).stem

    device = "cuda"
    
    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    batch_size = args.batch_size
    atom_dim = model.ddpm.n_dims + model.ddpm.atom_nf
    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]
    ref_ligand = utils.prepare_ligands_from_mols([ref_mol]*batch_size, model.lig_type_encoder, device=model.device)
    num_atoms = ref_mol.GetNumAtoms()
    num_nodes_lig = torch.ones(batch_size, dtype=int) * num_atoms
    metrics = args.objective.split(";")
    objective_fn = Objective(metrics, args.pocket_pdbfile)
    
    dimension = num_atoms * atom_dim
    num_parameters = model.ddpm.T+2 if args.diversify_from_timestep is None else args.diversify_from_timestep+2
    mu = torch.zeros(num_parameters, dimension, dtype=torch.float32).to(device)
    sigma = torch.ones(num_parameters, dimension, dtype=torch.float32).to(device)
    optimization_steps = args.optimization_steps
    generator = torch.Generator(device=device).manual_seed(seed)

    value_model = ValueModel(
        dimension=num_atoms * atom_dim,
    )
    value_model.to(device)

    for optimization_idx in tqdm(range(optimization_steps), desc="Optimization Steps", leave=True):

        batch_mu = einops.repeat(mu, 'T D -> T B D', B=batch_size)
        batch_sigma = einops.repeat(sigma, 'T D -> T B D', B=batch_size)
        batch_noise = batch_mu + batch_sigma**0.5 * torch.randn(batch_mu.size(), generator=generator, device=device)
        batch_noise_norm = batch_noise.norm(dim=-1)
        batch_noise_projected = batch_noise / batch_noise_norm[:,:,None] * batch_noise_norm[:,:,None]
        given_noise_list = einops.rearrange(batch_noise_projected, 'T B (M N) -> T (B M) N', B=batch_size, M=num_atoms, N=atom_dim)

        with torch.inference_mode():
            molecules, pred_z0_lig_traj = model.generate_ligands(
                args.pdbfile, batch_size, args.resi_list, args.ref_ligand, ref_ligand,
                num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
                relax_iter=(200 if args.relax else 0),
                diversify_from_timestep=args.diversify_from_timestep,
                given_noise_list=given_noise_list,
            )
        assert len(molecules) == batch_size

        success_indices = []
        success_molecules = []
        for i, mol in enumerate(molecules):
            if mol is not None:
                success_indices.append(i)
                success_molecules.append(mol)
        batch_noise = batch_noise[:,success_indices,:]
        
        # Evaluate and save molecules
        objective_values = torch.zeros((num_parameters, len(success_indices)), dtype=torch.float32).to(device)
        raw_metrics, objective_values_final  = objective_fn(success_molecules)
        objective_values_final = objective_values_final.sum(1)
        raw_metrics_inv = pd.DataFrame(raw_metrics).to_dict('list')

        # objective_values[:] = objective_values_final

        x = einops.rearrange(
            pred_z0_lig_traj[:,success_indices,:,:],
            "T B M N -> T B (M N)",
            M=num_atoms,
            N=atom_dim
        )

        value_model.add_model_data(x[-1],objective_values_final.to(device))
        for k, xk in enumerate(x[:-1]):
            estimated_values, variance = value_model.predict(xk)
            objective_values[k,:] = estimated_values
        objective_values[-1] = objective_values_final

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
        for metric in metrics:
            raw_values = torch.tensor(raw_metrics_inv[metric])
            log_dict[f"train/{metric}_mean"] = raw_values.mean()
            log_dict[f"train/{metric}_best"] = raw_values.max() if METRIC_MAXIMIZE[metric] else raw_values.min()
        wandb.log(log_dict)
