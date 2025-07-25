import argparse
from pathlib import Path

from utils import seed_everything
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
import pandas as pd
import random
import wandb
from objective import Objective, METRIC_MAXIMIZE
from collections import defaultdict
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


def diversify_ligands(model, pocket, mols, timesteps,
                    sanitize=False,
                    largest_frag=False,
                    relax_iter=0):
    """
    Diversify ligands for a specified pocket.
    
    Parameters:
        model: The model instance used for diversification.
        pocket: The pocket information including coordinates and types.
        mols: List of RDKit molecule objects to be diversified.
        timesteps: Number of denoising steps to apply during diversification.
        sanitize: If True, performs molecule sanitization post-generation (default: False).
        largest_frag: If True, only the largest fragment of the generated molecule is returned (default: False).
        relax_iter: Number of iterations for force field relaxation of the generated molecules (default: 0).
    
    Returns:
        A list of diversified RDKit molecule objects.
    """

    ligand = utils.prepare_ligands_from_mols(mols, model.lig_type_encoder, device=model.device)

    pocket_mask = pocket['mask']
    lig_mask = ligand['mask']

    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    out_lig, out_pocket, _, _, _ = model.ddpm.diversify(ligand, pocket, noising_steps=timesteps)

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

    return molecules


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--pocket_pdbfile', type=str, default='example/5ndu_pocket.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_C_8V2.sdf')
    parser.add_argument('--objective', type=str, default='qed')
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=32)
    parser.add_argument('--evolution_steps', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--outfile', type=Path, default='output.sdf')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    run = wandb.init(
        project="guide-sbdd",
        name=f"evolutionary-s{seed}-{args.objective}",
        config=args,
    )

    pdb_id = Path(args.pdbfile).stem

    device = "cuda"

    population_size = args.population_size
    evolution_steps = args.evolution_steps
    top_k = args.top_k

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # Prepare ligand + pocket
    # Load PDB
    pdb_model = PDBParser(QUIET=True).get_structure('', args.pdbfile)[0]
    # Define pocket based on reference ligand
    residues = utils.get_pocket_from_ligand(pdb_model, args.ref_ligand)
    pocket = model.prepare_pocket(residues, repeats=population_size)

    metrics = args.objective.split(";")
    objective_fn = Objective(metrics, args.pocket_pdbfile)

    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]

    # Store molecules in history dataframe 
    buffer = pd.DataFrame(columns=['generation', 'score', 'fate' 'mol', 'smiles'])

    # Population initialization
    _, ref_objective_value = objective_fn([ref_mol])
    ref_objective_value = ref_objective_value.sum(1)
    
    buffer_df = pd.DataFrame([{'generation': 0,
        'score': -ref_objective_value.item(),  # Put negative because we want to maximize buffer[score]
        'fate': 'initial', 'mol': ref_mol,
        'smiles': Chem.MolToSmiles(ref_mol)}])
    buffer = pd.concat([buffer, buffer_df], ignore_index=True)

    print(f"Reference molecule objective value (lower better): {ref_objective_value}")

    for generation_idx in range(evolution_steps):

        if generation_idx == 0:
            molecules = buffer['mol'].tolist() * population_size
        else:
            # Select top k molecules from previous generation
            previous_gen = buffer[buffer['generation'] == generation_idx]
            top_k_molecules = previous_gen.nlargest(top_k, 'score')['mol'].tolist()
            molecules = top_k_molecules * (population_size // top_k)

            # Update the fate of selected top k molecules in the buffer
            buffer.loc[buffer['generation'] == generation_idx, 'fate'] = 'survived'

            # Ensure the right number of molecules
            if len(molecules) < population_size:
                molecules += [random.choice(molecules) for _ in range(population_size - len(molecules))]


        # Diversify molecules
        assert len(molecules) == population_size, f"Wrong number of molecules: {len(molecules)} when it should be {population_size}"

        with torch.inference_mode():
            molecules = diversify_ligands(model,
                                        pocket,
                                        molecules,
                                    timesteps=args.timesteps,
                                    sanitize=True,
                                    relax_iter=(200 if args.relax else 0))
        
        
        # Evaluate and save molecules
        raw_metrics, objective_values = objective_fn(molecules)
        objective_values = objective_values.sum(1)
        raw_metrics_inv = pd.DataFrame(raw_metrics).to_dict('list')

        for mol, obj_value in zip(molecules, objective_values):
            buffer_df = pd.DataFrame([
                {'generation': generation_idx + 1,
                'score': -obj_value.item(), # Put negative because we want to maximize buffer[score]
                'fate': 'purged', 'mol': mol,
                'smiles': Chem.MolToSmiles(mol)}])
            buffer = pd.concat([buffer, buffer_df], ignore_index=True)

        # wandb log the score lower better
        log_dict = {
            "train/score_mean": objective_values.mean(),
            "train/score_best": objective_values.min(),
            "step": generation_idx,
            "train/feasible_mol_rate": len(molecules) / population_size,
        }
        for metric in metrics:
            raw_values = torch.tensor(raw_metrics_inv[metric])
            log_dict[f"train/{metric}_mean"] = raw_values.mean()
            log_dict[f"train/{metric}_best"] = raw_values.max() if METRIC_MAXIMIZE[metric] else raw_values.min()
        wandb.log(log_dict)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
    # Save buffer
    buffer.drop(columns=['mol'])
    buffer.to_csv(args.outfile.with_suffix('.csv'))
