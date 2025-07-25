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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--pocket_pdbfile', type=str, default='example/5ndu_pocket.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_C_8V2.sdf')
    parser.add_argument('--objective', type=str, default='qed')
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=32)
    parser.add_argument('--evolution_steps', type=int, default=300)
    parser.add_argument('--outfile', type=Path, default='output.sdf')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='egd', choices=['egd', 'sbdd-ea'])
    parser.add_argument('--sbdd_top_k', type=int, default=4)


    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    name_str = args.mode

    run = wandb.init(
        project=f"sbdd-multi-objective",
        name=f"{name_str}-s{seed}-{args.objective}",
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
            if args.mode == 'egd':
                # Crossover will double the batch size
                molecules_to_diversify = molecules_to_diversify + molecules_to_diversify
                ligands_to_diversify = utils.prepare_ligands_from_mols(molecules_to_diversify, model.lig_type_encoder, device=model.device)
                diversified_molecules, _ = model.generate_ligands(
                    pdb_file=args.pdbfile,
                    ref_ligand_path=args.ref_ligand,
                    ref_ligand=ligands_to_diversify,
                    n_samples=len(molecules_to_diversify),
                    diversify_from_timestep=args.timesteps,
                    sanitize=True,
                    relax_iter=(200 if args.relax else 0),
                    crossover=True
                )
            elif args.mode == 'sbdd-ea':
                ligands_to_diversify = utils.prepare_ligands_from_mols(molecules_to_diversify, model.lig_type_encoder, device=model.device)
                diversified_molecules, _ = model.generate_ligands(
                    pdb_file=args.pdbfile,
                    ref_ligand_path=args.ref_ligand,
                    ref_ligand=ligands_to_diversify,
                    n_samples=len(molecules_to_diversify),
                    diversify_from_timestep=args.timesteps,
                    sanitize=True,
                    relax_iter=(200 if args.relax else 0),
                    crossover=False
                )
        
        successful_molecules = [mol for mol in diversified_molecules if mol is not None]
        
        # 2. Evaluate, select, and update buffer
        raw_metrics, objective_values = objective_fn(successful_molecules)

        candicates = [
            EvaluatedMolecule(mol, obj_values, raw_metric)
            for mol, obj_values, raw_metric in zip(successful_molecules, objective_values, raw_metrics)
        ]

        log_molecules_objective_values(candicates, objectives_feedbacks=objective_fn.objectives_consumption, stage="candidates", commit=False)

        if args.mode == 'egd':
            # 3. Union with family, and select best molecules based on the objective values
            buffer.extend(candicates)
            buffer = sorted(buffer, reverse=False)[:population_size]
        elif args.mode == 'sbdd-ea':
            # 3. Drop the parents
            buffer = candicates
            buffer = sorted(buffer, reverse=False)[:sbdd_top_k]
            while len(buffer) < population_size:
                buffer.extend(buffer)
            buffer = buffer[:population_size]

        log_molecules_objective_values(candicates, objectives_feedbacks=objective_fn.objectives_consumption, stage="population", commit=True)