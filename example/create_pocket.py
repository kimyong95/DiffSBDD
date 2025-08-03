# This is the file to create the pocket.pdb
# This file is not intended to execute in this repository.
# Instead, run it in the Drugflow: https://github.com/LPDI-EPFL/DrugFlow
# Then copy over the generated pocket file to this repository.

from Bio.PDB import PDBParser
from rdkit import Chem
import torch
import warnings
import argparse
from src.data.molecule_builder import build_molecule
from Bio.PDB import PDBParser
from src.data.data_utils import process_raw_pair
from src.constants import atom_encoder, atom_decoder, aa_decoder, residue_decoder, aa_atom_index
from src.data.misc import protein_letters_1to3
from pathlib import Path

def pocket_to_rdkit(pocket, pocket_representation, atom_encoder=None,
                    atom_decoder=None, aa_decoder=None, residue_decoder=None,
                    aa_atom_index=None):

    rdpockets = []
    for i in torch.unique(pocket['mask']):

        node_coord = pocket['x'][pocket['mask'] == i]
        h = pocket['one_hot'][pocket['mask'] == i]
        atom_mask = pocket['atom_mask'][pocket['mask'] == i]

        pdb_infos = []

        if pocket_representation == 'side_chain_bead':
            coord = node_coord

            node_types = [residue_decoder[b] for b in h[:, -len(residue_decoder):].argmax(-1)]
            atom_types = ['C' if r == 'CA' else 'F' for r in node_types]

        elif pocket_representation == 'CA+':
            aa_types = [aa_decoder[b] for b in h.argmax(-1)]
            side_chain_vec = pocket['v'][pocket['mask'] == i]

            coord = []
            atom_types = []
            for resi, (xyz, aa, vec, am) in enumerate(zip(node_coord, aa_types, side_chain_vec, atom_mask)):

                # CA not treated differently with updated atom dictionary
                for atom_name, idx in aa_atom_index[aa].items():

                    if ~am[idx]:
                        warnings.warn(f"Missing atom {atom_name} in {aa}:{resi}")
                        continue

                    coord.append(xyz + vec[idx])
                    atom_types.append(atom_name[0])

                    info = Chem.AtomPDBResidueInfo()
                    # info.SetChainId('A')
                    info.SetResidueName(protein_letters_1to3[aa])
                    info.SetResidueNumber(resi + 1)
                    info.SetOccupancy(1.0)
                    info.SetTempFactor(0.0)
                    info.SetName(f' {atom_name:<3}')
                    pdb_infos.append(info)

            coord = torch.stack(coord, dim=0)

        else:
            raise NotImplementedError(f"{pocket_representation} residue representation not supported")

        atom_types = torch.tensor([atom_encoder[a] for a in atom_types])
        rdmol = build_molecule(coord, atom_types, atom_decoder=atom_decoder)

        if len(pdb_infos) == len(rdmol.GetAtoms()):
            for a, info in zip(rdmol.GetAtoms(), pdb_infos):
                a.SetPDBResidueInfo(info)

        rdpockets.append(rdmol)

    return rdpockets

def main():
    parser = argparse.ArgumentParser(description='Create pocket PDB file from protein and reference ligand')
    parser.add_argument('--ref_ligand', type=str, required=True,
                        help='Path to reference ligand SDF file')
    parser.add_argument('--target_protein', type=str, required=True,
                        help='Path to target protein PDB file')
    parser.add_argument('--out_pocket_path', type=str, required=True,
                        help='Output path for pocket PDB file')
    
    args = parser.parse_args()

    pdb_model = PDBParser(QUIET=True).get_structure('', args.target_protein)[0]
    rdmol = Chem.SDMolSupplier(args.ref_ligand)[0]

    ligand, pocket = process_raw_pair(
            pdb_model, rdmol,
            dist_cutoff=8.0,
            pocket_representation="CA+",
            compute_nerf_params=True,
            nma_input=None
        )

    pocket_rdkit = pocket_to_rdkit(
        pocket, "CA+",
        atom_encoder=atom_encoder,
        atom_decoder=atom_decoder,
        aa_decoder=aa_decoder,
        residue_decoder=residue_decoder,
        aa_atom_index=aa_atom_index
    )
    Chem.MolToPDBFile(pocket_rdkit[0], args.out_pocket_path)

if __name__ == "__main__":
    main()