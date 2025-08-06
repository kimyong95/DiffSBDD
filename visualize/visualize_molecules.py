import pymol
import os
from pymol import cmd
import pandas as pd
import warnings
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
from rdkit import Chem

def visualize_complex(cmd, protein_file, molecule_pdb_string, output_path, styling_script="visualize/visualize.pml"):
    """
    Visualizes a protein-ligand complex and saves it as an image using an existing PyMOL instance.

    Args:
        cmd: The PyMOL command object.
        protein_file (str): Path to the protein PDB file.
        molecule_pdb_string (str): The molecule data in PDB format as a string.
        output_path (str): Path to save the output PNG image.
        smiles (str): SMILES string of the molecule.
        formula (str): Molecular formula of the molecule.
        objectives (dict): Dictionary of objective values.
        styling_script (str, optional): Path to the PyMOL styling script.
    """
    # --- Main Logic ---

    # Check if input files exist
    if not os.path.exists(protein_file):
        print(f"Error: Protein file '{protein_file}' not found!")
        return

    if not os.path.exists(styling_script):
        print(f"Error: Styling script '{styling_script}' not found!")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        # 0. Clear the previous state
        cmd.delete("all")

        # 1. Programmatically load the molecules
        print(f"Loading protein: {protein_file}")
        cmd.load(protein_file, "pocket")

        print(f"Loading molecule from string")
        cmd.read_pdbstr(molecule_pdb_string, "molecule")

        # 2. Run the .pml script to apply styling and set the view
        print(f"Running styling script: {styling_script}")
        cmd.run(styling_script)

        # 3. Save the final image to a file
        print(f"Saving image to: {output_path}")
        cmd.png(output_path, width=1200, height=900, dpi=300, ray=1)

        print(f"Visualization for {output_path} finished successfully!")

    except Exception as e:
        print(f"An error occurred during visualization for {output_path}: {e}")
        return

def process_molecule(molecule_pdb_string):
    """
    Checks if the molecule represented by the PDB string is connected.
    This is a placeholder function; actual implementation may vary.
    """
    # For simplicity, we assume the molecule is connected if it has more than one atom
    mol = Chem.MolFromPDBBlock(molecule_pdb_string)
    if not mol:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) != 1:
        return None

    mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))
    Chem.SanitizeMol(mol)

    if not UFFHasAllMoleculeParams(mol):
        warnings.warn('UFF parameters not available for all atoms. '
                        'Returning None.')
        return None

    try:
        uff_relax(mol, 200)

        Chem.SanitizeMol(mol)
    except (RuntimeError, ValueError) as e:
        return None

    molecule_pdb_string = Chem.MolToPDBBlock(mol)

    return molecule_pdb_string

def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


# --- PyMOL Visualization Loop ---
# Use pymol.launching to start PyMOL in a quiet, headless mode
pymol.pymol_argv = ['pymol', '-ckq']
pymol.finish_launching()
cmd = pymol.cmd

df = pd.read_csv(f"visualize/data/img.csv")
molecules = df["molecule"].to_list()

try:
    for i, molecule in enumerate(molecules):

        processed_molecule = process_molecule(molecule)
        if processed_molecule is None:
            print(f"Skipping molecule {i} due to processing failure.")
            continue

        output_image = f"visualize/molecules/img_processed_{i}.png"
        visualize_complex(
            cmd=cmd,
            protein_file="example/5ndu_pocket.pdb",
            molecule_pdb_string=processed_molecule,
            output_path=output_image,
            styling_script="visualize/visualize.pml"
        )

finally:
    # Properly shut down the PyMOL instance
    cmd.quit()