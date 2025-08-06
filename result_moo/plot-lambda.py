import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from utils_moo import generate_weight_vectors, generate_iid_weight_vectors
import seaborn as sns
dir_path = os.path.dirname(os.path.realpath(__file__))

colors = sns.color_palette("husl", 9)

def plot_vectors(path, N, n=3, vector_type='structured'):
    """
    Generates and plots weight vectors based on the specified type.
    """
    print(f"--- Generating {vector_type} plot for N={N} ---")
    
    if vector_type == 'qmc':
        W = generate_weight_vectors(N, n)
        color = colors[6]
    elif vector_type == 'iid':
        W = generate_iid_weight_vectors(N, n)
        color = colors[1]
    else:
        raise ValueError("Invalid vector_type specified. Choose 'structured' or 'iid'.")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(W[0, :], W[1, :], W[2, :], alpha=1.0, color=color, s=30, depthshade=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=15, azim=45)
    
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved {vector_type} plot to {path}")
    plt.close(fig)

if __name__ == '__main__':
    # The dimension of the objective space is fixed at 3
    n_dimensions = 3
    
    # The values of N to iterate through
    N_values = [8, 16, 32, 64, 100, 200, 500, 1000]

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(dir_path, 'lambda')
    os.makedirs(output_dir, exist_ok=True)

    # Generate a plot for each value of N
    for N in N_values:
        structured_path = os.path.join(output_dir, f"qmc_N={N}.jpeg")
        iid_path = os.path.join(output_dir, f"iid_N={N}.jpeg")
        
        plot_vectors(structured_path, N, n_dimensions, vector_type='qmc')
        plot_vectors(iid_path, N, n_dimensions, vector_type='iid')
    
    print("\n--- All plots generated. ---")
