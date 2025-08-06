import wandb
import pandas as pd
import re
import numpy as np
import os
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import pymol
import sys
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
import warnings
import rdkit
from rdkit import Chem
from utils_moo import calculate_hypervolume
import pymol
import torch
import sys
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def get_pareto_front_indices(points):
    """
    Finds the indices of Pareto optimal points in a 2D array using pymoo.
    Assumes higher values are better by negating them for the minimization algorithm.
    """
    # NonDominatedSorting assumes minimization. Since higher values are better for us,
    # we find the Pareto front of the negated values.
    nds = NonDominatedSorting()
    fronts = nds.do(points)
    pareto_indices = fronts[0]
    return pareto_indices

api = wandb.Api()

wandb_path = "kimyong95/sbdd-multi-objective"
dir_path = os.path.dirname(os.path.realpath(__file__))


table_name = "molecules_table"

algorithms = ["img", "egd", "sbdd-ea-mean", "sbdd-ea-spea2"]
objectives = ["sa", "qed", "vina"]

all_objective_values = {}

for algorithm in algorithms:
    df = pd.read_csv(f"visualize/data/{algorithm}.csv")
    last_of = df.iloc[-1].objectives_feedbacks
    print(f"Processing {algorithm} with last objectives_feedbacks: {last_of}")
    df = df[df.objectives_feedbacks == last_of]
    objective_values = df[objectives].to_numpy()
    all_objective_values[algorithm] = objective_values

def visualize_pareto_front(pareto_front_values, objectives, labels, filepath="pareto_front.jpeg"):
    """
    Visualizes a 3D Pareto front with 2D projections and shaded regions.

    This version distinguishes between points on a 2D Pareto front and those that are not.
    - Points on a 2D Pareto front have a thick projection line to that specific plane.
    - Points not on any 2D Pareto front have thin, semi-transparent projection lines
      to all three XYZ planes.

    Args:
        pareto_front_values (np.ndarray): A NumPy array of shape (n_points, 3)
            containing the multi-objective values. Assumes lower values are better
            and all values are negative.
        objectives (list): A list of three strings for the axis labels.
        labels (dict): A dictionary where keys are string labels and values are lists of
            indices corresponding to points in pareto_front_values.
        filepath (str): The path to save the output image file.
    """
    # --- 1. Data Preparation ---
    points_to_plot = -pareto_front_values
    num_points = len(points_to_plot)

    # --- 2. Setup 3D Plot ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- 3. Plot the 3D Pareto Points ---
    unique_labels = list(labels.keys())
    markers = ['o', 's', '^', 'P']
    label_to_marker = dict(zip(unique_labels, markers))

    for label, indices in labels.items():
        if len(indices) > 0:
            ax.scatter(
                points_to_plot[indices, 0],
                points_to_plot[indices, 1],
                points_to_plot[indices, 2],
                c='black',
                marker=label_to_marker.get(label, 'o'),
                s=50,
                label=label,
                depthshade=False,
                alpha=0.5
            )

    # --- 4. Identify 2D Pareto Sets and Non-Pareto Points ---
    # Find the indices of points that are Pareto-optimal on each 2D plane.
    pareto_indices_xy = set(get_pareto_front_indices(pareto_front_values[:, [0, 1]]))
    pareto_indices_xz = set(get_pareto_front_indices(pareto_front_values[:, [0, 2]]))
    pareto_indices_yz = set(get_pareto_front_indices(pareto_front_values[:, [1, 2]]))

    # Create a single set of all points that are on at least one 2D Pareto front.
    all_2d_pareto_indices = pareto_indices_xy | pareto_indices_xz | pareto_indices_yz

    # Identify points that are NOT on any 2D Pareto front.
    all_indices = set(range(num_points))
    non_pareto_indices = all_indices - all_2d_pareto_indices

    # --- 5. Draw Projections and Shaded Regions ---
    axes_to_drop = [2, 1, 0] # Corresponds to Z (XY), Y (XZ), X (YZ) planes
    pareto_sets_by_plane = [pareto_indices_xy, pareto_indices_xz, pareto_indices_yz]
    base_colors = sns.color_palette("Set2", 3)
    dark_colors = sns.color_palette("Dark2", 3)

    # a. Draw projections for points on a 2D Pareto front and shade the regions.
    for i, axis_to_drop in enumerate(axes_to_drop):
        pareto_indices_2d = pareto_sets_by_plane[i]
        
        # Draw thick projection lines for points on this specific 2D front.
        for j in pareto_indices_2d:
            point_3d = points_to_plot[j]
            projected_point_3d = point_3d.copy()
            projected_point_3d[axis_to_drop] = 0

            line_coords = np.array([point_3d, projected_point_3d])
            ax.plot(line_coords[:, 0], line_coords[:, 1], line_coords[:, 2], color=base_colors[i], linewidth=1.5, alpha=1.0)

        # Create and draw the shaded dominated region for this plane.
        if axis_to_drop == 2: proj_coords_plot = points_to_plot[:, [0, 1]]
        elif axis_to_drop == 1: proj_coords_plot = points_to_plot[:, [0, 2]]
        else: proj_coords_plot = points_to_plot[:, [1, 2]]

        pareto_points_2d = proj_coords_plot[list(pareto_indices_2d)]
        if len(pareto_points_2d) > 0:
            sorted_indices = np.argsort(pareto_points_2d[:, 0])
            sorted_pareto_2d = pareto_points_2d[sorted_indices]
            
            first_point = sorted_pareto_2d[0]
            last_point = sorted_pareto_2d[-1]
            
            poly_points = np.vstack([
                [0, 0], [last_point[0], 0], sorted_pareto_2d[::-1],
                [0, first_point[1]], [0, 0]
            ])
            
            verts = [np.insert(poly_points, axis_to_drop, 0, axis=1)]
            poly = Poly3DCollection(verts, facecolors=dark_colors[i], alpha=0.1)
            ax.add_collection3d(poly)

    # b. Draw thin projection lines for points NOT on any 2D Pareto front.
    for j in non_pareto_indices:
        point_3d = points_to_plot[j]
        # Project this point to all three planes.
        for i, axis_to_drop in enumerate(axes_to_drop):
            projected_point_3d = point_3d.copy()
            projected_point_3d[axis_to_drop] = 0
            
            line_coords = np.array([point_3d, projected_point_3d])
            ax.plot(line_coords[:, 0], line_coords[:, 1], line_coords[:, 2], color=base_colors[i], linewidth=0.5, alpha=0.5)


    # --- 6. Formatting and Camera Angle ---
    ax.set_xlabel(f"{objectives[0]} (higher is better)", fontsize=12)
    ax.set_ylabel(f"{objectives[1]} (higher is better)", fontsize=12)
    ax.set_zlabel(f"{objectives[2]} (higher is better)", fontsize=12)

    ax.view_init(elev=25, azim=55)
    
    ax.xaxis._axinfo['grid']['linewidth'] = 0.2
    ax.yaxis._axinfo['grid']['linewidth'] = 0.2
    ax.zaxis._axinfo['grid']['linewidth'] = 0.2

    ax.set_xlim(0, np.max(points_to_plot[:, 0]) * 1.1)
    ax.set_ylim(0, np.max(points_to_plot[:, 1]) * 1.1)
    ax.set_zlim(0, np.max(points_to_plot[:, 2]) * 1.1)

    # --- 7. Legend ---
    plane_legend_elements = [
        Patch(facecolor=base_colors[0], alpha=1.0, label=f'Pareto on {objectives[0]}-{objectives[1]} Plane'),
        Patch(facecolor=base_colors[1], alpha=1.0, label=f'Pareto on {objectives[0]}-{objectives[2]} Plane'),
        Patch(facecolor=base_colors[2], alpha=1.0, label=f'Pareto on {objectives[1]}-{objectives[2]} Plane'),
    ]

    if len(labels) > 1:
        # Create legend for algorithms
        algo_legend_elements = []
        for label, marker in label_to_marker.items():
            algo_legend_elements.append(Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=label))
        
        # Position the algorithm legend
        algo_legend = fig.legend(handles=algo_legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(algo_legend_elements), fontsize=10)
        ax.add_artist(algo_legend)

        # Position the plane legend below the algorithm legend
        fig.legend(handles=plane_legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)
    else:
        # Only show the plane legend if there is one algorithm
        fig.legend(handles=plane_legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Pareto front visualization saved to {filepath}")
    plt.close()

img_pareto_indicces = get_pareto_front_indices(all_objective_values["img"])
img_pareto = all_objective_values["img"][img_pareto_indicces]
img_labels = { "IMG": list(range(len(img_pareto))) }

hv = calculate_hypervolume(torch.from_numpy(img_pareto))
print(f"Hypervolume of IMG Pareto front: {hv.item()}")
print(f"Number of points for IMG in Pareto front is {len(img_pareto_indicces)}")
visualize_pareto_front(
    img_pareto,
    [obj.upper() for obj in objectives],
    labels=img_labels,
    filepath=os.path.join(dir_path, "pareto_front_img.jpeg")
)

combines_objective_values = np.concatenate([all_objective_values[algo] for algo in algorithms], axis=0)
pareto_indices = get_pareto_front_indices(combines_objective_values)
combines_perato = combines_objective_values[pareto_indices]

labels = defaultdict(list)
offsets = np.cumsum([0] + [len(all_objective_values[algo]) for algo in algorithms])

algorithm_labels = {
    "img": "IMG",
    "egd": "EGD",
    "sbdd-ea-mean": "DiffSBDD-EA (Mean)",
    "sbdd-ea-spea2": "DiffSBDD-EA (Spea2)",
}

for i, original_idx in enumerate(pareto_indices):
    for algo_idx, algo in enumerate(algorithms):
        if offsets[algo_idx] <= original_idx < offsets[algo_idx+1]:
            algo_label = algorithm_labels[algo]
            labels[algo_label].append(i)
            break
hv = calculate_hypervolume(torch.from_numpy(combines_perato))
print(f"Hypervolume of combined Pareto front: {hv.item()}")
for algo, indices in labels.items():
    print(f"Number of points for {algo} in combine Pareto front is {len(indices)}")
visualize_pareto_front(
    combines_perato,
    [obj.upper() for obj in objectives],
    labels=labels,
    filepath=os.path.join(dir_path, "pareto_front_relative.jpeg")
)

