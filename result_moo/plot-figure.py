# plot 3x4 grid for metric
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import seaborn as sns
import os
import bisect

dir_path = os.path.dirname(os.path.realpath(__file__))

colors = sns.color_palette("husl", 9)

color_map = {
    "IMG (M=32)": colors[6],
    "IMG (M=16)": colors[6],
    "IMG (M=8)": colors[6],
    "IMG (M=4)": colors[6],
    "EGD+IMG": colors[7],

    "EGD":  colors[3],
    "DiffSBDD-EA (Mean)": colors[1],
    "DiffSBDD-EA (Spea2)": colors[2],
}

zorder_map = {
    "IMG (M=32)": 100,
    "IMG (M=16)": 100,
    "IMG (M=8)": 100,
    "IMG (M=4)": 100,
    "EGD+IMG": 100,

    "EGD":  3,
    "DiffSBDD-EA (Mean)": 2,
    "DiffSBDD-EA (Spea2)": 1,
    
}

seeds = [0, 1, 2]


cache_path = f"{dir_path}/wandb_cache.pkl"
with open(cache_path, 'rb') as f:
    wandb_cache = pickle.load(f)

fig, ax = plt.subplots(figsize=(10, 5))

# -------------------------- Our Method --------------------------
names = {
    "IMG (M=32)": "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:32-o=sa;qed;vina-s={seed}",
    "IMG (M=16)": "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:16-o=sa;qed;vina-s={seed}",
    "IMG (M=8)" : "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:8-o=sa;qed;vina-s={seed}",
    "IMG (M=4)" : "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:4-o=sa;qed;vina-s={seed}",
}

objectives_feedbacks = []

for i, (plot_name, run_name) in enumerate(names.items()):

    x_data = []
    y_data = []
    num_pareto = []
    runtime_data = []
    for seed in seeds:
        _run_name = run_name.format(seed=seed)
        _history = wandb_cache[_run_name]["history"]
        _step = _history["_step"]
        x_data.append(_history["objectives_feedbacks"][_step].values)
        _y = _history["intermediate/hypervolume"][_step].values
        
        # TODO: This is just for un-finished runs, remove this later
        _y[-1] = _history["final/hypervolume"].values[-1] if "final/hypervolume" in _history else _y[-1]
        y_data.append(_y)

        num_pareto.append(_history["final/number_of_pareto"].values[-1])
        runtime_data.append(_history["_runtime"].values[-1])

    x = x_data[0]
    y_mean = np.mean(np.stack(y_data, axis=0), axis=0)
    y_std = np.std(np.stack(y_data, axis=0), axis=0)
    num_pareto = np.mean(np.stack(num_pareto, axis=0), axis=0)
    avg_runtime = np.mean(runtime_data)
    
    # Convert runtime from seconds to hours and minutes
    hours = int(avg_runtime // 3600)
    minutes = int((avg_runtime % 3600) // 60)

    # Dotted line for intermediate during diffusion inference
    x_inter = x[:-1]
    y_mean_inter = y_mean[:-1]
    y_std_inter = y_std[:-1]
    label = "IMG" if i == 0 else None
    ax.plot(x_inter, y_mean_inter, linestyle='dotted', label=label, color=color_map[plot_name], linewidth=2, zorder=zorder_map[plot_name])
    # ax.fill_between(x_inter, y_mean_inter - y_std_inter, y_mean_inter + y_std_inter, color=color_map[plot_name], alpha=0.2, zorder=zorder_map[plot_name]-1)

    # A big dot for final hypervolume
    ax.plot(x[-1], y_mean[-1],'o', color=color_map[plot_name], zorder=zorder_map[plot_name]+1)

    objectives_feedbacks.append(x[-2])
    print(f"[{plot_name}] - objective feedbacks: {x[-2]}, mean: {y_mean[-1]:.4f}, std: {y_std[-1]:.4f}, num pareto: {num_pareto:.2f}, runtime: {hours}h {minutes}m")

    # Write text
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    x_offset = -0.01 * x_range  # 2% of x-range to the left
    y_offset = 0.02 * y_range   # 2% of y-range upward
    if plot_name == "IMG (M=32)":
        ax.text(x[-1] + x_offset, y_mean[-1] - y_offset, plot_name, ha='right', va='top', fontsize=10, color=color_map[plot_name])
    else:
        ax.text(x[-1] + x_offset, y_mean[-1] + y_offset, plot_name, ha='right', va='bottom', fontsize=10, color=color_map[plot_name])

# -------------------------- Our + EA --------------------------
names = {
    "EGD+IMG": "sample-and-select-ea=500-c=worst-ag=neglogsumexp-l=du-b=64:8-o=sa;qed;vina-s={seed}",
}
for plot_name, run_name in names.items():

    ea_step = 500
    x_data = []
    y_data = []
    runtime_data = []
    for seed in seeds:
        _run_name = run_name.format(seed=seed)
        _history = wandb_cache[_run_name]["history"]
        _step = _history["_step"]
        x_data.append(_history["objectives_feedbacks"][_step].values)
        _y = _history["population/hypervolume"][_step].values
        _y[ea_step:] = _history["intermediate/hypervolume"][_step].values[ea_step:]
        _y[-1] = _history["final/hypervolume"].values[-1]
        y_data.append(_y)
        runtime_data.append(_history["_runtime"].values[-1])

    x = x_data[0]
    y_mean = np.mean(np.stack(y_data, axis=0), axis=0)
    y_std = np.std(np.stack(y_data, axis=0), axis=0)
    avg_runtime = np.mean(runtime_data)
    
    # Convert runtime from seconds to hours and minutes
    hours = int(avg_runtime // 3600)
    minutes = int((avg_runtime % 3600) // 60)
    
    # Calculate final num_pareto for printing
    num_pareto_data = []
    for seed in seeds:
        _run_name = run_name.format(seed=seed)
        _history = wandb_cache[_run_name]["history"]
        num_pareto_data.append(_history["final/number_of_pareto"].values[-1])
    num_pareto = np.mean(num_pareto_data)
    
    print(f"[{plot_name}] - objective feedbacks: {x[-1]}, mean: {y_mean[-1]:.4f}, std: {y_std[-1]:.4f}, num pareto: {num_pareto:.2f}, runtime: {hours}h {minutes}m")


    # Solid line with fill_between until ea_step
    # Sample every interval steps for plotting, otherwise too dense and hard to read
    plot_interval = 10
    indices = np.arange(0, ea_step, plot_interval)
    # Always include the last point
    if indices[-1] != len(x) - 1:
        indices = np.append(indices, ea_step - 1)

    x_ea = x[:ea_step]
    y_mean_ea = y_mean[:ea_step]
    y_std_ea = y_std[:ea_step]
    ax.plot(x_ea[indices], y_mean_ea[indices], label=plot_name, color=color_map[plot_name], linewidth=1.0, zorder=zorder_map[plot_name])
    # ax.fill_between(x_ea[indices], y_mean_ea[indices] - y_std_ea[indices], y_mean_ea[indices] + y_std_ea[indices], color=color_map[plot_name], alpha=0.2, zorder=zorder_map[plot_name]-1)


    # Dotted line from ea_step to end (excluding final point)
    x_inter = x[ea_step-1:-1]
    y_mean_inter = y_mean[ea_step-1:-1]
    y_std_inter = y_std[ea_step-1:-1]
    ax.plot(x_inter, y_mean_inter, linestyle='dotted', color=color_map[plot_name], linewidth=2, zorder=zorder_map[plot_name])
    # ax.fill_between(x_inter, y_mean_inter - y_std_inter, y_mean_inter + y_std_inter, color=color_map[plot_name], alpha=0.2, zorder=zorder_map[plot_name]-1)


    # A big dot for final hypervolume
    ax.plot(x[-1], y_mean[-1],'o', color=color_map[plot_name], zorder=zorder_map[plot_name]+1)

    ax.text(x[-1] + x_offset, y_mean[-1] + y_offset, "EGD + IMG (M=8)", ha='right', va='bottom', fontsize=10, color=color_map[plot_name])

# -------------------------- Baseline --------------------------
names = {
    "EGD": "egd-o=sa;qed;vina-s{seed}",
    "DiffSBDD-EA (Mean)": "sbdd-ea-mean-o=sa;qed;vina-s{seed}",
    "DiffSBDD-EA (Spea2)": "sbdd-ea-spea2-o=sa;qed;vina-s{seed}",
}
for plot_name, run_name in names.items():

    x_data = []
    y_data = []
    num_pareto = []
    runtime_data = []
    for seed in seeds:
        _run_name = run_name.format(seed=seed)
        _history = wandb_cache[_run_name]["history"]
        _step = _history["_step"]
        x_data.append(_history["objectives_feedbacks"][_step].values)
        y_data.append(_history["population/hypervolume"][_step].values)
        num_pareto.append(_history["population/number_of_pareto"][_step].values)
        runtime_data.append(_history["_runtime"][_step].values)

    # TODO: This is just for un-finished runs, remove this later
    min_len = min(map(len, x_data))
    x_data = [arr[:min_len] for arr in x_data]
    y_data = [arr[:min_len] for arr in y_data]
    num_pareto = [arr[:min_len] for arr in num_pareto]
    runtime_data = [arr[:min_len] for arr in runtime_data]

    x = x_data[0]
    y_mean = np.mean(np.stack(y_data, axis=0), axis=0)
    y_std = np.std(np.stack(y_data, axis=0), axis=0)
    num_pareto = np.mean(np.stack(num_pareto, axis=0), axis=0)
    runtime_mean = np.mean(np.stack(runtime_data, axis=0), axis=0)

    for fb in objectives_feedbacks:
        i_ = bisect.bisect_left(x, fb)
        if i_ < len(x):
            num_pareto_val = _history["population/number_of_pareto"].values
            
            # Convert runtime from seconds to hours and minutes
            runtime_seconds = runtime_mean[i_]
            hours = int(runtime_seconds // 3600)
            minutes = int((runtime_seconds % 3600) // 60)
            
            print(f"{plot_name} - objective feedbacks: {x[i_]}, mean: {y_mean[i_]:.4f}, std: {y_std[i_]:.4f}, num pareto: {num_pareto_val[i_]:.2f}, runtime: {hours}h {minutes}m")

    # Sample every interval steps for plotting, otherwise too dense and hard to read
    plot_interval = 10
    indices = np.arange(0, len(x), plot_interval)
    # Always include the last point
    if indices[-1] != len(x) - 1:
        indices = np.append(indices, len(x) - 1)
    x = x[indices]
    y_mean = y_mean[indices]
    y_std = y_std[indices]

    ax.plot(x, y_mean, label=plot_name, color=color_map[plot_name], linewidth=1.0, zorder=zorder_map[plot_name])
    # ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color_map[plot_name], alpha=0.2, zorder=zorder_map[plot_name]-1)

ax.grid(linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_title(f"")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Objective Evaluations', ha='center', fontsize=14)
text_y = fig.text(-0.15 / fig.get_figwidth(), 0.5, 'Hypervolume', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig(f"{dir_path}/figure.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y), format='jpeg', pil_kwargs={"quality":93})


# -------------------------- Ablation --------------------------


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Ablation on c
c_values = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
hypervolumes_c = []
for c in c_values:
    run_name = f"sample-and-select-c={c}-ag=neglogsumexp-l=du-b=32:8-o=sa;qed;vina-s=0"
    if run_name in wandb_cache:
        history = wandb_cache[run_name]["history"]
        hypervolumes_c.append(history["final/hypervolume"].values[-1])


ax1.plot(c_values, hypervolumes_c, marker='o', color=colors[6])
ax1.set_xlabel("Coefficient Parameter (c)")
ax1.set_ylabel("Hypervolume")
ax1.set_title("Ablation on Coefficient Parameter")
ax1.grid(True)

# Ablation on n
n_values = [2, 4, 8, 16, 32]
hypervolumes_n = []
for n in n_values:
    run_name = f"sample-and-select-c=worst-ag=neglogsumexp-l=du-b={n}:32-o=sa;qed;vina-s=0"
    if run_name in wandb_cache:
        history = wandb_cache[run_name]["history"]
        hypervolumes_n.append(history["final/hypervolume"].values[-1])

ax2.plot(n_values, hypervolumes_n, marker='o', color=colors[6])
ax2.set_xlabel("Batch Size (N)")
ax2.set_ylabel("Hypervolume")
ax2.set_title("Ablation on Batch Size")
ax2.grid(True)

plt.tight_layout()
plt.savefig(f"{dir_path}/ablation_figure.jpeg", dpi=600, format='jpeg', pil_kwargs={"quality":93})
