
# plot 3x4 grid for metric
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import seaborn as sns
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

colors = sns.color_palette("hls", 4)

color_map = {
    "ours": colors[0],
    "evolutionary":  colors[1],
}

zorder_map = {
    "ours": 100,
    "evolutionary":  1,
}

cache_path = f"{dir_path}/wandb_cache.pkl"
with open(cache_path, 'rb') as f:
    wandb_cache = pickle.load(f)

names = ["sa", "qed", "vina", "gnina"]

score_dict = {}

def score_dict_key(algo,name):
    return f"{algo}-{name}"

def cache_key(algo,name):
    return f"{algo}-{name}"

##################### load scores ######################
algos = ["ours", "evolutionary"]

for algo in algos:
    for name in names:
        _cache_key = cache_key(algo,name)
        _score_key = score_dict_key(algo,name)
        _metric_key = f"train/{name}_mean"

        metric_y = wandb_cache[_cache_key]["history"][_metric_key]
        metric_x = wandb_cache[_cache_key]["history"][metric_y.notnull()]["step"].values.tolist()
        
        # unnormalize to raw score, refer to related_works/d3po/d3po_pytorch/rewards.py
        metric_y = metric_y[metric_y.notnull()].values.tolist()
        score_dict[_score_key] = dict(zip(metric_x, metric_y))
##################### load scores ######################

algo_label_map = {
    "ours": "BDTG (ours)",
    "evolutionary": "DiffSBDD (evolutionary)",
}

name_label_map = {
    "sa": "SA",
    "qed": "QED",
    "vina": "Vina",
    "gnina": "Gnina",
}

######################## main plot #########################
fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
figure_x_map = {
    "sa": np.arange(300),
    "qed": np.arange(300),
    "vina": np.arange(1000),
    "gnina": np.arange(1000),
}

for col in range(4):
    index =  col
    ax = axes[col]
    name = names[index]
    for algo_i, algo in enumerate(algos):
        
        _score_key = score_dict_key(algo,name)

        figure_x = figure_x_map[name]
        y = [score_dict[_score_key][k] for k in figure_x if k in score_dict[_score_key]]
        label = algo_label_map[algo] if index == 0 else None
        linewidth = 1.5
        x = figure_x[figure_x < len(y)] + 1
        ax.plot(x, y, label=label, color=color_map[algo], linewidth=linewidth, zorder=zorder_map[algo])
        
        ax.set_xlim(0, len(figure_x))
    
    ax.grid(linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_title(f"{name_label_map[name]}")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Objective Feedbacks', ha='center', fontsize=14)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Average Objective Values', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig(f"{dir_path}/figure.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y), format='jpeg', pil_kwargs={"quality":80})
######################## main plot #########################
