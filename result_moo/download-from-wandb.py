import wandb
import pickle
import os

api = wandb.Api()

wandb_path = "kimyong95/sbdd-multi-objective"
dir_path = os.path.dirname(os.path.realpath(__file__))

def search_run_from_name(path, name):
    runs = api.runs(
        path=path,
        filters={"displayName": name, "tags": { "$eq": "final" }},
    )
    if len(runs) == 0:
        print(f"Run [{name}] not found.")
        return None
    else:
        return runs[0]

seeds = [0,1,2]

algos = [
    "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:32-o=sa;qed;vina-s={seed}",
    "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:16-o=sa;qed;vina-s={seed}",
    "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:8-o=sa;qed;vina-s={seed}",
    "sample-and-select-c=worst-ag=neglogsumexp-l=du-b=64:4-o=sa;qed;vina-s={seed}",
    "egd-o=sa;qed;vina-s{seed}",
    "sbdd-ea-mean-o=sa;qed;vina-s{seed}",
    "sbdd-ea-spea2-o=sa;qed;vina-s{seed}",
    "sample-and-select-ea=500-c=worst-ag=neglogsumexp-l=du-b=64:8-o=sa;qed;vina-s={seed}",
]



# load cache
cache_path = f"{dir_path}/wandb_cache.pkl"
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        wandb_cache = pickle.load(f)
else:
    wandb_cache = {}

def update_cache(cache_key, wandb_run):
    if cache_key in wandb_cache:
        print(f"Updating cache [{cache_key}] ...")
    else:
        print(f"Adding cache [{cache_key}] ...")
    
    history = wandb_run.history(wandb_run.lastHistoryStep+1)
    wandb_cache[cache_key] = {
        "history": history,
        "run_id": wandb_run.id,
        "run_name": wandb_run.name,
        "state": wandb_run.state,
    }

def is_update_cache(cache_key, wandb_run):
    redownload = []
    is_update = (cache_key not in wandb_cache \
        or wandb_cache[cache_key]["run_id"] != wandb_run.id \
        or cache_key in redownload \
        or wandb_run.lastHistoryStep > len(wandb_cache[cache_key]["history"]) \
        or int(wandb_run.summary["_timestamp"]) != int(wandb_cache[cache_key]["history"]["_timestamp"].iloc[-1])
    )

    return is_update

is_save_cache = None
for algo in algos:
    for seed in seeds:
        algo_name = algo.format(seed=seed)
        _run = search_run_from_name(wandb_path, algo_name)
        if _run is None:
            print(f"Warning: Run [{algo_name}] not found. Skipping...")
            continue
        if is_update_cache(algo_name, _run):
            update_cache(algo_name, _run)
            is_save_cache = True

pass

if is_save_cache:
    with open(cache_path, 'wb') as f:
        pickle.dump(wandb_cache, f)