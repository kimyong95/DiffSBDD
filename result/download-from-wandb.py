import wandb
import pickle
import os

api = wandb.Api()

wandb_path = "kimyong95/guide-sbdd"

dir_path = os.path.dirname(os.path.realpath(__file__))

def run_name(algo,name):
    if algo == "ours":
        return f"bdtg-s0-{name}"
    else:
        return f"{algo}-s0-{name}"

def cache_key(algo,name):
    return f"{algo}-{name}"

def get_latest_run_id(path, name):
    runs = api.runs(
        path=path,
        filters={"displayName": name, "tags": { "$ne": "test" }},
    )
    if len(runs) == 0:
        print(f"Run [{name}] not found.")
        return None
    else:
        return runs[0].id

algos = ["ours", "evolutionary"]
names = ["sa", "qed", "vina", "gnina"]

# load cache
cache_path = f"{dir_path}/wandb_cache.pkl"
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        wandb_cache = pickle.load(f)
else:
    wandb_cache = {}

def update_cache(cache_key, wandb_run, run_id):
    if cache_key in wandb_cache:
        print(f"Updating cache [{cache_key}] ...")
    else:
        print(f"Adding cache [{cache_key}] ...")
    
    history = wandb_run.history(wandb_run.lastHistoryStep+1)
    wandb_cache[cache_key] = {
        "history": history,
        "run_id": run_id,
        "run_name": wandb_run.name,
        "state": wandb_run.state,
    }

def is_update_cache(cache_key, wandb_run, run_id):
    redownload = []
    is_update = (cache_key not in wandb_cache \
        or wandb_cache[cache_key]["run_id"] != run_id \
        or cache_key in redownload \
        or wandb_run.lastHistoryStep > len(wandb_cache[cache_key]["history"]) \
        or int(wandb_run.summary["_timestamp"]) != int(wandb_cache[cache_key]["history"]["_timestamp"].iloc[-1])
    )

    return is_update

is_save_cache = None
for algo in algos:
    for name in names:
        _cache_key = cache_key(algo,name)
        _run_id = get_latest_run_id(wandb_path, run_name(algo,name))
        _wandb_run = api.run(f"{wandb_path}/{_run_id}")
        if is_update_cache(_cache_key, _wandb_run, _run_id):
            update_cache(_cache_key, _wandb_run, _run_id)
            is_save_cache = True

if is_save_cache:
    with open(cache_path, 'wb') as f:
        pickle.dump(wandb_cache, f)