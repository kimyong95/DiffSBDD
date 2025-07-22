from sbdd_metrics.metrics import FullEvaluator
from typing import List
from collections import defaultdict
import torch
METRIC_EVALUATOR_MAP = { "qed": "medchem", "sa": "medchem", "reos": "reos", "vina": "gnina", "gnina": "gnina" }

# True means maximize
METRIC_MAXIMIZE = { "qed": True, "sa": False, "reos": True, "vina": False, "gnina": True }

METRIC_RANGE = {
    "qed": (0.0, 1.0),
    "sa": (1.0, 10.0),
    "vina": (-12.0, 0.0),
    "gnina": (0, 10.0),
}

class Objective:

    def __init__(self, metrics: List, pocket_pdbfile):
        exclude_evaluators = list(set([__e.ID for __e in FullEvaluator(gnina="gnina").evaluators]) - set([METRIC_EVALUATOR_MAP[met] for met in metrics]))
        self.evaluator = FullEvaluator(gnina="gnina", exclude_evaluators=exclude_evaluators)
        
        if "vina" in metrics and "gnina" not in metrics:
            for e in self.evaluator.evaluators:
                if e.ID == "gnina":
                    e.vina_only = True
        
        self.metrics = metrics
        self.pocket_pdbfile = pocket_pdbfile

        self.objectives_consumption = 0

    # Input: list of molecules
    # Output:
    # - metrics_breakdown: a dictionary with metric names as keys and raw values as lists
    # - normalized_objective_values: tensor of (N, K), lower is better, roughly in [0, 1] range
    def __call__(self, molecules):

        normalized_objective_values = torch.zeros((len(molecules), len(self.metrics)))
        raw_metrics = []
        
        # mol_results = []
        # for molecule in molecules:
        #     mol_results.append(self.evaluator.evaluate(molecule, protein=self.pocket_pdbfile))

        mol_results = self.evaluator.evaluate_batch(molecules, proteins=[self.pocket_pdbfile]*len(molecules))

        for i, mol_result in enumerate(mol_results):
            raw_metric = {}
            for j, metric in enumerate(self.metrics):
                raw_value, normalized_obj_value = self.get_objective_values(mol_result, metric)
                normalized_objective_values[i, j] = normalized_obj_value
                raw_metric[metric] = raw_value
            raw_metrics.append(raw_metric)
        
        self.objectives_consumption += len(molecules)

        return raw_metrics, normalized_objective_values
    
    # Input: raw metric value (could be None)
    # Output:
    # - raw_value: the raw metric value
    # - normalized_objective_value: scale the objective_value to roughly [0, 1] range, lower is better
    def get_objective_values(self, result, metric):
        if metric == "qed":
            if not result["medchem.valid"] or result["medchem.qed"] is None:
                raw_value = 0.0
            else:
                raw_value = result["medchem.qed"]
        elif metric == "sa":
            if not result["medchem.valid"] or result["medchem.sa"] is None:
                raw_value = 10.0
            else:
                raw_value = result["medchem.sa"]
        elif metric == "reos":
            reos_filters = ['reos.Glaxo', 'reos.Dundee', 'reos.BMS', 'reos.PAINS', 'reos.SureChEMBL', 'reos.MLSMR', 'reos.Inpharmatica', 'reos.LINT']
            raw_value = sum([float(result[obj]) for obj in reos_filters])
        elif metric == "vina":
            raw_value = result["gnina.vina_score"]
        elif metric == "gnina":
            raw_value = result["gnina.gnina_score"]
        else:
            raise ValueError(f"Metric {metric} not recognized.")
        
        if METRIC_MAXIMIZE[metric]:
            min_value, max_value = METRIC_RANGE[metric]
            normalized_objective_value = (min_value - raw_value) / (max_value - min_value)
        else:
            min_value, max_value = METRIC_RANGE[metric]
            normalized_objective_value = (raw_value - max_value) / (max_value - min_value)

        return raw_value, normalized_objective_value

