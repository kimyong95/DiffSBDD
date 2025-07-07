
from sbdd_metrics.metrics import FullEvaluator
from typing import List
from collections import defaultdict
import torch
METRIC_EVALUATOR_MAP = { "qed": "medchem", "sa": "medchem", "reos": "reos", "vina": "gnina", "gnina": "gnina" }

# True means maximize
METRIC_MAXIMIZE = { "qed": True, "sa": False, "reos": True, "vina": False, "gnina": True }

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

    def __call__(self, molecules):

        objective_values = torch.zeros((len(molecules), len(self.metrics)))
        raw_values = torch.zeros((len(molecules), len(self.metrics)))
        mol_results = self.evaluator.evaluate_batch(molecules, proteins=[self.pocket_pdbfile]*len(molecules))
        for i, mol_result in enumerate(mol_results):
            for j, metric in enumerate(self.metrics):
                obj_value, raw_value = self.process_result_metric(mol_result, metric)
                objective_values[i, j] = obj_value
                raw_values[i, j] = raw_value

        objective_values = objective_values.sum(dim=1)
        metrics_breakdown = {
            metric_name: raw_values[:, i].tolist()
            for i, metric_name in enumerate(self.metrics)
        }

        return objective_values, metrics_breakdown
    
    # Ensure minimize the returned [objective_value]
    def process_result_metric(self, result, metric):
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
        
        sign = {True: -1, False: +1}
        objective_value = sign[METRIC_MAXIMIZE[metric]] * raw_value

        return objective_value, raw_value

            
