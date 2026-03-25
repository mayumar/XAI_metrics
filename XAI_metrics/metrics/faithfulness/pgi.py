# XAI_metrics/metrics/faithfulness/pgu.py
from openxai import Evaluator
from openxai.explainers.perturbation_methods import NormalPerturbation
import torch
import numpy as np
from XAI_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Mapping, Any

@register_metric
class PGI(BaseMetric):
    NAME = "PGI"

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

    def run(self):
        ctx = self.context
        p = self.params

        k = float(p.get("k", 0.25))
        auc = bool(p.get("AUC", True))
        std = float(p.get("std", 0.1))
        n_samples = int(p.get("n_samples", 100))
        seed = int(p.get("seed", -1))
        n_jobs = int(p.get("n_jobs", -1))

        ctx.model.eval()

        param_dict = {
            "k": k,
            "AUC": auc,
            "n_samples": n_samples,
            "seed": seed,
            "n_jobs": n_jobs,
            "inputs": torch.tensor(
                ctx.X_test.loc[ctx.observations].values,
                dtype=torch.float32
            ),
            "explanations": torch.tensor(ctx.attributions, dtype=torch.float32),
            "feature_metadata": ctx.X_test.columns,
            "perturb_method": NormalPerturbation("tabular", mean=0.0, std_dev=std, flip_percentage=np.sqrt(2/np.pi)*std)
        }

        metric_evaluator = Evaluator(ctx.model, self.NAME)
        score, mean_score = metric_evaluator.evaluate(**param_dict)

        return score