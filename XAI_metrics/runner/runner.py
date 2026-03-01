# XAI_metrics/runner/runner.py
from XAI_metrics.base import METRIC_REGISTRY, MetricContext
from XAI_metrics.metrics import autodiscover_metrics
import XAI_metrics.metrics as metrics_pkg
import inspect

def run_all_metrics(context: MetricContext):
    autodiscover_metrics(metrics_pkg)

    out = {'results': {}}
    deps = context.extras or {}

    for name, MetricCls in METRIC_REGISTRY.items():
        sig = inspect.signature(MetricCls.__init__)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in deps.items() if k in allowed}

        m = MetricCls(context, **kwargs)
        out['results'][name] = m.run()

    return out