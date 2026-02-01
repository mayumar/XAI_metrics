from XAI_metrics.base import METRIC_REGISTRY, MetricContext
from XAI_metrics.metrics import autodiscover_metrics
import XAI_metrics.metrics as metrics_pkg

def run_all_metrics(context: MetricContext):
    autodiscover_metrics(metrics_pkg)

    out = {'results': {}}

    for name, MetricCls in METRIC_REGISTRY.items():
        m = MetricCls(context)
        out['results'][name] = m.run()

    return out