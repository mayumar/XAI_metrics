# XAI_metrics/runner/runner.py
from pathlib import Path
import yaml

from XAI_metrics.base import MetricContext, build_metrics_from_config
from XAI_metrics.metrics import autodiscover_metrics
import XAI_metrics.metrics as metrics_pkg
import XAI_metrics.base.registry as registry

from typing import Iterable, Any, Mapping

def _normalize_selection(
    selected_metrics: Iterable[str] | None,
    available_metrics: Iterable[str],
) -> list[str]:
    available_list = list(available_metrics)

    if selected_metrics is None:
        return available_list

    selected = list(selected_metrics)
    available_set = set(available_list)

    missing = [name for name in selected if name not in available_set]
    if missing:
        raise ValueError(
            f"Metrics not registered: {missing}. "
            f"Available: {sorted(available_list)}"
        )

    return selected


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if isinstance(config, Mapping):
        return dict(config)

    config_path = Path(config)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_all_metrics(
    context: MetricContext,
    selected_metrics: Iterable[str] | None = None,
    only_metric_modules: Iterable[str] | None = None,
    exclude_metric_modules: Iterable[str] | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    autodiscover_metrics(
        metrics_pkg,
        only_modules=only_metric_modules,
        exclude_modules=exclude_metric_modules,
    )

    resolved_config = _load_config(config)
    metric_names = _normalize_selection(selected_metrics, registry.METRIC_REGISTRY.keys())
    allowed = set(metric_names)

    filtered_cfg = {
        "metrics": [
            m for m in resolved_config.get("metrics", [])
            if m.get("name") in allowed
        ]
    }

    deps = context.extras or {}
    metrics = build_metrics_from_config(filtered_cfg, context, dependencies=deps)

    out = {"results": {}}
    for metric in metrics:
        name = getattr(metric, "NAME", metric.__class__.__name__)
        out["results"][name] = metric.run()

    return out