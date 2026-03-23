# tests/test_runner.py
from pathlib import Path
import pytest

import XAI_metrics.runner.runner as runner_module
import XAI_metrics.base.registry as registry_module
from XAI_metrics.runner import run_all_metrics

def test_normalize_seleccion_returns_all_when_none():
    assert runner_module._normalize_selection(None, ["A", "B"]) == ["A", "B"]

def test_normalize_seleccion_returns_selected_when_valid():
    assert runner_module._normalize_selection(["B"], ["A", "B"]) == ["B"]

def test_normalize_selection_raises_for_missing_metrics():
    with pytest.raises(ValueError, match="Metrics not registered"):
        runner_module._normalize_selection(["Z"], ["A", "B"])

def test_load_config_from_mapping():
    config = {"metrics": [{"name": "A"}]}
    loaded = runner_module._load_config(config)
    assert loaded == config

def test_load_config_form_yaml_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("metrics:\n  - name: Example\n", encoding="utf-8")

    loaded = runner_module._load_config(config_path)

    assert loaded == {"metrics": [{"name": "Example"}]}

def test_load_config_uses_default_path_when_none(monkeypatch):
    default_path = Path(runner_module.__file__).resolve().parents[1] / "config.yaml"
    original = default_path.read_text(encoding="utf-8")

    try:
        default_path.write_text("metrics:\n  - name: DefaultMetric\n", encoding="utf-8")
        loaded = runner_module._load_config(None)
        assert loaded == {"metrics": [{"name": "DefaultMetric"}]}
    finally:
        default_path.write_text(original, encoding="utf-8")

def test_run_all_metrics_runs_selected_metrics(monkeypatch, metric_context):
    class FakeMetric:
        NAME = "FakeMetric"

        def __init__(self, context, params=None) -> None:
            self.context = context
            self.params = params or {}

        def run(self):
            return {"value": 123}
        
    monkeypatch.setattr(runner_module, "autodiscover_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry_module, "METRIC_REGISTRY", {"FakeMetric": FakeMetric})

    config = {
        "metrics": [
            {"name": "FakeMetric", "params": {"flag": True}},
            {"name": "IgnoredMetric", "params": {}},
        ]
    }

    result = run_all_metrics(
        context=metric_context,
        selected_metrics=["FakeMetric"],
        config=config
    )

    assert result == {"results": {"FakeMetric": {"value": 123}}}

def test_run_all_metrics_passes_extras_as_dependencies(monkeypatch, metric_context, explain_func):
    class DependencyMetric:
        NAME = "DependencyMetric"

        def __init__(self, context, params=None, explain_func=None):
            self.explain_func = explain_func

        def run(self):
            return self.explain_func is not None
        
    metric_context.extras["explain_func"] = explain_func

    monkeypatch.setattr(runner_module, "autodiscover_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry_module, "METRIC_REGISTRY", {"DependencyMetric": DependencyMetric})

    config = {"metrics": [{"name": "DependencyMetric"}]}

    result = run_all_metrics(
        context=metric_context,
        selected_metrics=["DependencyMetric"],
        config=config,
    )

    assert result == {"results": {"DependencyMetric": True}}