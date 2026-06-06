# tests/test_runner.py
from typing import Any, Mapping

import pytest

import xai_metrics.runner.runner as runner_module
from xai_metrics.base import METRIC_REGISTRY, BaseMetric, MetricSkipped
from xai_metrics.runner import run_evaluation


def test_resolve_metric_selection_returns_configured_registered_metrics():
    with pytest.warns(
        UserWarning,
        match="Configured metrics not registered and will be skipped",
    ):
        result = runner_module._resolve_metric_selection(
            selected_metrics=None,
            configured_metrics=['dummy', 'missing_metric'],
            registered_metrics=['dummy']
        )

    assert result == ['dummy']


def test_resolve_metric_selection_filters_selected_metrics():
    with pytest.warns(UserWarning) as warnings:
        result = runner_module._resolve_metric_selection(
            selected_metrics=['dummy', 'missing_metric'],
            configured_metrics=['dummy', 'other_metric'],
            registered_metrics=['dummy', 'other_metric']
        )

    warning_messages = [str(warning.message) for warning in warnings]

    assert result == ['dummy']
    assert any(
        "Selected metrics not present in config and will be skipped" in message
        for message in warning_messages
    )
    assert any(
        "Selected metrics not registered and will be skipped" in message
        for message in warning_messages
    )


def test_validate_context_metadata_requires_metadata():
    with pytest.raises(ValueError, match="metadata must also be provided"):
        runner_module._validate_context_metadata(None)


def test_validate_context_metadata_requires_expected_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        runner_module._validate_context_metadata(
            {
                "dataset_name": "dataset",
                "model_name": "model"
            }
        )


def test_run_evaluation_runs_registered_metric_without_saving_reports(
    monkeypatch,
    clean_registry,
    metric_context,
    dummy_metric_class
):
    METRIC_REGISTRY['dummy'] = dummy_metric_class

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime"
        },
        config={
            "metrics": [
                {"name": "dummy", "params": {"value": 3.0}}
            ]
        },
        report_output_dir=None
    )

    assert result['contexts'][0]['results']['dummy'] == 3.0
    assert result['contexts'][0]['skipped'] == {}
    assert result['report_paths'] == {}

    report = result['reports']['dataset']['model']
    report_by_metric = report.set_index("metric")

    assert report_by_metric.loc["dummy", "dataset_name"] == "dataset"
    assert report_by_metric.loc["dummy", "model_name"] == "model"
    assert report_by_metric.loc["dummy", "metric_params"] == {"value": 3.0}
    assert report_by_metric.loc['dummy', 'lime'] == 3.0


def test_run_evaluation_records_skipped_metrics(
    monkeypatch,
    clean_registry,
    metric_context
):
    class SkippedMetric(BaseMetric):
        NAME = "skipped"

        def run(self):
            raise MetricSkipped("not applicable")
        
    METRIC_REGISTRY['skipped'] = SkippedMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime",
        },
        config={"metrics": [{"name": "skipped"}]},
        report_output_dir=None,
    )

    assert result['contexts'][0]['results'] == {}
    assert result['contexts'][0]['skipped'] == {
        "skipped": "not applicable",
    }


def test_run_evaluation_passes_explain_func_by_xai_method(
    monkeypatch,
    clean_registry,
    metric_context
):
    def lime_explain_func():
        return "lime explanation"
    
    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def __init__(self, context, params=None, explain_func=None):
            super().__init__(context, params)
            self.explain_func = explain_func

        def run(self):
            return self.explain_func()
        
    METRIC_REGISTRY['explain_func_metric'] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime"
        },
        config={"metrics": [{"name": "explain_func_metric"}]},
        report_output_dir=None,
        explain_funcs={"lime": lime_explain_func}
    )

    assert result['contexts'][0]['results']['explain_func_metric'] == "lime explanation"


def test_run_evaluation_accepts_single_explain_func(
    monkeypatch,
    clean_registry,
    metric_context
):
    def explain_func():
        return "single explanation"
    
    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def __init__(self, context, params=None, explain_func=None):
            super().__init__(context, params)
            self.explain_func = explain_func

        def run(self):
            return self.explain_func()
        
    METRIC_REGISTRY['explain_func_metric'] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime"
        },
        config={"metrics": [{"name": "explain_func_metric"}]},
        report_output_dir=None,
        explain_func=explain_func
    )

    assert result['contexts'][0]['results']['explain_func_metric'] == "single explanation"


def test_run_evaluation_fails_when_explain_func_is_missing(
    monkeypatch,
    clean_registry,
    metric_context
):
    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def run(self):
            return 1.0
        
    METRIC_REGISTRY['explain_func_metric'] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="No explain_func provided"):
        run_evaluation(
            context=metric_context,
            metadata={
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "shap"
            },
            config={"metrics": [{"name": "explain_func_metric"}]},
            report_output_dir=None,
            explain_funcs={"lime": lambda: None}
        )