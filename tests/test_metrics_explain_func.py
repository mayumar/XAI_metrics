import importlib
from unittest.mock import Mock, patch

import pytest

EXPLAIN_METRICS = [
    (
        "XAI_metrics.metrics.robustness.local_lipschitz_estimate",
        "LocalLipschitzEstimate",
        "LocalLipschitzEstimate",
        "train",
    ),
    (
        "XAI_metrics.metrics.robustness.max_sensitivity",
        "MaxSensitivity",
        "MaxSensitivity",
        "train",
    ),
    (
        "XAI_metrics.metrics.robustness.relative_input_stability",
        "RelativeInputStability",
        "RelativeInputStability",
        "eval",
    ),
    (
        "XAI_metrics.metrics.robustness.relative_output_stability",
        "RelativeOutputStability",
        "RelativeOutputStability",
        "eval",
    ),
    (
        "XAI_metrics.metrics.sensitivity.avg_sensitivity",
        "AvgSensitivity",
        "AvgSensitivity",
        "train",
    ),
]

@pytest.mark.parametrize(
    "module_path,class_name,quantus_name,_expected_mode",
    EXPLAIN_METRICS
)
def test_explain_metrics_require_explain_func(
    metric_context,
    module_path,
    class_name,
    quantus_name,
    _expected_mode
):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    with pytest.raises(ValueError, match="requires 'explain_func'"):
        metric_cls(metric_context, params={"abs": True})

@pytest.mark.parametrize(
    "module_path,class_name,quantus_name,expected_mode",
    EXPLAIN_METRICS
)
def test_explain_metrics_run_with_explain_func(
    metric_context,
    explain_func,
    module_path,
    class_name,
    quantus_name,
    expected_mode
):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    quantus_runner = Mock(return_value=["ok"])

    with patch(f"{module_path}.quantus.{quantus_name}", return_value=quantus_runner):
        metric = metric_cls(
            metric_context,
            params={"abs": True, "normalise": False, "nr_samples": 7},
            explain_func=explain_func
        )
        result = metric.run()

    assert result == ["ok"]
    assert metric_context.model.mode == expected_mode

    call_kwargs = quantus_runner.call_args.kwargs
    assert call_kwargs["model"] is metric_context.model
    assert call_kwargs["a_batch"] is metric_context.attributions
    assert call_kwargs["explain_func"] is explain_func