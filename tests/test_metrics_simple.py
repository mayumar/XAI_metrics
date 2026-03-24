# tests/test_metrics_simple.py
import pytest
import importlib
from unittest.mock import Mock, patch

SIMPLE_METRICS = [
    (
        "XAI_metrics.metrics.complexity.complexity_metric",
        "Complexity",
        "Complexity",
        "train",
        False,
    ),
    (
        "XAI_metrics.metrics.complexity.sparseness",
        "Sparseness",
        "Sparseness",
        "train",
        False,
    ),
    (
        "XAI_metrics.metrics.faithfulness.consistency",
        "Consistency",
        "Consistency",
        "eval",
        True,
    ),
    (
        "XAI_metrics.metrics.faithfulness.faithfulness_estimate",
        "FaithfulnessEstimate",
        "FaithfulnessEstimate",
        "eval",
        True,
    ),
    (
        "XAI_metrics.metrics.faithfulness.monotonicity_correlation",
        "MonotonicityCorrelation",
        "MonotonicityCorrelation",
        "eval",
        True,
    ),
    (
        "XAI_metrics.metrics.faithfulness.monotonicity",
        "Monotonicity",
        "Monotonicity",
        "eval",
        True,
    ),
    (
        "XAI_metrics.metrics.faithfulness.sensitivity_n",
        "SensitivityN",
        "SensitivityN",
        "eval",
        True,
    ),
    (
        "XAI_metrics.metrics.faithfulness.sufficiency",
        "Sufficiency",
        "Sufficiency",
        "eval",
        True,
    ),
    (
        "XAI_metrics.metrics.fidelity.completeness.completeness_metric",
        "Completeness",
        "Completeness",
        "eval",
        True,
    ),
    (
        "XAI_metrics.metrics.fidelity.soundness.non_sensitivity",
        "NonSensitivity",
        "NonSensitivity",
        "eval",
        True,
    ),
]

@pytest.mark.parametrize(
    "module_path,class_name,quantus_name,expected_mode,uses_values",
    SIMPLE_METRICS
)
def test_simple_metric_runs_quantus_contract(
    metric_context,
    module_path,
    class_name,
    quantus_name,
    expected_mode,
    uses_values
):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    quantus_runner = Mock(return_value=["ok"])

    with patch(f"{module_path}.quantus.{quantus_name}", return_value=quantus_runner):
        metric = metric_cls(metric_context, params={"abs": True, "normalise": False})
        result = metric.run()

    assert result == ["ok"]
    assert metric_context.model.mode == expected_mode

    call_kwargs = quantus_runner.call_args.kwargs
    assert call_kwargs["model"] is metric_context.model
    assert call_kwargs["a_batch"] is metric_context.attributions

    if uses_values:
        assert hasattr(call_kwargs["x_batch"], "shape")
        assert hasattr(call_kwargs["y_batch"], "shape")
    else:
        assert list(call_kwargs["x_batch"].index) == metric_context.observations
        assert list(call_kwargs["y_batch"].index) == metric_context.observations