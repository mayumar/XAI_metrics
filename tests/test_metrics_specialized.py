import importlib
from dataclasses import replace
from unittest.mock import Mock, patch, call
import numpy as np
import pandas as pd
import pytest

@pytest.mark.parametrize(
    "module_path,class_name,metric_name",
    [
        ("XAI_metrics.metrics.faithfulness.pgi", "PGI", "PGI"),
        ("XAI_metrics.metrics.faithfulness.pgu", "PGU", "PGU")
    ]
)
def test_openxai_metrics_run(metric_context, module_path, class_name, metric_name):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    evaluator_instance = Mock()
    evaluator_instance.evaluate.return_value = ([0.2, 0.8], 0.5)

    with patch(f"{module_path}.Evaluator", return_value=evaluator_instance) as evaluator_cls, \
         patch(f"{module_path}.NormalPerturbation", return_value="fake_perturbation") as perturbation_cls:
        metric = metric_cls(
            metric_context,
            params={
                "k": 0.5,
                "AUC": False,
                "std": 0.2,
                "n_samples": 12,
                "seed": 7,
                "n_jobs": 3
            }
        )
        result = metric.run()

    assert result == [0.2, 0.8]
    assert metric_context.model.mode == "eval"

    evaluator_cls.assert_called_once_with(metric_context.model, metric_name)

    perturbation_cls.assert_called_once()
    perturb_args = perturbation_cls.call_args
    assert perturb_args.args == ("tabular",)
    assert perturb_args.kwargs["mean"] == 0.0
    assert perturb_args.kwargs["std_dev"] == 0.2
    assert perturb_args.kwargs["flip_percentage"] == pytest.approx(np.sqrt(2 / np.pi) * 0.2)

    call_kwargs = evaluator_instance.evaluate.call_args.kwargs
    assert call_kwargs["k"] == 0.5
    assert call_kwargs["AUC"] is False
    assert call_kwargs["n_samples"] == 12
    assert call_kwargs["seed"] == 7
    assert call_kwargs["n_jobs"] == 3
    assert call_kwargs["perturb_method"] == "fake_perturbation"
    assert list(call_kwargs["feature_metadata"]) == ["f1", "f2"]

    np.testing.assert_allclose(
        call_kwargs["inputs"].numpy(),
        metric_context.X_test.loc[metric_context.observations].values
    )
    np.testing.assert_allclose(
        call_kwargs["explanations"].numpy(),
        metric_context.attributions
    )

@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("XAI_metrics.metrics.faithfulness.pgi", "PGI"),
        ("XAI_metrics.metrics.faithfulness.pgu", "PGU")
    ]
)
def test_openxai_metrics_use_default_params(metric_context, module_path, class_name):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    evaluator_instance = Mock()
    evaluator_instance.evaluate.return_value = ([1.0], 1.0)

    with patch(f"{module_path}.Evaluator", return_value=evaluator_instance), \
         patch(f"{module_path}.NormalPerturbation", return_value="fake_perturbation") as perturbation_cls:
        metric = metric_cls(metric_context)
        metric.run()

    perturb_kwargs = perturbation_cls.call_args.kwargs
    assert perturb_kwargs["std_dev"] == 0.1
    assert perturb_kwargs["flip_percentage"] == pytest.approx(np.sqrt(2 / np.pi) * 0.1)

    call_kwargs = evaluator_instance.evaluate.call_args.kwargs
    assert call_kwargs["k"] == 0.25
    assert call_kwargs["AUC"] is True
    assert call_kwargs["n_samples"] == 100
    assert call_kwargs["seed"] == -1
    assert call_kwargs["n_jobs"] == -1

@pytest.mark.parametrize(
    "module_path,class_name,function_name,expected_scores",
    [
        (
            "XAI_metrics.metrics.faithfulness.faithfulness",
            "Faithfulness",
            "faithfulness_metric",
            [0.25, 0.75]
        ),
        (
            "XAI_metrics.metrics.faithfulness.monotonicity_metric",
            "MonotonicityMetric",
            "monotonicity_metric",
            [True, False]
        )
    ]
)
def test_aix360_metrics_run_with_explicit_base_values(
    metric_context,
    module_path,
    class_name,
    function_name,
    expected_scores
):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    mocked_metric = Mock(side_effect=expected_scores)
    base_values = np.asarray([9.0, 8.0], dtype=float)

    with patch(f"{module_path}.{function_name}", mocked_metric):
        metric = metric_cls(
            metric_context,
            params={"base_values": base_values}
        )
        result = metric.run()

    assert result == expected_scores
    assert mocked_metric.call_count == 2

    calls = mocked_metric.call_args_list

    assert calls[0].kwargs["model"] is metric_context.model
    np.testing.assert_allclose(calls[0].kwargs["x"], [1.0, 2.0])
    np.testing.assert_allclose(calls[0].kwargs["coefs"], [0.1, 0.9])
    np.testing.assert_allclose(calls[0].kwargs["base"], base_values)

    assert calls[1].kwargs["model"] is metric_context.model
    np.testing.assert_allclose(calls[1].kwargs["x"], [3.0, 4.0])
    np.testing.assert_allclose(calls[1].kwargs["coefs"], [0.3, 0.7])
    np.testing.assert_allclose(calls[1].kwargs["base"], base_values)

@pytest.mark.parametrize(
    "module_path,class_name,function_name",
    [
        (
            "XAI_metrics.metrics.faithfulness.faithfulness",
            "Faithfulness",
            "faithfulness_metric"
        ),
        (
            "XAI_metrics.metrics.faithfulness.monotonicity_metric",
            "MonotonicityMetric",
            "monotonicity_metric"
        )
    ]
)
def test_aix360_metrics_use_mean_base_by_default(
    metric_context,
    module_path,
    class_name,
    function_name
):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    mocked_metric = Mock(side_effect=[0.1, 0.2])

    with patch(f"{module_path}.{function_name}", mocked_metric):
        metric = metric_cls(metric_context)
        metric.run()

    expected_base = np.asarray([2.0, 3.0], dtype=float)

    for metric_call in mocked_metric.call_args_list:
        np.testing.assert_allclose(metric_call.kwargs["base"], expected_base)

@pytest.mark.parametrize(
    "base_strategy,expected_base",
    [
        ("median", np.asarray([20.0, 30.0], dtype=float)),
        ("zero", np.asarray([0.0, 0.0], dtype=float))
    ]
)
def test_aix360_metrics_support_base_strategy_and_x_reference(
    metric_context,
    base_strategy,
    expected_base
):
    module_path = "XAI_metrics.metrics.faithfulness.faithfulness"
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, "Faithfulness")

    x_reference = pd.DataFrame(
        [[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]],
        columns=["f1", "f2"]
    )
    context = replace(
        metric_context,
        extras={"X_reference": x_reference}
    )

    mocked_metric = Mock(side_effect=[0.4, 0.6])

    with patch(f"{module_path}.faithfulness_metric", mocked_metric):
        metric = metric_cls(context, params={"base_strategy": base_strategy})

    for metric_call in mocked_metric.call_args_list:
        np.testing.assert_allclose(metric_call.kwargs["base"], expected_base)

def test_monotonicity_metric_uses_prediction_model_from_extras(metric_context):
    module_path = "XAI_metrics.metrics.faithfulness.monotonicity_metric"
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, "MonotonicityMetric")

    prediction_model = Mock(name="prediction_model")
    context = replace(
        metric_context,
        extras={"prediction_model": prediction_model}
    )

    mocked_metric = Mock(side_effect=[True, True])

    with patch(f"{module_path}.monotonicity_metric", mocked_metric):
        metric = metric_cls(context)
        result = metric.run()

    assert result == [True, True]

    for metric_call in mocked_metric.call_args_list:
        assert metric_call.kwargs["model"] is prediction_model

@pytest.mark.parametrize(
    "module_path,class_name",
    [
        (
            "XAI_metrics.metrics.faithfulness.faithfulness",
            "Faithfulness",
        ),
        (
            "XAI_metrics.metrics.faithfulness.monotonicity_metric",
            "MonotonicityMetric",
        )
    ]
)
def test_aix360_metrics_raise_for_unknown_base_strategy(
    metric_context,
    module_path,
    class_name
):
    module = importlib.import_module(module_path)
    metric_cls = getattr(module, class_name)

    metric = metric_cls(metric_context, params={"base_strategy": "invalid"})

    with pytest.raises(ValueError, match="Unknown base_strategy: invalid"):
        metric.run()