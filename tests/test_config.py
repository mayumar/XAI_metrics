# tests/test_config.py
import pandas as pd
import pytest
import torch.nn as nn
import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.base import MetricContext

def test_config_controller_loads_config_from_mapping():
    config = {
        "metrics": [
            {"name": "dummy", "params": {"value": 1.0}}
        ]
    }

    controller = ConfigController(config=config)

    assert controller.config == config
    assert controller.get_metrics_config() == config['metrics']


def test_config_controller_loads_config_from_yaml_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "metrics:\n"
            "  - name: dummy\n"
            "    params:\n"
            "      value: 3.0\n"
        ),
        encoding="utf-8"
    )

    controller = ConfigController(config=config_path)

    assert controller.get_metrics_config() == [
        {"name": "dummy", "params": {"value": 3.0}}
    ]


def test_validate_x_y_indexes_reorders_y_to_match_x():
    controller = ConfigController(config={})

    X_test = pd.DataFrame({"x1": [1.0, 2.0]}, index=[20, 10])
    y_test = pd.Series([0, 1], index=[10, 20])

    X_valid, y_valid = controller._validate_X_y_indexes(X_test, y_test)

    assert X_valid.index.tolist() == [20, 10]
    assert y_valid.index.tolist() == [20, 10]
    assert y_valid.tolist() == [1, 0]


def test_validate_x_y_indexes_rejects_mismatched_indexes():
    controller = ConfigController(config={})

    X_test = pd.DataFrame({"x1": [1.0]}, index=[1])
    y_test = pd.Series([0], index=[2])

    with pytest.raises(ValueError, match="X_test and y_test must contain the same indexes"):
        controller._validate_X_y_indexes(X_test, y_test)


def test_validate_observations_casts_to_x_test_index_dtype():
    controller = ConfigController(config={})

    observations = controller._validate_observations(
        observations=["1", "2"],
        X_test_index=pd.Index([1, 2, 3])
    )

    assert observations == [1, 2]


def test_build_context_loads_model_data_labels_and_attributions(tmp_path):
    X_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    attributions_path = tmp_path / "attributions.csv"

    pd.DataFrame(
        {"x1": [1.0, 2.0], "x2": [3.0, 4.0]},
        index=[10, 11],
    ).to_csv(X_test_path)

    pd.DataFrame(
        {"target": [0, 1]},
        index=[10, 11],
    ).to_csv(y_test_path)

    pd.DataFrame(
        {"x1": [0.1, 0.2], "x2": [0.3, 0.4]},
        index=[10, 11],
    ).to_csv(attributions_path)

    config = {
        "context": {
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime",
            "model_path": str(tmp_path / "model.pt"),
            "X_test_path": str(X_test_path),
            "y_test_path": str(y_test_path),
            "attributions_path": str(attributions_path),
            "device": "cpu",
        }
    }

    controller = ConfigController(
        config=config,
        model_loader=lambda model_path: nn.Identity()
    )

    context, metadata = controller.build_context()

    assert isinstance(context, MetricContext)
    assert isinstance(context.model, nn.Identity)
    assert context.X_test.shape == (2, 2)
    assert context.y_test.tolist() == [0, 1]
    assert context.observations == [10, 11]
    assert np.array_equal(
        context.attributions,
        np.array([[0.1, 0.3], [0.2, 0.4]])
    )
    assert context.device == "cpu"

    assert metadata == {
        "dataset_name": "dataset",
        "model_name": "model",
        "xai_method_name": "lime"
    }


def test_build_context_rejects_non_torch_model(tmp_path):
    X_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    attributions_path = tmp_path / "attributions.csv"

    pd.DataFrame({"x1": [1.0]}, index=[0]).to_csv(X_test_path)
    pd.DataFrame({"target": [0]}, index=[0]).to_csv(y_test_path)
    pd.DataFrame({"x1": [0.1]}, index=[0]).to_csv(attributions_path)

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "lime",
                "model_path": str(tmp_path / "model.pkl"),
                "X_test_path": str(X_test_path),
                "y_test_path": str(y_test_path),
                "attributions_path": str(attributions_path),
            }
        },
        model_loader=lambda model_path: object(),
    )

    with pytest.raises(TypeError, match="torch.nn.Module"):
        controller.build_context()