# tests/test_reporting.py
import numpy as np
import pandas as pd
import json

from xai_metrics.reporting.reporting import (
    _serialize,
    _to_numeric_array,
    build_reports,
    save_reports
)

def test_serialize_converts_numpy_and_pandas_values():
    payload = {
        "array": np.array([1.0, 2.0]),
        "float": np.float64(3.5),
        "int": np.int64(2),
        "bool": np.bool_(True),
        "timestamp": pd.Timestamp("2024-01-01")
    }

    result = _serialize(payload)

    assert result["array"] == [1.0, 2.0]
    assert result["float"] == 3.5
    assert result["int"] == 2
    assert result["bool"] is True
    assert result["timestamp"] == "2024-01-01T00:00:00"


def test_to_numeric_array_returns_flat_non_nan_values():
    result = _to_numeric_array([[1, 2], [3, np.nan]])

    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_build_reports_groups_metrics_by_dataset_model_and_xai_method():
    context_outputs = [
        {
            "metadata": {
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "lime",
            },
            "results": {
                "MetricA": np.array([1.0, 2.0, 3.0]),
                "MetricB": {"part1": [2.0, 4.0]},
            },
            "metric_params": {
                "MetricA": {"normalise": True},
                "MetricB": {"base_strategy": "mean"},
            }
        },
        {
            "metadata": {
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "shap",
            },
            "results": {
                "MetricA": np.array([4.0, 6.0]),
                "MetricB": {"part1": [10.0]},
            },
            "metric_params": {
                "MetricA": {"normalise": True},
                "MetricB": {"base_strategy": "mean"},
            }
        }
    ]

    reports = build_reports(context_outputs)
    report = reports['dataset']['model']
    report_by_metric = report.set_index("metric")

    assert isinstance(report, pd.DataFrame)
    assert report.columns.tolist() == [
        "dataset_name",
        "model_name",
        "metric",
        "metric_params",
        "lime",
        "shap"
    ]
    assert report_by_metric.loc["MetricA", "dataset_name"] == "dataset"
    assert report_by_metric.loc["MetricA", "model_name"] == "model"
    assert report_by_metric.loc["MetricA", "metric_params"] == {"normalise": True}
    assert report_by_metric.loc["MetricA", "lime"] == 2.0
    assert report_by_metric.loc["MetricA", "shap"] == 5.0
    assert report_by_metric.loc["MetricB.part1", "metric_params"] == {"base_strategy": "mean"}
    assert report_by_metric.loc["MetricB.part1", "lime"] == 3.0
    assert report_by_metric.loc["MetricB.part1", "shap"] == 10.0


def test_build_reports_keeps_non_numeric_values_serialized():
    context_outputs = [
        {
            "metadata": {
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "lime",
            },
            "results": {
                "MetricText": {"status": "ok"},
                "MetricArray": np.array(["a", "b"]),
            },
        }
    ]

    report = build_reports(context_outputs)['dataset']['model']
    report_by_metric = report.set_index("metric")

    assert report_by_metric.loc["MetricText.status", "lime"] == "ok"
    assert report_by_metric.loc["MetricArray", "lime"] == ["a", "b"]
    assert report_by_metric.loc["MetricArray", "metric_params"] == {}


def test_save_reports_creates_csv_and_json(tmp_path):
    reports = {
        "dataset": {
            "model": pd.DataFrame(
                {
                    "dataset_name": ['dataset'],
                    "model_name": ['model'],
                    "metric": ['MetricA'],
                    "metric_params": [{"normalise": True}],
                    "lime": [1.0],
                    "shap": [2.0]
                }
            )
        }
    }

    paths = save_reports(reports, output_dir=tmp_path)

    csv_path = paths['csv']
    json_path = paths['json']

    assert csv_path.endswith("xai_metrics_report.csv")
    assert json_path.endswith("xai_metrics_report.json")

    saved_csv = pd.read_csv(csv_path)

    assert saved_csv.loc[0, "dataset_name"] == "dataset"
    assert saved_csv.loc[0, "model_name"] == "model"
    assert saved_csv.loc[0, "metric"] == "MetricA"
    assert saved_csv.loc[0, "metric_params"] == '{"normalise": true}'
    assert saved_csv.loc[0, "lime"] == 1.0
    assert saved_csv.loc[0, "shap"] == 2.0

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload['report'][0]['dataset_name'] == "dataset"
    assert payload['report'][0]['model_name'] == "model"
    assert payload['report'][0]['metric'] == "MetricA"
    assert payload['report'][0]['metric_params'] == {"normalise": True}
    assert payload['report'][0]['lime'] == 1.0
    assert payload['report'][0]['shap'] == 2.0