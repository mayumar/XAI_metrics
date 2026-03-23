# tests/test_reporting.py
import numpy as np
import pandas as pd
import json

import XAI_metrics.reporting.reporting as reporting_module
from XAI_metrics.reporting import load_scope_from_yaml, metrics_to_dataframe, metrics_report_markdown, save_metrics_report

def test_load_scope_from_yaml_reads_metric_types(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "metrics:\n"
            "  - name: LocalMetric\n"
            "    params:\n"
            "      Type: Local\n"
            "  - name: GlobalMetric\n"
            "    params:\n"
            "      Type: Global\n"
        ),
        encoding="utf-8",
    )

    result = load_scope_from_yaml(config_path)

    assert result == {"LocalMetric": "local", "GlobalMetric": "global"}

def test_serialize_converts_numpy_and_pandas_values():
    payload = {
        "array": np.array([1.0, 2.0]),
        "float": np.float64(3.5),
        "int": np.int64(2),
        "bool": np.bool_(True),
        "timestamp": pd.Timestamp("2024-01-01")
    }

    result = reporting_module._serialize(payload)

    assert result["array"] == [1.0, 2.0]
    assert result["float"] == 3.5
    assert result["int"] == 2
    assert result["bool"] is True
    assert result["timestamp"] == "2024-01-01T00:00:00"

def test_flatten_results_expandas_nested_mapping():
    results = {
        "MetricA": {"part1": 1, "part2": 2},
        "MetricB": 3
    }

    flattened = reporting_module._flatten_results(results)

    assert ("MetricA.part1", 1) in flattened
    assert ("MetricA.part2", 2) in flattened
    assert ("MetricB", 3) in flattened

def test_to_numeric_array_returns_flat_non_nan_values():
    result = reporting_module._to_numeric_array([[1, 2], [3, np.nan]])
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

def test_metrics_to_dataframe_creates_aggregate_and_observation_rows():
    metric_results = {"results": {"AnyMetric": np.array([0.2, 0.4])}}

    df = metrics_to_dataframe(metric_results, observations=["obs1", "obs2"])

    assert set(df["row_type"]) == {"aggregate", "observation"}
    assert set(df.columns) == {
        "metric",
        "scope",
        "row_type",
        "observation",
        "value",
        "value_raw",
        "n",
        "mean",
        "std",
        "min",
        "max",
    }

def test_metrics_to_dataframe_uses_numeric_index_when_observations_do_not_match():
    metric_results = {"results": {"AnyMetric": np.array([0.2, 0.4, 0.6])}}

    df = metrics_to_dataframe(metric_results, observations=["only_one"])
    obs_rows = df[df["row_type"] == "observation"]

    assert list(obs_rows["observation"]) == [0, 1, 2]

def test_fmt_num_formats_special_and_numeric_values():
    assert reporting_module._fmt_num(None) == "-"
    assert reporting_module._fmt_num(np.nan) == "-"
    assert reporting_module._fmt_num(1.2345678910) == "1.234568"
    assert reporting_module._fmt_num("x") == "x"

def test_metrics_report_markdown_returns_empty_message_for_no_results():
    report = metrics_report_markdown({"results": {}})
    assert "No metrics available." in report

def test_metrics_report_markdown_contains_table_sections():
    metrics_results = {"results": {"SomeMetric": np.array([0.2, 0.4])}}

    report = metrics_report_markdown(metrics_results, observations=["obs1", "obs2"])

    assert "# Metrics Report" in report
    assert "## Local Metrics (Aggregated)" in report
    assert "## Local Metrics (Per Observation)" in report

def test_save_metrics_report_creates_expected_files(tmp_path):
    metric_results = {"results": {"MetricA": np.array([0.2, 0.4])}}

    paths = save_metrics_report(
        metric_results,
        observations=["obs1", "obs2"],
        output_dir=tmp_path,
        report_name="report"
    )

    with open(paths["json"], "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert "generated_at" in payload
    assert "results" in payload
    assert "summary" in payload
    assert "observations" in payload
    assert payload["results"]["MetricA"] == [0.2, 0.4]