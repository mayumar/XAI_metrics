# XAI_metrics/reporting/reporting.py
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from typing import Mapping, Any, Sequence, Dict

def _serialize(value: Any) -> Any:
    """
    Convert values to JSON-serializable Python objects.

    NumPy arrays, NumPy scalar values, booleans, timestamps, mappings and
    sequences are recursively converted into standard Python types that can be
    safely written to JSON.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    Any
        JSON-serializable representation of ``value``.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _to_numeric_array(value: Any) -> np.ndarray:
    """
    Convert a metric value to a one-dimensional numeric array.

    The function is used to extract numeric values from metric outputs so that
    they can be averaged in reports. Non-numeric values are converted to an
    empty array. Missing values represented as ``nan`` are removed.

    Parameters
    ----------
    value : Any
        Metric output value to convert.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of numeric values. If ``value`` cannot be
        converted to numeric values, an empty array is returned.
    """
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, (float, int, np.floating, np.integer)):
        return np.array([float(value)], dtype=float)
    try:
        arr = np.asanyarray(value, dtype=float).ravel()
    except(TypeError, ValueError):
        return np.array([], dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr[~np.isnan(arr)]


def build_reports(
    context_outputs: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build metric reports from context outputs.

    Each context output is expected to contain a ``metadata`` dictionary with
    ``dataset_name``, ``model_name`` and ``xai_method_name`` fields, and a
    ``results`` dictionary containing metric outputs.

    Reports are grouped by dataset and model. For each report, rows correspond
    to metric names and columns correspond to XAI methods. Numeric metric
    outputs are averaged before being inserted into the report. Non-numeric
    outputs are serialized and inserted as-is.

    If a metric output is a mapping, each key is expanded into a separate row
    using the format ``"<metric_name>.<key>"``.

    Parameters
    ----------
    context_outputs : Sequence[Mapping[str, Any]]
        Sequence of context output dictionaries. Each dictionary must contain
        ``metadata`` and ``results`` entries.

    Returns
    -------
    Dict[str, Dict[str, pandas.DataFrame]]
        Nested dictionary of reports. The first level is indexed by dataset
        name, the second level by model name, and each value is a report
        dataframe.
    """
    grouped = {}

    for context_out in context_outputs:
        metadata = context_out['metadata']

        dataset_name = metadata['dataset_name']
        model_name = metadata['model_name']
        xai_method_name = metadata['xai_method_name']

        metrics = {}
        for metric_name, metric_value in context_out["results"].items():
            if isinstance(metric_value, Mapping):
                values_to_report = {
                    f"{metric_name}.{key}": value
                    for key, value in metric_value.items()
                }
            else:
                values_to_report = {metric_name: metric_value}

            for report_metric_name, value in values_to_report.items():
                numeric_values = _to_numeric_array(value)

                if numeric_values.size:
                    metrics[report_metric_name] = float(np.mean(numeric_values))
                else:
                    metrics[report_metric_name] = _serialize(value)

        grouped.setdefault((dataset_name, model_name), {})
        grouped[(dataset_name, model_name)][xai_method_name] = metrics


    reports = {}

    for (dataset_name, model_name), methods_results in grouped.items():
        report_df = pd.DataFrame(methods_results)
        report_df.index.name = "metric"

        reports.setdefault(dataset_name, {})
        reports[dataset_name][model_name] = report_df.sort_index()

    return reports


def save_reports(
    reports: Mapping[str, Mapping[str, pd.DataFrame]],
    output_dir: str | Path,
) -> Dict[str, dict[str, dict[str, str]]]:
    """
    Save metric reports to CSV and JSON files.

    For each dataset and model report, this function creates two files in
    ``output_dir``: one CSV file containing the report dataframe and one JSON
    file containing the dataset name, model name and serialized report records.

    Parameters
    ----------
    reports : Mapping[str, Mapping[str, pandas.DataFrame]]
        Nested dictionary of reports, usually returned by :func:`build_reports`.
        The first level is indexed by dataset name and the second level by
        model name.
    output_dir : str or pathlib.Path
        Directory where report files will be saved. The directory is created if
        it does not exist.

    Returns
    -------
    dict[str, dict[str, dict[str, str]]]
        Nested dictionary with the saved file paths. For each dataset and model,
        the dictionary contains the keys ``"csv"`` and ``"json"``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for dataset_name, dataset_reports in reports.items():
        saved_paths.setdefault(dataset_name, {})

        for model_name, report_df in dataset_reports.items():
            report_name = f"{dataset_name}_{model_name}_xai_metrics_report"

            csv_path = out_dir / f"{report_name}.csv"
            json_path = out_dir / f"{report_name}.json"

            report_df.to_csv(csv_path, index=True)

            payload = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "report": _serialize(report_df.reset_index().to_dict(orient="records")),
            }

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            saved_paths[dataset_name][model_name] = {
                "csv": str(csv_path),
                "json": str(json_path),
            }

    return saved_paths