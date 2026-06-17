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
    if value is pd.NA:
        return None
    if isinstance(value, np.ndarray):
        return _serialize(value.tolist())
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
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


def _to_csv_cell(value: Any) -> Any:
    """
    Convert a report value into a CSV-compatible value.

    Structured values such as dictionaries and lists are encoded as JSON
    strings. Scalar values are returned directly.

    Parameters
    ----------
    value : Any
        Report cell value.

    Returns
    -------
    Any
        CSV-compatible representation of ``value``.
    """
    serialized = _serialize(value)

    if isinstance(serialized, (Mapping, list)):
        return json.dumps(
            serialized,
            ensure_ascii=False,
            sort_keys=True,
            default=str
        )
    
    return serialized


def build_reports(
    context_outputs: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build metric reports from context outputs.

    Each contxt output must contain a ``metadata`` dictionary and a ``results`` dictionary. The metadata must define ``dataset_name``, ``model_name`` and ``xai_method_name``. An optional ``metric_params`` mapping may be associate each metric with the parameters used during its evaluation.

    Results are grouped by dataset and model. The resulting dataframe contains
    the fixed columns ``dataset_name``, ``model_name``, ``metric`` and
    ``metric_params``, followed by one result column for each XAI method.
    
    Numeric metric results are flattened, missing values are removed, and the
    remaining values are represented by their mean. Non-numeric values are
    serialized without aggregation. When a metric result is a mapping, each
    entry is expanded into an independent report row using the name
    ``"<metric_name>.<result_key>"``. All XAI methods evaluated for the same
    dataset and model must use identical parameters for a given report metric.
    
    Parameters
    ----------
    context_outputs : Sequence[Mapping[str, Any]]
        Sequence of context output mappings. Each element must contain:
        
        - ``metadata`` : Mapping[str, Any]
          Metadata identifying the dataset, model and XAI method.
        - ``results`` : Mapping[str, Any]
          Results indexed by metric name.
        - ``metric_params`` : Mapping[str, Any], optional
          Parameters used by each metric.
        
    Returns
    -------
    Dict[str, Dict[str, pandas.DataFrame]]
        Nested report dictionary. The first level is indexed by dataset name
        and the second level by model name. Each value is a dataframe with one
        row per metric and one result column per XAI method.
    
    Raises
    ------
    KeyError
        If a context output does not contain the required metadata or result
        fields.
    ValueError
        If more than one context output is provided for the same dataset,
        model and XAI method; if different parameter configurations are found
        for the same metric; or if an XAI method name conflicts with a
        reserved report column.
    """
    grouped = {}

    for context_out in context_outputs:
        metadata = context_out['metadata']

        dataset_name = metadata['dataset_name']
        model_name = metadata['model_name']
        xai_method_name = metadata['xai_method_name']

        context_metric_params = context_out.get("metric_params", {})

        group = grouped.setdefault(
            (dataset_name, model_name),
            {
                "methods": {},
                "metric_params": {}
            }
        )

        if xai_method_name in group['methods']:
            raise ValueError(
                "Duplicated context output for "
                f"dataset '{dataset_name}', "
                f"model '{model_name}' and "
                f"XAI method '{xai_method_name}'."
            )
        
        methods_results = {}

        metrics = {}
        for metric_name, metric_value in context_out['results'].items():
            metric_params = _serialize(
                context_metric_params.get(metric_name, {})
            )

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

                stored_params = group['metric_params']
                previous_params = stored_params.get(report_metric_name)

                if previous_params is None:
                    stored_params[report_metric_name] = metric_params
                else:
                    previous_params_signature = json.dumps(
                        _serialize(previous_params),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str
                    )
                    
                    metric_params_signature = json.dumps(
                        _serialize(metric_params),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str
                    )

                    if previous_params_signature != metric_params_signature:
                        raise ValueError(
                            "Different parameters were found for metric "
                            f"'{report_metric_name}' in dataset "
                            f"'{dataset_name}' and model '{model_name}'. "
                            "A single metric_params column requires the "
                            "parameters to be identical for every XAI method."
                        )
                    else:
                        stored_params[report_metric_name] = metric_params

        group['methods'][xai_method_name] = metrics

    reports = {}

    for (dataset_name, model_name), group in grouped.items():
        methods_results = group['methods']
        params_by_metric = group['metric_params']

        reserved_columns = {
            "dataset_name",
            "model_name",
            "metric",
            "metric_params"
        }

        conflicting_methods = reserved_columns.intersection(methods_results)

        if conflicting_methods:
            raise ValueError(
                "The following XAI method names conflict with reserved "
                f"report columns: {sorted(conflicting_methods)}."
            )
        
        report_df = pd.DataFrame(methods_results)
        report_df.index.name = "metric"

        report_df = report_df.sort_index().reset_index()

        report_df.insert(0, "dataset_name", dataset_name)
        report_df.insert(1, "model_name", model_name)
        report_df.insert(
            3,
            "metric_params",
            report_df['metric'].map(
                lambda metric_name: params_by_metric.get(metric_name, {})
            )
        )

        reports.setdefault(dataset_name, {})
        reports[dataset_name][model_name] = report_df

    return reports


def save_reports(
    reports: Mapping[str, Mapping[str, pd.DataFrame]],
    output_dir: str | Path,
    report_name: str = "xai_metrics_report"
) -> Dict[str, str]:
    """ 
    Save all metric reports as consolidated CSV and JSON files.
    
    The report dataframes for every dataset and model are concatenated into a
    single report. Rows are sorted by dataset name, model name and metric
    name, while XAI method columns are sorted alphabetically.
    
    Structured CSV cell values, such as metric parameter dictionaries, are
    encoded as JSON strings. The JSON file contains the complete report as a
    list of serialized record mappings under the ``"report"`` key.
    
    Parameters
    ----------
    reports : Mapping[str, Mapping[str, pandas.DataFrame]]
        Nested dictionary of report dataframes, usually returned by
        :func:`build_reports`. The first level must be indexed by dataset name
        and the second level by model name.
    output_dir : str or pathlib.Path
        Directory where the report files will be saved. The directory and its
        missing parent directories are created automatically.
    report_name : str, default="xai_metrics_report"
        Base filename used for the generated report files. The ``.csv`` and
        ``.json`` extensions are added automatically.
        
    Returns
    -------
    Dict[str, str]
        Dictionary containing the paths of the generated files under the keys ``"csv"`` and ``"json"``.
        
    Raises
    ------
    ValueError
        If ``reports`` does not contain any report dataframes to save.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_frames = []

    for dataset_reports in reports.values():
        for report_df in dataset_reports.values():
            report_frames.append(report_df.copy())

    if not report_frames:
        raise ValueError("No reoprts are available to save.")
    
    combined_report = pd.concat(
        report_frames,
        ignore_index=True,
        sort=False
    )

    fixed_columns = [
        "dataset_name",
        "model_name",
        "metric",
        "metric_params"
    ]

    xai_method_columns = [
        column
        for column in combined_report.columns
        if column not in fixed_columns
    ]

    combined_report = combined_report[
        fixed_columns + sorted(xai_method_columns)
    ]

    combined_report = combined_report.sort_values(
        by=["dataset_name", "model_name", "metric"],
        ignore_index=True
    )

    csv_path = out_dir / f"{report_name}.csv"
    json_path = out_dir / f"{report_name}.json"

    csv_df = combined_report.copy()

    for column_name in csv_df.columns:
        csv_df[column_name] = csv_df[column_name].map(
            _to_csv_cell
        )

    csv_df.to_csv(csv_path, index=False, encoding="utf-8")

    payload = {
        "report": _serialize(
            combined_report.to_dict(orient="records")
        )
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        )

    return {
        "csv": str(csv_path),
        "json": str(json_path),
    }