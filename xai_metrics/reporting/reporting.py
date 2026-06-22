# xai_metrics/reporting/reporting.py
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from typing import Mapping, Any, Sequence, Dict, List

def _serialize(value: Any) -> Any:
    """
    Convert values to JSON-serializable Python objects.

    NumPy arrays, NumPy scalar values, booleans, timestamps, mappings and
    sequences are recursively converted into standard Python types that can be
    safely written to JSON. Non-finite floating-point values are converted to
    ``None``.

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
    Convert a metric output to a one-dimensional numeric array.

    The function is used to extract numeric values from metric outputs so that
    they can be averaged in summary reports. Non-numeric values are converted
    to an empty array. Missing values represented as ``nan`` are removed.

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

    Structured values such as dictionaries and lists are serialized as JSON
    strings. Scalar values are returned directly after JSON-safe conversion.

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


def _timestamp_suffix(experiment_started_at: Any | None = None) -> str:
    """
    Build the timestamp suffix used in generated report filenames.

    Parameters
    ----------
    experiment_started_at : Any or None, optional
        Timestamp associated with the experiment. If ``None``, the current
        datetime is used. Datetime objects are formatted as
        ``YYYYMMDD_HHMMSS``. Other values are converted to strings.

    Returns
    -------
    str
        Timestamp suffix used to make report filenames unique.
    """
    if experiment_started_at is None:
        experiment_started_at = datetime.datetime.now()

    if isinstance(experiment_started_at, datetime.datetime):
        return experiment_started_at.strftime("%Y%m%d_%H%M%S")
    
    return str(experiment_started_at)


def _observation_values(value: Any, observations: Sequence[Any]) -> List[Any] | None:
    """
    Convert a metric output into one value per observation.

    The function is used by observation-level reports. It attempts to transform
    a metric output into a list whose length matches the number of evaluated
    observations. If this is not possible, ``None`` is returned.

    Parameters
    ----------
    value : Any
        Metric output value to convert.
    observations : Sequence[Any]
        Observation identifiers expected in the observation-level report.

    Returns
    -------
    List[Any] or None
        Serialized values aligned with ``observations``. ``None`` is returned
        when the metric output cannot be aligned with the observation list.
    """
    serialized = _serialize(value)
    n_observations = len(observations)

    if isinstance(value, pd.Series):
        values = value.to_list()
    elif isinstance(value, pd.DataFrame):
        values = value.to_numpy().tolist()
    elif isinstance(value, np.ndarray):
        arr = np.asarray(value)

        if arr.ndim == 0:
            values = [arr.item()]
        elif arr.shape[0] == n_observations:
            values = arr.tolist()
        else:
            values = arr.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [serialized]

    if len(values) != n_observations:
        return None

    return [_serialize(item) for item in values]


def _safe_path_name(value: Any) -> str:
    """
    Convert a value into a filesystem-safe path component.

    Characters that are not alphanumeric and are not ``.``, ``_`` or ``-`` are
    replaced with underscores. If the resulting value is empty, ``"report"`` is
    returned.

    Parameters
    ----------
    value : Any
        Value to convert into a safe path component.

    Returns
    -------
    str
        Filesystem-safe string.
    """
    safe = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(value)
    ).strip("_")

    return safe or "report"


def build_reports(
    context_outputs: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build summary metric reports from context outputs.

    Each context output must contain a ``metadata`` mapping and a ``results``
    sequence. Metadata must define ``dataset_name``, ``model_name`` and
    ``xai_method_name``. Each metric result in ``results`` must contain the
    fields ``metric`` and ``value`` and may contain ``metric_params``.

    Results are grouped by metric, dataset, model and metric parameters. Each
    generated dataframe contains the fixed columns ``dataset_name``,
    ``model_name``, ``metric`` and ``metric_params``, followed by one result
    column for each XAI method.

    Numeric metric values are flattened, ``nan`` values are removed and the
    remaining values are averaged. Non-numeric values are serialized without
    aggregation. If a metric value is a mapping, each entry is expanded into a
    separate report row named ``"<metric_name>.<result_key>"``.

    Parameters
    ----------
    context_outputs : Sequence[Mapping[str, Any]]
        Sequence of context output mappings. Each element must contain:

        - ``metadata`` : Mapping[str, Any]
          Metadata identifying the dataset, model and XAI method.
        - ``results`` : Sequence[Mapping[str, Any]]
          Metric result entries. Each entry must contain ``metric`` and
          ``value`` and may contain ``metric_params``.

    Returns
    -------
    Dict[str, Dict[str, pandas.DataFrame]]
        Nested report dictionary. The first level is indexed by dataset name
        and the second level by model name. Each dataframe has one row per
        report metric and parameter configuration, and one result column per
        XAI method.

    Raises
    ------
    KeyError
        If a context output or metric result does not contain the required
        fields.
    ValueError
        If an XAI method name conflicts with a reserved report column or if a
        duplicated result is found for the same metric, parameter configuration
        and XAI method.
    """
    grouped = {}

    reserved_columns = {
        "dataset_name",
        "model_name",
        "metric",
        "metric_params"
    }

    for context_out in context_outputs:
        metadata = context_out['metadata']

        dataset_name = metadata['dataset_name']
        model_name = metadata['model_name']
        xai_method_name = metadata['xai_method_name']

        if xai_method_name in reserved_columns:
            raise ValueError(
                f"XAI method name '{xai_method_name}' conflicts "
                "with a reserved report column."
            )
        
        for metric_result in context_out['results']:
            metric_name = metric_result['metric']
            metric_value = metric_result['value']
            metric_params = _serialize(
                metric_result.get("metric_params", {})
            )

            metric_params_signature = json.dumps(
                metric_params,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            if isinstance(metric_value, Mapping):
                values_to_report = {
                    f"{metric_name}.{key}": value
                    for key, value in metric_value.items()
                }
            else:
                values_to_report = {metric_name: metric_value}

            for report_metric_name, value in values_to_report.items():
                group_key = (
                    report_metric_name,
                    dataset_name,
                    model_name,
                    metric_params_signature,
                )

                group = grouped.setdefault(
                    group_key,
                    {
                        "metric_params": metric_params,
                        "methods": {},
                    },
                )

                if xai_method_name in group["methods"]:
                    raise ValueError(
                        "Duplicated result for metric "
                        f"'{report_metric_name}', parameters "
                        f"{metric_params} and XAI method "
                        f"'{xai_method_name}'."
                    )

                numeric_values = _to_numeric_array(value)

                if numeric_values.size:
                    group['methods'][xai_method_name] = float(np.mean(numeric_values))
                else:
                    group['methods'][xai_method_name] = _serialize(value)

    rows_by_context = {}

    for (metric_name, dataset_name, model_name, _), group in grouped.items():
        row = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "metric": metric_name,
            "metric_params": group["metric_params"],
            **group["methods"],
        }

        context_key = (dataset_name, model_name)

        rows_by_context.setdefault(context_key, []).append(row)
        
    reports = {}

    for (dataset_name, model_name), rows in rows_by_context.items():
        report_df = pd.DataFrame(rows)

        fixed_columns = [
            "dataset_name",
            "model_name",
            "metric",
            "metric_params",
        ]

        xai_method_columns = sorted(
            column
            for column in report_df.columns
            if column not in fixed_columns
        )

        report_df = report_df[
            fixed_columns + xai_method_columns
        ]

        report_df = report_df.sort_values(
            by=["metric"],
            ignore_index=True,
        )

        reports.setdefault(dataset_name, {})
        reports[dataset_name][model_name] = report_df

    return reports


def build_observation_reports(
    context_outputs: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Build observation-level metric reports from context outputs.

    Observation-level reports preserve one metric value per evaluated
    observation instead of aggregating metric outputs. Reports are grouped by
    metric, dataset and model. Each dataframe contains one row per observation
    and one column per XAI method.

    Metric outputs that cannot be aligned with the observation list are stored
    as ``None`` for the corresponding XAI method. Different parameter
    configurations for the same metric are kept as separate rows because the
    grouping includes the serialized ``metric_params`` signature.

    Parameters
    ----------
    context_outputs : Sequence[Mapping[str, Any]]
        Sequence of context output mappings. Each element must contain:

        - ``metadata`` : Mapping[str, Any]
          Metadata identifying the dataset, model and XAI method.
        - ``results`` : Sequence[Mapping[str, Any]]
          Metric result entries. Each entry must contain ``metric`` and
          ``value`` and may contain ``metric_params``.
        - ``observations`` : Sequence[Any]
          Observation identifiers corresponding to the metric outputs.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, pandas.DataFrame]]]
        Nested report dictionary. The first level is indexed by metric name,
        the second by dataset name and the third by model name.

    Raises
    ------
    ValueError
        If a context output does not include ``observations``, if observation
        indexes differ across XAI methods for the same report, or if a
        duplicated observation-level result is found for the same metric,
        parameter configuration and XAI method.
    KeyError
        If required metadata or metric result fields are missing.
    """
    grouped = {}

    for context_out in context_outputs:
        metadata = context_out['metadata']

        dataset_name = metadata['dataset_name']
        model_name = metadata['model_name']
        xai_method_name = metadata['xai_method_name']

        observations = list(context_out.get("observations", []))

        if not observations:
            raise ValueError(
                "Observation reports require each context output to include "
                "an 'observations' field."
            )
        

        for metric_result in context_out['results']:
            metric_name = metric_result['metric']
            metric_value = metric_result['value']
            metric_params = _serialize(metric_result.get("metric_params", {}))

            metric_params_signature = json.dumps(
                metric_params,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            if isinstance(metric_value, Mapping):
                values_to_report = {
                    f"{metric_name}.{key}": value
                    for key, value in metric_value.items()
                }
            else:
                values_to_report = {metric_name: metric_value}

            for report_metric_name, value in values_to_report.items():
                group_key = (
                    report_metric_name,
                    dataset_name,
                    model_name,
                    metric_params_signature,
                )

                group = grouped.setdefault(
                    group_key,
                    {
                        "metric_params": metric_params,
                        "observations": observations,
                        "methods": {},
                    },
                )

                if list(group["observations"]) != observations:
                    raise ValueError(
                        "Observation indexes must be identical for every XAI "
                        f"method in metric '{report_metric_name}', dataset "
                        f"'{dataset_name}' and model '{model_name}'."
                    )

                if xai_method_name in group["methods"]:
                    raise ValueError(
                        "Duplicated observation result for metric "
                        f"'{report_metric_name}', parameters "
                        f"{metric_params} and XAI method "
                        f"'{xai_method_name}'."
                    )

                observation_values = _observation_values(
                    value,
                    observations,
                )

                if observation_values is None:
                    observation_values = [
                        None for _ in observations
                    ]

                group["methods"][xai_method_name] = (
                    observation_values
                )

    frames_by_report = {}

    for (metric_name, dataset_name, model_name, _), group in grouped.items():
        report_df = pd.DataFrame(group['methods'], index=group['observations'])

        report_df.index.name = "observation"
        report_df = report_df.reset_index()

        report_df.insert(0, "dataset_name", dataset_name)
        report_df.insert(1, "model_name", model_name)
        report_df.insert(
            3,
            "metric_params",
            [group["metric_params"] for _ in range(len(report_df))]
        )

        report_key = (
            metric_name,
            dataset_name,
            model_name,
        )

        frames_by_report.setdefault(
            report_key,
            [],
        ).append(report_df)

    reports = {}

    for (metric_name, dataset_name, model_name), report_frames in frames_by_report.items():
        report_df = pd.concat(
            report_frames,
            ignore_index=True,
            sort=False
        )

        fixed_columns = [
            "dataset_name",
            "model_name",
            "observation",
            "metric_params",
        ]

        xai_method_columns = sorted(
            column
            for column in report_df.columns
            if column not in fixed_columns
        )

        report_df = report_df[
            fixed_columns + xai_method_columns
        ]

        reports.setdefault(metric_name, {})
        reports[metric_name].setdefault(dataset_name, {})
        reports[metric_name][dataset_name][model_name] = report_df

    return reports


def save_attributions(
    context_outputs: Sequence[Mapping[str, Any]],
    output_dir: str | Path
) -> Dict[str, str]:
    """
    Save generated attributions to CSV files.

    For each context output, the attribution matrix is converted into a
    dataframe using the stored observation identifiers as index and feature
    names as columns. Files are saved under
    ``<output_dir>/<dataset>/<model>/<xai_method>/``.

    Parameters
    ----------
    context_outputs : Sequence[Mapping[str, Any]]
        Sequence of context output mappings. Each element must contain
        ``metadata``, ``attributions``, ``observations`` and ``features``.
    output_dir : str or pathlib.Path
        Root directory where attribution files will be saved. Missing
        directories are created automatically.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping each XAI method name to the path of the generated
        attribution CSV file.

    Raises
    ------
    KeyError
        If any required field is missing from a context output.
    """
    output_root = Path(output_dir)
    paths = {}

    for context_out in context_outputs:
        metadata = context_out['metadata']

        dataset_name = str(metadata['dataset_name'])
        model_name = str(metadata['model_name'])
        xai_method_name = str(metadata['xai_method_name'])

        attributions_df = pd.DataFrame(
            context_out['attributions'],
            index=context_out['observations'],
            columns=context_out['features']
        )

        method_dir = output_root / dataset_name / model_name / xai_method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        path = (
            method_dir
            / f"{dataset_name}_{model_name}_{xai_method_name.lower()}_attributions.csv"
        )

        attributions_df.to_csv(path)
        paths[xai_method_name] = str(path)

    return paths


def save_reports(
    reports: Mapping[str, Mapping[str, pd.DataFrame]],
    output_dir: str | Path,
    report_name: str = "xai_metrics_report",
    observation_reports: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]] | None = None,
    experiment_started_at: Any | None = None
) -> Dict[str, str]:
    """
    Save summary reports and optional observation-level reports.

    Summary report dataframes for every dataset and model are concatenated into
    a single report. Rows are sorted by dataset name, model name and metric
    name, while XAI method columns are sorted alphabetically.

    The summary report is saved as both CSV and JSON. Structured CSV cell
    values, such as metric parameter dictionaries, are encoded as JSON strings.
    The JSON file contains the complete summary report as serialized records
    under the ``"report"`` key.

    If ``observation_reports`` is provided, one CSV file is saved for each
    metric. These files preserve observation-level metric values and are stored
    in metric-specific subdirectories.

    Parameters
    ----------
    reports : Mapping[str, Mapping[str, pandas.DataFrame]]
        Nested dictionary of summary report dataframes, usually returned by
        :func:`build_reports`. The first level must be indexed by dataset name
        and the second level by model name.
    output_dir : str or pathlib.Path
        Directory where the report files will be saved. The directory and its
        missing parent directories are created automatically.
    report_name : str, default="xai_metrics_report"
        Base filename used for the generated summary report files. A timestamp
        suffix is added automatically.
    observation_reports : Mapping[str, Mapping[str, Mapping[str, pandas.DataFrame]]] or None, optional
        Optional nested dictionary of observation-level reports, usually
        returned by :func:`build_observation_reports`.
    experiment_started_at : Any or None, optional
        Timestamp used to generate deterministic report filenames. If ``None``,
        the current datetime is used.

    Returns
    -------
    Dict[str, str]
        Dictionary containing the generated summary report paths under
        ``"csv"`` and ``"json"``. If observation reports are saved, the
        dictionary also contains ``"observation_csv"``, mapping each metric name
        to its observation-level CSV path.

    Raises
    ------
    ValueError
        If ``reports`` does not contain any report dataframes to save.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _timestamp_suffix(experiment_started_at)
    timestamped_report_name = f"{report_name}_{timestamp}"

    report_frames = []

    for dataset_reports in reports.values():
        for report_df in dataset_reports.values():
            report_frames.append(report_df.copy())

    if not report_frames:
        raise ValueError("No reports are available to save.")
    
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

    csv_path = out_dir / f"{timestamped_report_name}.csv"
    json_path = out_dir / f"{timestamped_report_name}.json"

    csv_df = combined_report.copy()

    for column_name in csv_df.columns:
        csv_df[column_name] = csv_df[column_name].map(_to_csv_cell)

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

    paths = {
        "csv": str(csv_path),
        "json": str(json_path),
    }

    if observation_reports is not None:
        observation_paths = {}

        for metric_name, metric_reports in observation_reports.items():
            metric_frames = []

            for dataset_reports in metric_reports.values():
                for report_df in dataset_reports.values():
                    metric_frames.append(report_df.copy())

            if not metric_frames:
                continue

            combined_metric_report = pd.concat(
                metric_frames,
                ignore_index=True,
                sort=False
            )

            fixed_observation_columns = [
                "dataset_name",
                "model_name",
                "observation",
                "metric_params"
            ]

            xai_method_columns = [
                column
                for column in combined_metric_report.columns
                if column not in fixed_observation_columns
            ]

            combined_metric_report = combined_metric_report[
                fixed_observation_columns + sorted(xai_method_columns)
            ]

            combined_metric_report = combined_metric_report.sort_values(
                by=["dataset_name", "model_name", "observation"],
                ignore_index=True,
            )

            safe_metric_name = _safe_path_name(metric_name)
            metric_dir = out_dir / safe_metric_name
            metric_dir.mkdir(parents=True, exist_ok=True)

            metric_csv_path = (
                metric_dir
                / f"{safe_metric_name}_{timestamped_report_name}.csv"
            )

            metric_csv_df = combined_metric_report.copy()

            for column_name in metric_csv_df.columns:
                metric_csv_df[column_name] = metric_csv_df[column_name].map(_to_csv_cell)

            metric_csv_df.to_csv(metric_csv_path, index=False, encoding="utf-8")

            observation_paths[metric_name] = str(metric_csv_path)

        paths['observation_csv'] = observation_paths

    return paths