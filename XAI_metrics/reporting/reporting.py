# XAI_metrics/reporting/reporting.py
import json
import yaml
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from typing import Mapping, Any, Sequence, List

def load_scope_from_yaml(config_path="XAI_metrics/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scope_map = {}
    for m in cfg.get("metrics", []):
        name = m.get("name")
        t = (m.get("params", {}).get("Type", "") or "").strip().lower()
        if name and t in {"local", "global"}:
            scope_map[name] = t
    return scope_map

def _serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value

def _flatten_results(results: Mapping[str, Any]) -> List[tuple[str, Any]]:
    flattened = []
    for metric_name, metric_value in results.items():
        if isinstance(metric_value, Mapping):
            for k, v in metric_value.items():
                flattened.append((f"{metric_name}.{k}", v))
        else:
            flattened.append((metric_name, metric_value))
    return flattened

def _to_numeric_array(value: Any) -> np.ndarray:
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

def metrics_to_dataframe(
    metric_results: Mapping[str, Any],
    observations: Sequence[Any] | None = None,
) -> pd.DataFrame:
    if "results" in metric_results and isinstance(metric_results["results"], Mapping):
        results = metric_results["results"]
    else:
        results = metric_results

    rows = []
    scope_map = load_scope_from_yaml()

    for metric_name, metric_values in _flatten_results(results):
        numeric_values = _to_numeric_array(metric_values)
        scope = scope_map.get(metric_name, "local")

        rows.append(
            {
                "metric": metric_name,
                "scope": scope,
                "row_type": "aggregate",
                "observation": None,
                "value": None,
                "value_raw": _serialize(metric_values),
                "n": int(numeric_values.size),
                "mean": float(np.mean(numeric_values)) if numeric_values.size else np.nan,
                "std": float(np.std(numeric_values)) if numeric_values.size else np.nan,
                "min": float(np.min(numeric_values)) if numeric_values.size else np.nan,
                "max": float(np.max(numeric_values)) if numeric_values.size else np.nan,
            }
        )

        if scope == "local" and numeric_values.size:
            if observations is not None and len(observations) == numeric_values.size:
                obs_labels = list(observations)
            else:
                obs_labels = list(range(numeric_values.size))

            for obs, val in zip(obs_labels, numeric_values):
                rows.append(
                    {
                        "metric": metric_name,
                        "scope": scope,
                        "row_type": "observation",
                        "observation": obs,
                        "value": float(val),
                        "value_raw": None,
                        "n": None,
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
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
            ]
        )
    
    return pd.DataFrame(rows).sort_values(["metric", "row_type"]).reset_index(drop=True)

def _fmt_num(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, int, np.floating, np.integer)):
        if pd.isna(value):
            return "-"
        return f"{float(value):.6f}"
    return str(value)

def metrics_report_markdown(
    metric_results: Mapping[str, Any],
    observations: Sequence[Any] | None = None
) -> str:
    df = metrics_to_dataframe(
        metric_results=metric_results,
        observations=observations
    )
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [f"# Metrics Report", "", "Generated at: {generated_at}", ""]
    if df.empty:
        lines.extend(["No metrics available.", ""])
        return "\n".join(lines)
    
    agg = df[df["row_type"] == "aggregate"].copy()
    local_agg = agg[agg["scope"] == "local"]
    global_agg = agg[agg["scope"] == "global"]
    local_obs = df[((df["row_type"] == "observation") & (df["scope"] == "local"))]

    if not local_agg.empty:
        lines.extend(
            [
                "## Local Metrics (Aggregated)",
                "",
                "| metric | n | mean | std | min | max |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in local_agg.iterrows():
            lines.append(
                f"| {row['metric']} | {int(row['n'])} | {_fmt_num(row['mean'])} | {_fmt_num(row['std'])} | {_fmt_num(row['min'])} | {_fmt_num(row['max'])} |"
            )
        lines.append("")

    if not local_obs.empty:
        lines.extend(
            [
                "## Local Metrics (Per Observation)",
                "",
                "| metric | observation | value |",
                "|---|---|---:|",
            ]
        )
        for _, row in local_obs.iterrows():
            lines.append(
                f"| {row['metric']} | {row['observation']} | {_fmt_num(row['value'])} |"
            )
        lines.append("")

    if not global_agg.empty:
        lines.extend(
            [
                "## Global Metrics",
                "",
                "| metric | n | mean | std | min | max |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in global_agg.iterrows():
            lines.append(
                f"| {row['metric']} | {int(row['n'])} | {_fmt_num(row['mean'])} | {_fmt_num(row['std'])} | {_fmt_num(row['min'])} | {_fmt_num(row['max'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def save_metrics_report(
    metric_results: Mapping[str, Any],
    observations: Sequence[Any] | None = None,
    output_dir: str | Path = ".",
    report_name: str = "metrics_report"
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "results" in metric_results and isinstance(metric_results["results"], Mapping):
        results = metric_results["results"]
    else:
        results = metric_results

    df = metrics_to_dataframe(
         metric_results=metric_results,
         observations=observations
    )

    markdown = metrics_report_markdown(
        metric_results=metric_results,
        observations=observations
    )

    summary_df = df[df["row_type"] == "aggregate"].copy()
    observations_df = df[df["row_type"] == "observation"].copy()
    
    summary_csv_path = out_dir / f"{report_name}_summary.csv"
    observations_csv_path = out_dir / f"{report_name}_observations.csv"
    json_path = out_dir / f"{report_name}.json"
    md_path = out_dir / f"{report_name}.md"

    summary_df.to_csv(summary_csv_path, index=False)
    observations_df.to_csv(observations_csv_path, index=False)

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "results": _serialize(results),
        "summary": summary_df.to_dict(orient="records"),
        "observations": observations_df.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_path.write_text(markdown, encoding="utf-8")

    return {
        "summary_csv": str(summary_csv_path),
        "observations_csv": str(observations_csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }