"""Write scalar summaries and aggregate statistics for experiment sweeps."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from saealib.acquisition.base import direction_to_minimize_sign
from saealib.exceptions import ValidationError
from saealib.experiment._config import INDICATOR_SPECS
from saealib.experiment._trial import RunResult
from saealib.utils import gd, gd_plus, hypervolume, igd, igd_plus, spacing, spread

SUMMARY_FILE = "summary.csv"
AGGREGATE_FILE = "aggregate.json"
_INDICATORS = {
    "gd": gd,
    "gd_plus": gd_plus,
    "igd": igd,
    "igd_plus": igd_plus,
    "spread": spread,
    "spacing": spacing,
    "hypervolume": hypervolume,
}
_MAX_INDICATORS = {"hypervolume"}


def write_summary(
    results: Sequence[RunResult],
    output_dir: str | Path,
    indicator: Mapping[str, Any] | None = None,
) -> Path:
    """Write one scalar summary row per trial.

    Parameters
    ----------
    results : sequence of RunResult
        Trial outcomes in sweep order.
    output_dir : str or Path
        Experiment root receiving ``summary.csv``.
    indicator : mapping or None, optional
        Multi-objective indicator specification.

    Returns
    -------
    Path
        The written summary path.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(result, indicator) for result in results]
    metric_names = {row["metric_name"] for row in rows if row["metric_name"]}
    metric_name = next(iter(metric_names)) if len(metric_names) == 1 else None
    fields = ["problem", "algorithm", "seed", "fe", "gen", "wall_time"]
    if metric_name is not None and all(row["metric_name"] for row in rows):
        fields.append(metric_name)
        for row in rows:
            row[metric_name] = row["metric"]
    with (destination / SUMMARY_FILE).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination / SUMMARY_FILE


def write_aggregate(
    summary_path: str | Path,
    results: Sequence[RunResult],
    indicator: Mapping[str, Any] | None = None,
) -> Path | None:
    """Write seed-wise aggregate statistics for a summary CSV.

    Parameters
    ----------
    summary_path : str or Path
        Summary CSV written by :func:`write_summary`.
    results : sequence of RunResult
        Trial outcomes corresponding to the summary rows.
    indicator : mapping or None, optional
        Multi-objective indicator specification used by the summary.

    Returns
    -------
    Path or None
        The aggregate path, or ``None`` when no scalar metric exists.
    """
    path = Path(summary_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metric_columns = [
        name
        for name in (rows[0].keys() if rows else ())
        if name not in {"problem", "algorithm", "seed", "fe", "gen", "wall_time"}
    ]
    if not metric_columns or len(metric_columns) != 1:
        return None
    metric = metric_columns[0]
    if len(rows) != len(results):
        raise ValidationError("summary rows and results must have matching lengths")
    groups: dict[tuple[str, str], tuple[str, list[float]]] = {}
    for row, result in zip(rows, results, strict=True):
        key = (row["problem"], row["algorithm"])
        orientation = _orientation(result, metric, indicator)
        if key in groups and groups[key][0] != orientation:
            raise ValidationError(
                f"Inconsistent orientation in aggregate group {key!r}."
            )
        groups.setdefault(key, (orientation, []))[1].append(float(row[metric]))
    aggregate = {
        "metric": metric,
        "groups": [
            {
                "problem": problem,
                "algorithm": algorithm,
                "orientation": orientation,
                "n_trials": len(values),
                "median": float(np.median(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                "best": float(min(values) if orientation == "min" else max(values)),
                "worst": float(max(values) if orientation == "min" else min(values)),
            }
            for (problem, algorithm), (orientation, values) in groups.items()
        ],
    }
    destination = path.with_name(AGGREGATE_FILE)
    destination.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return destination


def _summary_row(
    result: RunResult, indicator: Mapping[str, Any] | None
) -> dict[str, str]:
    row = {
        "problem": str(result.labels.get("problem", "")),
        "algorithm": str(result.labels.get("algorithm", "")),
        "seed": str(result.labels.get("seed", result.seed)),
        "fe": str(result.fe),
        "gen": str(result.gen),
        "wall_time": str(result.wall_time),
        "metric_name": "",
        "metric": "",
    }
    if len(result.direction) == 1:
        row["metric_name"] = "best_f"
        row["metric"] = str(float(np.asarray(result.best_f).reshape(-1)[0]))
    elif indicator is not None:
        name = str(indicator["type"])
        row["metric_name"] = name
        row["metric"] = str(_multi_objective_value(result, indicator))
    return row


def _multi_objective_value(result: RunResult, indicator: Mapping[str, Any]) -> float:
    name = str(indicator["type"])
    params = indicator.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError("indicator params must be a mapping")
    sign = np.asarray(direction_to_minimize_sign(result.direction), dtype=float)
    front = np.asarray(result.best_f, dtype=float) * sign
    indicator_function = _INDICATORS[name]
    function_params = dict(params)
    if INDICATOR_SPECS[name] == "reference_point":
        function_params["reference_point"] = (
            np.asarray(function_params["reference_point"], dtype=float) * sign
        )
    elif INDICATOR_SPECS[name] == "reference_front":
        function_params["reference_front"] = (
            np.asarray(function_params["reference_front"], dtype=float) * sign
        )
    return float(indicator_function(front, **function_params))


def _orientation(
    result: RunResult, metric: str, indicator: Mapping[str, Any] | None
) -> str:
    if metric == "best_f" and len(result.direction) == 1:
        return "max" if result.direction[0] > 0 else "min"
    return "max" if metric in _MAX_INDICATORS else "min"
