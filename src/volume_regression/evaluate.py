"""Evaluate direct prostate volume regression predictions.

Purpose:
    Read ``test_predictions.csv`` from ``train.py`` and write summary metrics as
    JSON. This script can also evaluate any prediction CSV with the required
    columns.

Required input columns:
    - ``volume_ml_true``: reference whole-gland volume in mL.
    - ``volume_ml_pred``: model-predicted whole-gland volume in mL.

Optional input columns:
    - ``median_volume_ml_pred``: constant median-volume baseline prediction.
      When present, this script reports the same metrics for the baseline.

Output metrics:
    MAE, median absolute error, RMSE, MAPE, Pearson and Spearman correlation,
    Bland-Altman bias and limits of agreement, and mean true/predicted volume.
    Undefined correlations are written as JSON ``null``.

Example:
    uv run python src/volume_regression/evaluate.py \
      --predictions work_dir_volume_regression_resnet/resnet18_pooled2/test_predictions.csv \
      --output work_dir_volume_regression_resnet/resnet18_pooled2/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import json_safe, regression_metrics, resolve_path


def evaluate_predictions(predictions: str | Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(predictions)
    required = {"volume_ml_true", "volume_ml_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    metrics = {
        "direct_regression": regression_metrics(df["volume_ml_true"], df["volume_ml_pred"]),
    }
    if "median_volume_ml_pred" in df.columns:
        metrics["median_volume_baseline"] = regression_metrics(
            df["volume_ml_true"],
            df["median_volume_ml_pred"],
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prostate volume regression predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="results/volume_regression/resnet18_pooled2/metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = resolve_path(args.predictions)
    output = resolve_path(args.output)
    metrics = evaluate_predictions(predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(metrics), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
