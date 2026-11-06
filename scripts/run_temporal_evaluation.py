from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.modeling import MODEL_FEATURES, TARGET_COLUMN
from ml.models.xgboost_model import fit_xgboost_model, predict_fraud_probability
from ml.split.temporal import temporal_split
from scripts.run_model_benchmark import (
    BEHAVIORAL_PATH,
    TRANSACTION_PATH,
    build_random_forest,
    load_modeling_data,
    metric_values,
)


DEFAULT_OUTPUT = Path("reports/temporal_robustness")
DEFAULT_FOLDS = 4
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_BLOCK_LENGTH = 50
DEFAULT_BOOTSTRAP_REPLICATES = 500
DEFAULT_SEED = 20261107


def expanding_window_slices(
    timestamps: pd.Series,
    *,
    n_splits: int = DEFAULT_FOLDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding train / next-period test row positions.

    Boundaries are formed from distinct timestamp values, so simultaneous
    transactions are never split between train and test. Every test window is
    strictly later than its training window. The initial training window uses
    the earliest 1 - n_splits * test_fraction share of distinct timestamps.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if n_splits * test_fraction >= 0.5:
        raise ValueError("n_splits * test_fraction must be below 0.5.")
    if timestamps.isna().any():
        raise ValueError("timestamps must not contain missing values.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order.")

    unique = timestamps.drop_duplicates().to_numpy()
    initial_train = int(len(unique) * (1.0 - n_splits * test_fraction))
    if initial_train < 2 or len(unique) - initial_train < n_splits:
        raise ValueError("Not enough distinct timestamps for requested folds.")

    remaining = len(unique) - initial_train
    boundaries = np.linspace(
        initial_train, len(unique), n_splits + 1, dtype=int
    )
    if np.any(np.diff(boundaries) < 1):
        raise ValueError("A temporal test fold would be empty.")

    group_codes = pd.Series(timestamps.to_numpy()).map(
        pd.Series(np.arange(len(unique)), index=unique)
    ).to_numpy()

    folds = []
    for fold in range(n_splits):
        test_start = int(boundaries[fold])
        test_end = int(boundaries[fold + 1])
        train_rows = np.flatnonzero(group_codes < test_start)
        test_rows = np.flatnonzero(
            (group_codes >= test_start) & (group_codes < test_end)
        )
        if not len(train_rows) or not len(test_rows):
            raise ValueError(f"Fold {fold + 1} has an empty partition.")
        if timestamps.iloc[train_rows].max() >= timestamps.iloc[test_rows].min():
            raise AssertionError("Temporal fold boundary is not strictly forward.")
        folds.append((train_rows, test_rows))
    return folds


def moving_time_block_indices(
    timestamps: pd.Series,
    rng: np.random.Generator,
    *,
    block_length: int,
) -> np.ndarray:
    """Sample chronological moving blocks of complete timestamp groups.

    Sampling units are distinct TransactionDT groups, not individual rows.
    This preserves tied timestamps and local temporal ordering within blocks.
    Blocks are sampled without circular wrap-around; the final block is
    shortened to reach the original number of timestamp groups.
    """
    if block_length < 1:
        raise ValueError("block_length must be at least 1.")
    if timestamps.empty or timestamps.isna().any():
        raise ValueError("timestamps must be non-empty and contain no missing values.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order.")

    group_codes, unique = pd.factorize(timestamps, sort=True)
    n_groups = len(unique)
    if block_length > n_groups:
        raise ValueError("block_length cannot exceed distinct timestamp count.")

    row_groups = [np.flatnonzero(group_codes == i) for i in range(n_groups)]
    selected_groups: list[int] = []
    while len(selected_groups) < n_groups:
        start = int(rng.integers(0, n_groups - block_length + 1))
        take = min(block_length, n_groups - len(selected_groups))
        selected_groups.extend(range(start, start + take))

    return np.concatenate([row_groups[group] for group in selected_groups])


def time_block_bootstrap(
    y_true: np.ndarray,
    timestamps: pd.Series,
    predictions: dict[str, np.ndarray],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Paired moving-time-block bootstrap for ranking-metric differences."""
    y = np.asarray(y_true, dtype=int)
    if len(y) != len(timestamps):
        raise ValueError("y_true and timestamps must have equal lengths.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Bootstrap input must contain both classes.")
    if replicates < 100:
        raise ValueError("replicates must be at least 100.")
    for name, scores in predictions.items():
        if len(scores) != len(y):
            raise ValueError(f"Prediction length mismatch for {name}.")
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite predictions for {name}.")

    if "xgboost" not in predictions or "random_forest" not in predictions:
        raise ValueError("Predictions must include xgboost and random_forest.")

    rng = np.random.default_rng(seed)
    deltas = {"average_precision": [], "roc_auc": []}
    for _ in range(replicates):
        sampled = moving_time_block_indices(
            timestamps, rng, block_length=block_length
        )
        labels = y[sampled]
        if len(np.unique(labels)) < 2:
            continue
        xgb_metrics = metric_values(labels, predictions["xgboost"][sampled])
        rf_metrics = metric_values(labels, predictions["random_forest"][sampled])
        deltas["average_precision"].append(
            xgb_metrics["pr_auc"] - rf_metrics["pr_auc"]
        )
        deltas["roc_auc"].append(xgb_metrics["roc_auc"] - rf_metrics["roc_auc"])

    rows = []
    for metric, values in deltas.items():
        if len(values) < int(replicates * 0.90):
            raise RuntimeError(
                "Too many single-class time-block replicates; increase "
                "block_length or inspect the temporal label distribution."
            )
        rows.append(
            {
                "comparison": "xgboost_minus_random_forest",
                "metric": metric,
                "ci_95_low": float(np.quantile(values, 0.025)),
                "ci_95_high": float(np.quantile(values, 0.975)),
                "valid_replicates": len(values),
                "requested_replicates": replicates,
                "block_length_timestamp_groups": block_length,
                "method": "paired_moving_time_block_percentile_bootstrap",
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def evaluate_walk_forward(
    data: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_FOLDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fixed XGBoost and Random Forest candidates on each past window."""
    if not data["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Modeling data must be sorted by TransactionDT.")

    folds = expanding_window_slices(
        data["TransactionDT"],
        n_splits=n_splits,
        test_fraction=test_fraction,
    )
    rows = []
    for fold_number, (train_positions, test_positions) in enumerate(folds, 1):
        train = data.iloc[train_positions]
        test = data.iloc[test_positions]
        y_train = train[TARGET_COLUMN]
        y_test = test[TARGET_COLUMN]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(
                f"Fold {fold_number} must contain both classes in train and test."
            )

        models = {
            "xgboost": lambda: fit_xgboost_model(train),
            "random_forest": lambda: build_random_forest(train).fit(
                train[MODEL_FEATURES], y_train
            ),
        }
        for model_name, factory in models.items():
            started = time.perf_counter()
            model = factory()
            fit_seconds = time.perf_counter() - started
            if model_name == "xgboost":
                scores = predict_fraud_probability(model, test).to_numpy(dtype=float)
            else:
                scores = model.predict_proba(test[MODEL_FEATURES])[:, 1]

            metrics = metric_values(y_test.to_numpy(dtype=int), scores)
            rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_fraud_cases": int(y_train.sum()),
                    "test_fraud_cases": int(y_test.sum()),
                    "train_end_transaction_dt": int(train["TransactionDT"].max()),
                    "test_start_transaction_dt": int(test["TransactionDT"].min()),
                    "test_end_transaction_dt": int(test["TransactionDT"].max()),
                    **metrics,
                    "fit_seconds": fit_seconds,
                }
            )

    report = pd.DataFrame(rows)
    summary = (
        report.groupby("model", as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_average_precision=("pr_auc", "mean"),
            std_average_precision=("pr_auc", "std"),
            min_average_precision=("pr_auc", "min"),
            max_average_precision=("pr_auc", "max"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_brier=("brier", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
    )
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expanding-window temporal evaluation and time-block bootstrap."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("Loading the existing point-in-time 28-feature modeling frame...")
    data = load_modeling_data()
    if not data["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Input data is not sorted by TransactionDT.")

    print("Running expanding-window walk-forward model evaluation...")
    fold_report, summary = evaluate_walk_forward(
        data, n_splits=args.folds, test_fraction=args.test_fraction
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_report.to_csv(args.output_dir / "walk_forward_folds.csv", index=False)
    summary.to_csv(args.output_dir / "walk_forward_summary.csv", index=False)

    print("Fitting the locked TRAIN-only benchmark models for block-bootstrap comparison...")
    split = temporal_split(data)
    xgb = fit_xgboost_model(split.train)
    rf = build_random_forest(split.train).fit(
        split.train[MODEL_FEATURES], split.train[TARGET_COLUMN]
    )
    predictions = {
        "xgboost": predict_fraud_probability(xgb, split.test).to_numpy(dtype=float),
        "random_forest": rf.predict_proba(split.test[MODEL_FEATURES])[:, 1],
    }
    block_ci = time_block_bootstrap(
        split.test[TARGET_COLUMN].to_numpy(dtype=int),
        split.test["TransactionDT"].reset_index(drop=True),
        predictions,
        replicates=args.bootstrap_replicates,
        block_length=args.block_length,
        seed=args.seed,
    )
    block_ci.to_csv(args.output_dir / "test_time_block_bootstrap.csv", index=False)

    metadata = {
        "dataset": "IEEE-CIS Fraud Detection",
        "walk_forward": {
            "n_splits": args.folds,
            "test_fraction_of_distinct_timestamps_per_fold": args.test_fraction,
            "initial_train_fraction_of_distinct_timestamps": 1 - args.folds * args.test_fraction,
            "strategy": "expanding training window; each test window is strictly later",
            "timestamp_ties_preserved": True,
        },
        "time_block_bootstrap": {
            "replicates_requested": args.bootstrap_replicates,
            "block_length_timestamp_groups": args.block_length,
            "seed": args.seed,
            "method": "paired moving blocks of complete distinct TransactionDT groups",
        },
        "interpretation": (
            "Walk-forward folds are repeated historical simulations, not "
            "independent datasets. Block-bootstrap intervals quantify "
            "uncertainty conditional on this holdout and chosen block length; "
            "they do not establish future or live payment performance."
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Walk-forward reports: {args.output_dir}")
    print(fold_report.to_string(index=False))
    print("\nWalk-forward summary")
    print(summary.to_string(index=False))
    print("\nTime-block bootstrap")
    print(block_ci.to_string(index=False))


if __name__ == "__main__":
    main()
