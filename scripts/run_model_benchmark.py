from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.logistic_baseline import (
    fit_logistic_baseline,
    predict_fraud_probability as predict_logistic,
)
from ml.models.preprocessing import build_feature_preprocessor
from ml.models.xgboost_model import (
    fit_xgboost_model,
    predict_fraud_probability as predict_xgboost,
)
from ml.split.temporal import temporal_split


BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")
DEFAULT_OUTPUT = Path("reports/model_benchmark")
BOOTSTRAP_SEED = 20261104


def load_modeling_data() -> pd.DataFrame:
    behavioral = pd.read_parquet(BEHAVIORAL_PATH)
    transaction_columns = [
        "TransactionID",
        TARGET_COLUMN,
        *CATEGORICAL_FEATURES,
        *CD_FEATURES,
    ]
    transactions = pd.read_csv(
        TRANSACTION_PATH,
        usecols=transaction_columns,
    )
    modeling = build_modeling_frame(
        behavioral=behavioral,
        transaction_features=transactions,
        labels=transactions,
    )
    if modeling.feature_columns != MODEL_FEATURES:
        raise RuntimeError("Modeling feature contract differs from MODEL_FEATURES.")
    return modeling.data


def build_random_forest(train: pd.DataFrame) -> Pipeline:
    """A fixed, documented tree-ensemble baseline; no validation/test fitting."""
    positives = int(train[TARGET_COLUMN].sum())
    if positives == 0 or positives == len(train):
        raise ValueError("TRAIN must contain both fraud and non-fraud rows.")
    return Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def metric_values(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) != 2:
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
        }
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def evaluate_partition(
    partition: str,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    rows = []
    for model_name, scores in predictions.items():
        values = metric_values(y, scores)
        rows.append(
            {
                "partition": partition,
                "model": model_name,
                "rows": len(frame),
                "fraud_cases": int(y.sum()),
                "fraud_rate": float(y.mean()),
                **values,
            }
        )
    return pd.DataFrame(rows)


def make_test_groups(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Create label-independent test slices; amount cut points come from TRAIN."""
    groups = pd.DataFrame(index=test.index)

    # Quantile bins are based on distinct timestamp ranks so tied timestamps stay
    # in the same period. The labels are never consulted to form these periods.
    time_rank = test["TransactionDT"].rank(method="dense")
    period_codes = pd.qcut(
        time_rank,
        q=4,
        labels=False,
        duplicates="drop",
    )
    groups["time_period"] = period_codes.map(
        lambda value: f"period_{int(value) + 1}" if pd.notna(value) else "unknown"
    )

    amount_edges = np.unique(
        train["TransactionAmt"].quantile([0.25, 0.50, 0.75]).to_numpy()
    )
    amount_edges = np.unique(
        np.concatenate(([-np.inf], amount_edges, [np.inf]))
    )
    amount_labels = [f"amount_band_{i + 1}" for i in range(len(amount_edges) - 1)]
    groups["amount_band"] = pd.cut(
        test["TransactionAmt"],
        bins=amount_edges,
        labels=amount_labels,
        include_lowest=True,
    ).astype("string")

    product = test["ProductCD"].astype("string").fillna("__MISSING__")
    groups["product_cd"] = product

    history = test["card_txn_count_24h"]
    groups["card_history_24h"] = pd.cut(
        history,
        bins=[-np.inf, 0, 2, np.inf],
        labels=["no_prior_24h", "1_to_2_prior_24h", "3plus_prior_24h"],
    ).astype("string")

    return groups


def evaluate_slices(
    test: pd.DataFrame,
    groups: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    y_all = test[TARGET_COLUMN].to_numpy(dtype=int)
    for dimension in groups.columns:
        values = groups[dimension].fillna("__MISSING__")
        for group_value in sorted(values.unique()):
            mask = values.to_numpy() == group_value
            y = y_all[mask]
            for model_name, scores_all in predictions.items():
                scores = scores_all[mask]
                metrics = metric_values(y, scores)
                rows.append(
                    {
                        "dimension": dimension,
                        "segment": str(group_value),
                        "model": model_name,
                        "rows": int(mask.sum()),
                        "fraud_cases": int(y.sum()),
                        "fraud_rate": float(y.mean()) if len(y) else float("nan"),
                        **metrics,
                        "status": (
                            "ok"
                            if len(np.unique(y)) == 2
                            else "insufficient_class_variation"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def stratified_bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired, class-stratified bootstrap CIs for TEST ranking metrics."""
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("TEST must contain both classes for bootstrap intervals.")

    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {
        name: {"pr_auc": [], "roc_auc": []} for name in predictions
    }
    paired_delta: dict[str, dict[str, list[float]]] = {
        name: {"pr_auc": [], "roc_auc": []}
        for name in predictions
        if name != "xgboost"
    }

    for _ in range(replicates):
        sampled = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        rng.shuffle(sampled)
        y_sample = y[sampled]
        for name, scores in predictions.items():
            score_sample = scores[sampled]
            samples[name]["pr_auc"].append(
                float(average_precision_score(y_sample, score_sample))
            )
            samples[name]["roc_auc"].append(
                float(roc_auc_score(y_sample, score_sample))
            )
        for name in paired_delta:
            for metric in ("pr_auc", "roc_auc"):
                paired_delta[name][metric].append(
                    samples["xgboost"][metric][-1] - samples[name][metric][-1]
                )

    rows = []
    for name, metrics in samples.items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "estimate": float(
                        metric_values(y, predictions[name])[metric]
                    ),
                    "ci_95_low": float(np.quantile(values, 0.025)),
                    "ci_95_high": float(np.quantile(values, 0.975)),
                    "replicates": replicates,
                    "method": "paired_stratified_percentile_bootstrap",
                }
            )
    for name, metrics in paired_delta.items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "comparison": f"xgboost_minus_{name}",
                    "metric": f"delta_{metric}",
                    "estimate": float(
                        metric_values(y, predictions["xgboost"])[metric]
                        - metric_values(y, predictions[name])[metric]
                    ),
                    "ci_95_low": float(np.quantile(values, 0.025)),
                    "ci_95_high": float(np.quantile(values, 0.975)),
                    "replicates": replicates,
                    "method": "paired_stratified_percentile_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def fit_and_score(
    name: str,
    fit: Callable[[], object],
    predict: Callable[[object, pd.DataFrame], pd.Series],
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    started = time.perf_counter()
    model = fit()
    fit_seconds = time.perf_counter() - started

    validation_scores = predict(model, validation).to_numpy(dtype=float)
    test_scores = predict(model, test).to_numpy(dtype=float)
    return (
        {
            f"{name}_validation": validation_scores,
            f"{name}_test": test_scores,
        },
        {"model": name, "fit_seconds": fit_seconds},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fair chronological benchmark and TEST robustness report for "
            "Logistic Regression, Random Forest and XGBoost."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100.")

    print("Loading and validating the 28-feature modeling frame...")
    modeling = load_modeling_data()
    split = temporal_split(modeling)

    for name, frame in (
        ("TRAIN", split.train),
        ("VALIDATION", split.validation),
        ("TEST", split.test),
    ):
        print(
            f"{name:10s} rows={len(frame):,} "
            f"fraud={int(frame[TARGET_COLUMN].sum()):,} "
            f"rate={frame[TARGET_COLUMN].mean():.3%}"
        )

    # Every estimator and every learned preprocessing step is fitted on TRAIN.
    # Validation is for comparison; TEST is only reported after fitting.
    factories: list[tuple[str, Callable[[], object], Callable]] = [
        (
            "logistic_regression",
            lambda: fit_logistic_baseline(split.train),
            predict_logistic,
        ),
        (
            "random_forest",
            lambda: (
                lambda pipeline: pipeline.fit(
                    split.train[MODEL_FEATURES],
                    split.train[TARGET_COLUMN],
                )
            )(build_random_forest(split.train)),
            lambda model, frame: pd.Series(
                model.predict_proba(frame[MODEL_FEATURES])[:, 1],
                index=frame.index,
                name="fraud_probability",
            ),
        ),
        (
            "xgboost",
            lambda: fit_xgboost_model(split.train),
            predict_xgboost,
        ),
    ]

    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    timing_rows = []

    for name, fit, predict in factories:
        print(f"Fitting {name} on TRAIN only...")
        scored, timing = fit_and_score(
            name,
            fit,
            predict,
            split.validation,
            split.test,
        )
        validation_predictions[name] = scored[f"{name}_validation"]
        test_predictions[name] = scored[f"{name}_test"]
        timing_rows.append(timing)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_report = evaluate_partition(
        "VALIDATION",
        split.validation,
        validation_predictions,
    )
    test_report = evaluate_partition("TEST", split.test, test_predictions)
    pd.concat([validation_report, test_report], ignore_index=True).to_csv(
        args.output_dir / "model_metrics.csv",
        index=False,
    )

    group_frame = make_test_groups(split.test, split.train)
    robustness = evaluate_slices(split.test, group_frame, test_predictions)
    robustness.to_csv(args.output_dir / "test_segment_metrics.csv", index=False)

    intervals = stratified_bootstrap(
        split.test[TARGET_COLUMN].to_numpy(dtype=int),
        test_predictions,
        replicates=args.bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    intervals.to_csv(args.output_dir / "test_bootstrap_intervals.csv", index=False)

    metadata = {
        "dataset": "IEEE-CIS Fraud Detection",
        "split": "chronological 70/15/15 by distinct TransactionDT values",
        "feature_count": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "train_fraud_cases": int(split.train[TARGET_COLUMN].sum()),
        "validation_fraud_cases": int(split.validation[TARGET_COLUMN].sum()),
        "test_fraud_cases": int(split.test[TARGET_COLUMN].sum()),
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 18,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
        "logistic_regression": "existing fixed class_weight=balanced baseline",
        "xgboost": "existing fixed xgb-001 estimator configuration",
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "method": "paired class-stratified percentile bootstrap",
        },
        "fit_timings": timing_rows,
        "interpretation": (
            "PR-AUC is the primary ranking metric. Brier/log-loss are "
            "descriptive only because class weighting changes score "
            "calibration. No model or threshold is selected using TEST."
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nMODEL METRICS")
    print(pd.concat([validation_report, test_report]).to_string(index=False))
    print("\nTEST BOOTSTRAP INTERVALS")
    print(intervals.to_string(index=False))
    print(f"\nReports written to: {args.output_dir}")
    print("TEST was not used for model selection or threshold tuning.")


if __name__ == "__main__":
    main()
