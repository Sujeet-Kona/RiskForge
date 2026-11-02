from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.calibration import (
    CALIBRATION_METHODS,
    DEFAULT_CALIBRATION_FOLDS,
    apply_calibrator,
    calibration_metrics_to_dict,
    cross_validate_calibration,
    evaluate_calibration,
    fit_calibrator,
    reliability_table,
)
from ml.models.evaluation import evaluate_predictions, metrics_to_dict
from ml.models.xgboost_model import fit_xgboost_model, predict_fraud_probability
from ml.split.temporal import temporal_split


BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")

TRANSACTION_COLUMNS = [
    "TransactionID",
    TARGET_COLUMN,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]

RELIABILITY_BINS = 10

SELECTION_RULE = (
    "Selected method = lowest out-of-fold VALIDATION Brier score. "
    "Pre-registered before any TEST evaluation."
)


def print_header(step: str, title: str) -> None:
    print(f"\n[{step}] {title}...")


def print_calibration_metrics(name: str, metrics) -> None:
    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    for key, value in calibration_metrics_to_dict(metrics).items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.6f}")
        else:
            print(f"{key:32s}: {value:,}")


def print_ranking_metrics(name: str, metrics) -> None:
    print("\n" + "-" * 80)
    print(f"{name} — LOCKED EVALUATION DEFINITIONS (threshold 0.5)")
    print("-" * 80)

    for key, value in metrics_to_dict(metrics).items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.6f}")
        else:
            print(f"{key:32s}: {value:,}")


def print_reliability_table(name: str, y_true, probability) -> None:
    print("\n" + "-" * 80)
    print(f"{name} — RELIABILITY TABLE ({RELIABILITY_BINS} equal-width bins)")
    print("-" * 80)
    print(
        f"{'bin':>14s}  {'mean_pred':>10s}  "
        f"{'observed':>10s}  {'count':>8s}"
    )

    table = reliability_table(y_true, probability, n_bins=RELIABILITY_BINS)

    for entry in table:
        print(
            f"[{entry.lower:.1f},{entry.upper:.1f})".rjust(15)
            + f"  {entry.mean_predicted_probability:10.6f}"
            + f"  {entry.observed_fraud_rate:10.6f}"
            + f"  {entry.count:8,}"
        )

    scored = sum(entry.count for entry in table)

    if scored != len(probability):
        raise AssertionError(
            f"Reliability table covered {scored} of {len(probability)} rows."
        )


def main() -> None:
    started = time.perf_counter()

    print("=" * 80)
    print("RISKFORGE — PHASE 3 PROBABILITY CALIBRATION")
    print("=" * 80)
    print(f"\n{SELECTION_RULE}")
    print(
        "Calibrators are fitted on VALIDATION scores only. "
        "TEST is scored once, after selection."
    )

    print_header("1/9", "Loading behavioral artifact")

    behavioral = pd.read_parquet(BEHAVIORAL_PATH)

    print(f"Behavioral rows: {len(behavioral):,}")

    print_header("2/9", "Loading transaction features and fraud labels")

    transactions = pd.read_csv(TRANSACTION_PATH, usecols=TRANSACTION_COLUMNS)

    print(f"Transaction rows: {len(transactions):,}")
    print(f"Fraud cases: {int(transactions[TARGET_COLUMN].sum()):,}")

    print_header("3/9", "Building modeling frame")

    modeling = build_modeling_frame(
        behavioral=behavioral,
        transaction_features=transactions,
        labels=transactions,
    )

    print(f"Modeling rows: {len(modeling.data):,}")
    print(f"Model features: {len(modeling.feature_columns)}")

    print_header("4/9", "Applying the locked chronological split")

    split = temporal_split(modeling.data)

    for name, frame in [
        ("TRAIN", split.train),
        ("VALIDATION", split.validation),
        ("TEST", split.test),
    ]:
        print(
            f"{name:10s} rows={len(frame):8,} "
            f"fraud={int(frame[TARGET_COLUMN].sum()):6,} "
            f"fraud_rate={frame[TARGET_COLUMN].mean():.3%} "
            f"dt=[{frame['TransactionDT'].min():,} "
            f"-> {frame['TransactionDT'].max():,}]"
        )

    print_header("5/9", "Fitting the locked XGBoost model (unmodified)")

    fit_started = time.perf_counter()

    model = fit_xgboost_model(split.train)

    print(f"scale_pos_weight: {model.scale_pos_weight:.6f}")
    print(f"Fit seconds: {time.perf_counter() - fit_started:.2f}")

    print_header("6/9", "Generating raw fraud scores for all three partitions")

    raw = {
        name: predict_fraud_probability(model, frame)
        for name, frame in [
            ("TRAIN", split.train),
            ("VALIDATION", split.validation),
            ("TEST", split.test),
        ]
    }

    labels = {
        name: frame[TARGET_COLUMN]
        for name, frame in [
            ("TRAIN", split.train),
            ("VALIDATION", split.validation),
            ("TEST", split.test),
        ]
    }

    for name in ("TRAIN", "VALIDATION", "TEST"):
        line = (
            f"{name:10s} raw mean={raw[name].mean():.6f} "
            f"range=[{raw[name].min():.6f} -> {raw[name].max():.6f}]"
        )

        # The TEST event rate is withheld until after method selection so the
        # log itself demonstrates that TEST cannot inform the choice.
        if name != "TEST":
            line += f" observed={labels[name].mean():.6f}"

        print(line)

    print("\n" + "=" * 80)
    print("STEP A — IS THE RAW SCORE CALIBRATED? (no fitting involved)")
    print("=" * 80)
    print("TEST is deliberately excluded from every step before selection.")

    for name in ("TRAIN", "VALIDATION"):
        print_calibration_metrics(
            f"RAW XGBOOST — {name}",
            evaluate_calibration(
                labels[name],
                raw[name],
                n_bins=RELIABILITY_BINS,
            ),
        )

    print_reliability_table("RAW XGBOOST — VALIDATION", labels["VALIDATION"], raw["VALIDATION"])

    print("\n" + "=" * 80)
    print("STEP B — WHY THE CALIBRATOR IS NOT FIT ON TRAIN")
    print("=" * 80)
    print(
        "TRAIN scores are in-sample for the model that produced them, so a"
        " mapping learned there inherits that over-confidence. Compare TRAIN"
        " and VALIDATION raw Brier/ECE above."
    )

    print("\n" + "=" * 80)
    print("STEP C — METHOD SELECTION ON VALIDATION, OUT-OF-FOLD")
    print("=" * 80)
    print(
        f"Temporal {DEFAULT_CALIBRATION_FOLDS}-fold CV within VALIDATION. "
        "TEST is not read here."
    )

    selection: dict[str, object] = {}

    for method in CALIBRATION_METHODS:
        cv_started = time.perf_counter()

        out_of_fold = cross_validate_calibration(
            method,
            raw["VALIDATION"],
            labels["VALIDATION"],
            split.validation["TransactionDT"],
            DEFAULT_CALIBRATION_FOLDS,
        )

        metrics = evaluate_calibration(
            labels["VALIDATION"],
            out_of_fold,
            n_bins=RELIABILITY_BINS,
        )

        selection[method] = {
            "metrics": metrics,
            "out_of_fold": out_of_fold,
            "seconds": time.perf_counter() - cv_started,
        }

        print_calibration_metrics(
            f"OUT-OF-FOLD {method.upper()} — VALIDATION",
            metrics,
        )
        print(f"cv_seconds: {selection[method]['seconds']:.2f}")

    selected = min(
        CALIBRATION_METHODS,
        key=lambda method: selection[method]["metrics"].brier_score,
    )

    print("\n" + "=" * 80)
    print(f"SELECTED METHOD: {selected.upper()}")
    print("=" * 80)

    for method in CALIBRATION_METHODS:
        metrics = selection[method]["metrics"]

        print(
            f"{method:10s} oof_brier={metrics.brier_score:.6f} "
            f"oof_log_loss={metrics.log_loss:.6f} "
            f"oof_ece={metrics.expected_calibration_error:.6f} "
            f"oof_pr_auc={metrics.pr_auc:.6f} "
            f"oof_roc_auc={metrics.roc_auc:.6f}"
        )

    print(f"\n{SELECTION_RULE}")

    raw_validation_ranking = evaluate_predictions(
        labels["VALIDATION"],
        raw["VALIDATION"],
    )
    print_ranking_metrics("RAW XGBOOST — VALIDATION", raw_validation_ranking)

    oof_ranking = evaluate_predictions(
        labels["VALIDATION"],
        selection[selected]["out_of_fold"],
    )
    print_ranking_metrics(
        f"OUT-OF-FOLD {selected.upper()} — VALIDATION",
        oof_ranking,
    )

    print_reliability_table(
        f"OUT-OF-FOLD {selected.upper()} — VALIDATION",
        labels["VALIDATION"],
        selection[selected]["out_of_fold"],
    )

    print("\n" + "=" * 80)
    print(f"STEP D — REFIT {selected.upper()} ON ALL OF VALIDATION")
    print("=" * 80)

    calibrator = fit_calibrator(
        selected,
        raw["VALIDATION"],
        labels["VALIDATION"],
    )

    print(f"Fitted calibrator: {calibrator.method}")
    print("TEST labels have not been read by any fitting or selection step.")

    in_sample_validation = apply_calibrator(calibrator, raw["VALIDATION"])

    print_calibration_metrics(
        f"IN-SAMPLE {selected.upper()} — VALIDATION (refit, optimistic)",
        evaluate_calibration(
            labels["VALIDATION"],
            in_sample_validation,
            n_bins=RELIABILITY_BINS,
        ),
    )

    print("\n" + "=" * 80)
    print("STEP E — SINGLE TEST EVALUATION")
    print("=" * 80)

    calibrated_test = apply_calibrator(calibrator, raw["TEST"])

    raw_test_ranking = evaluate_predictions(labels["TEST"], raw["TEST"])
    calibrated_test_ranking = evaluate_predictions(
        labels["TEST"],
        calibrated_test,
    )

    print_calibration_metrics(
        "RAW XGBOOST — TEST",
        evaluate_calibration(
            labels["TEST"],
            raw["TEST"],
            n_bins=RELIABILITY_BINS,
        ),
    )

    print_calibration_metrics(
        f"CALIBRATED {selected.upper()} — TEST",
        evaluate_calibration(
            labels["TEST"],
            calibrated_test,
            n_bins=RELIABILITY_BINS,
        ),
    )

    print_ranking_metrics("RAW XGBOOST — TEST", raw_test_ranking)

    print_ranking_metrics(
        f"CALIBRATED {selected.upper()} — TEST",
        calibrated_test_ranking,
    )

    print_reliability_table("RAW XGBOOST — TEST", labels["TEST"], raw["TEST"])

    print_reliability_table(
        f"CALIBRATED {selected.upper()} — TEST",
        labels["TEST"],
        calibrated_test,
    )

    print("\n" + "=" * 80)
    print("LOCKED BASELINE COMPARISON")
    print("=" * 80)
    print("Recorded TEST pr_auc : 0.449254")
    print("Recorded TEST roc_auc: 0.865318")
    print(f"Raw TEST pr_auc now  : {raw_test_ranking.pr_auc:.6f}")
    print(f"Raw TEST roc_auc now : {raw_test_ranking.roc_auc:.6f}")
    print("Calibration must not move these raw values.")

    print(f"\nTotal seconds: {time.perf_counter() - started:.2f}")

    print("\n" + "=" * 80)
    print("PHASE 3 CALIBRATION EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
