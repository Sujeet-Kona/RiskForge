from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.calibration import (
    DEFAULT_CALIBRATION_FOLDS,
    apply_calibrator,
    cross_validate_calibration,
    evaluate_calibration,
    fit_calibrator,
)
from ml.models.evaluation import evaluate_predictions
from ml.models.xgboost_model import fit_xgboost_model, predict_fraud_probability
from ml.split.temporal import temporal_split
from policy.costs import ASSUMPTION_NOTICE, EXPERIMENTAL_COST_ASSUMPTIONS
from policy.evaluation import (
    DEFAULT_POLICY_VERSION,
    candidate_thresholds,
    degenerate_baseline_costs,
    evaluate_policy,
    policy_outcome_to_dict,
    search_thresholds,
)


BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")

TRANSACTION_COLUMNS = [
    "TransactionID",
    TARGET_COLUMN,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]

LOCKED_CALIBRATION_METHOD = "isotonic"

# Locked Phase 3 results. This script must reproduce them exactly; any drift
# means the model, the split or the calibrator was touched, which Phase 4
# forbids.
LOCKED_RAW_TEST_PR_AUC = 0.449254
LOCKED_RAW_TEST_ROC_AUC = 0.865318
LOCKED_CALIBRATED_TEST_PR_AUC = 0.438919
LOCKED_CALIBRATED_TEST_ROC_AUC = 0.864850

# Locked Phase 4 primary cost model. ASSUMPTION, not measurement.
PRIMARY_COSTS = EXPERIMENTAL_COST_ASSUMPTIONS

# Report-only sensitivity levels for the review-effectiveness assumption.
REVIEW_EFFECTIVENESS_LEVELS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)

# Report-only operational review-volume caps. Not part of the primary objective.
REVIEW_CAP_LEVELS = (0.05, 0.10)

RUNNER_UP_COUNT = 5

METHOD_NOTE = (
    "Threshold selection uses forward-chaining out-of-fold VALIDATION "
    "probabilities: the first chronological fold is a warm-up period and is "
    "excluded; every later fold is calibrated only from strictly earlier "
    "folds. TEST probabilities still come from the single isotonic calibrator "
    "refitted on all VALIDATION — the frozen deployment mapping. The two "
    "calibration procedures are therefore not identical, and the selected "
    "thresholds are not a perfectly unbiased estimate of deployed policy cost. "
    "The reported VALIDATION cost is measured only on the forward-scored rows."
)


def print_header(step: str, title: str) -> None:
    print(f"\n[{step}] {title}...")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_outcome(name: str, outcome) -> None:
    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    for key, value in policy_outcome_to_dict(outcome).items():
        if isinstance(value, float):
            print(f"{key:34s}: {value:.6f}")
        else:
            print(f"{key:34s}: {value:,}" if isinstance(value, int)
                  else f"{key:34s}: {value}")


def print_model_metrics(name: str, metrics) -> None:
    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    for key in ("pr_auc", "roc_auc", "brier_score", "log_loss",
                "expected_calibration_error"):
        if hasattr(metrics, key):
            print(f"{key:34s}: {getattr(metrics, key):.6f}")


def main() -> None:
    started = time.perf_counter()

    print_section("RISKFORGE — PHASE 4 COST-SENSITIVE DECISION ENGINE")
    print(ASSUMPTION_NOTICE)
    print(
        "\nMODEL MEASUREMENT vs BUSINESS ASSUMPTION: every ranking and "
        "calibration figure below is measured from IEEE-CIS. Every cost figure "
        "is a synthetic project assumption. For this project experiment we "
        "assume the costs printed next; we make no claim about real payment "
        "networks."
    )
    print(f"\nC_FN (missed fraud, fraud ALLOWED)   : "
          f"{PRIMARY_COSTS.false_negative_cost:.2f}   ASSUMPTION")
    print(f"C_FP (false decline, legit BLOCKED)  : "
          f"{PRIMARY_COSTS.false_positive_cost:.2f}    ASSUMPTION")
    print(f"C_REVIEW (manual review)             : "
          f"{PRIMARY_COSTS.review_cost:.2f}    ASSUMPTION")
    print(f"e (review fraud capture rate)        : "
          f"{PRIMARY_COSTS.review_fraud_capture_rate:.2f}    ASSUMPTION")
    print(
        "\nPrimary objective: minimum expected VALIDATION cost, no review-rate "
        "constraint. TEST labels are not read until the final step."
    )

    print_header("1/8", "Loading the locked data artifacts")

    behavioral = pd.read_parquet(BEHAVIORAL_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH, usecols=TRANSACTION_COLUMNS)
    modeling = build_modeling_frame(
        behavioral=behavioral,
        transaction_features=transactions,
        labels=transactions,
    )
    split = temporal_split(modeling.data)

    print(f"Behavioral rows : {len(behavioral):,}")
    print(f"Transaction rows: {len(transactions):,}")
    print(f"Modeling rows   : {len(modeling.data):,}")
    print(f"Model features  : {len(modeling.feature_columns)} (locked contract)")

    for name, frame in [
        ("TRAIN", split.train),
        ("VALIDATION", split.validation),
        ("TEST", split.test),
    ]:
        line = (
            f"{name:10s} rows={len(frame):8,} "
            f"dt=[{frame['TransactionDT'].min():,} -> {frame['TransactionDT'].max():,}]"
        )
        if name != "TEST":
            line += (
                f" fraud={int(frame[TARGET_COLUMN].sum()):6,} "
                f"rate={frame[TARGET_COLUMN].mean():.3%}"
            )
        print(line)

    print("TEST fraud count and rate are withheld until STEP 8.")

    print_header("2/8", "Fitting the locked XGBoost model (unmodified)")

    fit_started = time.perf_counter()
    model = fit_xgboost_model(split.train)

    print(f"scale_pos_weight: {model.scale_pos_weight:.6f} (locked: 27.410063)")
    print(f"Fit seconds     : {time.perf_counter() - fit_started:.2f}")

    print_header("3/8", "Generating raw fraud scores")

    validation_labels = split.validation[TARGET_COLUMN]
    raw_validation = predict_fraud_probability(model, split.validation)
    raw_test = predict_fraud_probability(model, split.test)

    print(
        f"VALIDATION raw mean={raw_validation.mean():.6f} "
        f"range=[{raw_validation.min():.6f} -> {raw_validation.max():.6f}] "
        f"observed={validation_labels.mean():.6f}"
    )
    print(
        f"TEST       raw mean={raw_test.mean():.6f} "
        f"range=[{raw_test.min():.6f} -> {raw_test.max():.6f}] "
        "observed=WITHHELD"
    )

    print_header(
        "4/8",
        f"Out-of-fold {LOCKED_CALIBRATION_METHOD} calibration of VALIDATION "
        "(threshold-selection probabilities)",
    )

    oof_started = time.perf_counter()
    oof_validation = cross_validate_calibration(
        LOCKED_CALIBRATION_METHOD,
        raw_validation,
        validation_labels,
        split.validation["TransactionDT"],
        DEFAULT_CALIBRATION_FOLDS,
    )
    oof_seconds = time.perf_counter() - oof_started

    scored_rows = oof_validation.notna()
    scored_validation_labels = validation_labels.loc[scored_rows]
    scored_oof_validation = oof_validation.loc[scored_rows]

    oof_metrics = evaluate_calibration(
        scored_validation_labels, scored_oof_validation
    )

    print(
        f"Folds      : {DEFAULT_CALIBRATION_FOLDS} "
        "(forward-chaining; first fold warm-up)"
    )
    print(f"Scored rows: {len(scored_oof_validation):,} / {len(oof_validation):,}")
    print(f"Seconds    : {oof_seconds:.2f}")
    print_model_metrics(
        f"OUT-OF-FOLD {LOCKED_CALIBRATION_METHOD.upper()} — VALIDATION",
        oof_metrics,
    )

    for label, series in [
        ("below 0.01", (scored_oof_validation < 0.01).mean()),
        ("below 0.05", (scored_oof_validation < 0.05).mean()),
        ("below 0.10", (scored_oof_validation < 0.10).mean()),
        ("at or above 0.50", (scored_oof_validation >= 0.50).mean()),
    ]:
        print(f"OOF share {label:16s}: {series:.4%}")

    print_section("STEP 5 — THRESHOLD SEARCH ON VALIDATION ONLY")
    print(
        "Search space: candidate thresholds are the empirical percentiles of "
        "the OOF VALIDATION scores plus a fixed interpretable grid, collapsed "
        "by the number of rows each candidate places below it. Every feasible "
        "(T_REVIEW, T_BLOCK) pair with T_REVIEW < T_BLOCK is evaluated exactly "
        "— the grid is searched, not sampled."
    )

    candidates = candidate_thresholds(scored_oof_validation)

    print(f"Candidate threshold values: {candidates.size}")
    print(f"Candidate min / max       : "
          f"{candidates.min():.6f} / {candidates.max():.6f}")

    search_started = time.perf_counter()
    search = search_thresholds(
        scored_oof_validation,
        scored_validation_labels,
        PRIMARY_COSTS,
        candidates=candidates,
        policy_version=DEFAULT_POLICY_VERSION,
        runner_up_count=RUNNER_UP_COUNT,
    )
    search_seconds = time.perf_counter() - search_started

    print(f"Feasible pairs evaluated  : {search.feasible_pairs_evaluated:,}")
    print(f"Review-rate constraint    : "
          f"{'none' if search.max_review_rate is None else search.max_review_rate}")
    print(f"Search seconds            : {search_seconds:.2f}")

    baselines = degenerate_baseline_costs(scored_validation_labels, PRIMARY_COSTS)

    print("\nDegenerate single-action floors on VALIDATION (ASSUMPTION costs):")
    for name, cost in baselines.items():
        print(f"  {name:14s}: {cost:,.2f} total  "
              f"({cost / len(scored_validation_labels):.4f} per transaction)")

    frozen_config = search.config
    frozen_review = frozen_config.review_threshold
    frozen_block = frozen_config.block_threshold

    print_section(f"SELECTED POLICY — {frozen_config.policy_version} (FROZEN)")
    print(f"T_REVIEW = {frozen_review:.6f}")
    print(f"T_BLOCK  = {frozen_block:.6f}")
    print(f"policy_version = {frozen_config.policy_version}")
    print(f"C_FN={frozen_config.costs.false_negative_cost}, "
          f"C_FP={frozen_config.costs.false_positive_cost}, "
          f"C_REVIEW={frozen_config.costs.review_cost}, "
          f"e={frozen_config.costs.review_fraud_capture_rate}")
    print(
        "\nWhy it won: it is the global minimum of expected VALIDATION cost "
        "over the entire feasible threshold grid under the stated ASSUMPTION "
        "costs, with no review-rate constraint. Ties are broken deterministically "
        "by fewest reviews, then lowest T_REVIEW, then lowest T_BLOCK."
    )
    print(f"\nFast-path cost check: {search.winner_fast_path_cost:.6f} vs "
          f"authoritative {search.outcome.total_expected_cost:.6f}")

    if abs(search.winner_fast_path_cost
           - search.outcome.total_expected_cost) > 1e-6:
        raise AssertionError("Threshold-search fast path disagrees with the "
                             "authoritative cost definition.")

    print_section("COMPETING POLICIES (next-best on VALIDATION cost)")
    print(f"{'rank':>4s}  {'T_REVIEW':>9s}  {'T_BLOCK':>9s}  {'cost':>14s}  "
          f"{'per_txn':>9s}  {'review%':>8s}  {'block%':>7s}  {'capture':>8s}")
    print(f"{'WIN':>4s}  {frozen_review:9.6f}  {frozen_block:9.6f}  "
          f"{search.outcome.total_expected_cost:14,.2f}  "
          f"{search.outcome.expected_cost_per_transaction:9.4f}  "
          f"{search.outcome.review_rate:8.4%}  "
          f"{search.outcome.block_rate:7.4%}  "
          f"{search.outcome.fraud_capture_rate:8.4%}")

    for rank, runner in enumerate(search.runners_up, start=1):
        print(f"{rank:>4d}  {runner.review_threshold:9.6f}  "
              f"{runner.block_threshold:9.6f}  "
              f"{runner.total_expected_cost:14,.2f}  "
              f"{runner.expected_cost_per_transaction:9.4f}  "
              f"{runner.review_rate:8.4%}  {runner.block_rate:7.4%}  "
              f"{runner.fraud_capture_rate:8.4%}")

    print_section("STEP 6 — VALIDATION RESULTS FOR THE FROZEN POLICY")
    print(
        "Measured on forward-chaining out-of-fold VALIDATION probabilities "
        "(initial warm-up fold excluded), the same rows used for threshold "
        "selection. Cost columns combine measured outcomes with ASSUMPTION costs."
    )

    print_outcome(
        f"{frozen_config.policy_version} — VALIDATION (forward out-of-fold)",
        search.outcome,
    )

    refit_validation = apply_calibrator(
        fit_calibrator(LOCKED_CALIBRATION_METHOD, raw_validation, validation_labels),
        raw_validation,
    )
    print_outcome(
        f"{frozen_config.policy_version} — VALIDATION (in-sample refit, "
        "OPTIMISTIC, report only)",
        evaluate_policy(frozen_config, validation_labels, refit_validation),
    )

    print_section("STEP 6A — SENSITIVITY: REVIEW EFFECTIVENESS e (report only)")
    print(
        "Each row re-runs the full threshold search at a different assumed "
        "review capture rate. This is NOT used to select the primary policy; "
        f"the primary policy uses e = {PRIMARY_COSTS.review_fraud_capture_rate}."
    )
    print(f"{'e':>5s}  {'T_REVIEW':>9s}  {'T_BLOCK':>9s}  {'cost':>14s}  "
          f"{'allow%':>8s}  {'review%':>8s}  {'block%':>7s}  "
          f"{'capture':>8s}  {'FN':>6s}  {'FP':>7s}")

    for level in REVIEW_EFFECTIVENESS_LEVELS:
        costs = replace(
            PRIMARY_COSTS, review_fraud_capture_rate=level
        )
        result = search_thresholds(
            scored_oof_validation,
            scored_validation_labels,
            costs,
            candidates=candidates,
            policy_version=DEFAULT_POLICY_VERSION,
            runner_up_count=0,
        )
        outcome = result.outcome
        marker = "  <- primary" if level == PRIMARY_COSTS.review_fraud_capture_rate else ""

        print(f"{level:5.2f}  {outcome.review_threshold:9.6f}  "
              f"{outcome.block_threshold:9.6f}  "
              f"{outcome.total_expected_cost:14,.2f}  "
              f"{outcome.allow_rate:8.4%}  {outcome.review_rate:8.4%}  "
              f"{outcome.block_rate:7.4%}  {outcome.fraud_capture_rate:8.4%}  "
              f"{outcome.false_negatives:6,}  {outcome.false_positives:7,}"
              f"{marker}")

    print_section("STEP 6B — SENSITIVITY: REVIEW-VOLUME CAPS (report only)")
    print(
        "PROJECT ASSUMPTION about operational review capacity, not part of the "
        "primary objective. The primary policy has no cap. Each row shows what "
        "the minimum-cost policy would be if review volume were capped."
    )
    print(f"{'cap':>6s}  {'T_REVIEW':>9s}  {'T_BLOCK':>9s}  {'cost':>14s}  "
          f"{'review%':>8s}  {'block%':>7s}  {'capture':>8s}  {'vs primary':>12s}")

    primary_cost = search.outcome.total_expected_cost

    for cap in REVIEW_CAP_LEVELS:
        result = search_thresholds(
            scored_oof_validation,
            scored_validation_labels,
            PRIMARY_COSTS,
            candidates=candidates,
            max_review_rate=cap,
            policy_version=DEFAULT_POLICY_VERSION,
            runner_up_count=0,
        )
        outcome = result.outcome

        print(f"{cap:6.0%}  {outcome.review_threshold:9.6f}  "
              f"{outcome.block_threshold:9.6f}  "
              f"{outcome.total_expected_cost:14,.2f}  "
              f"{outcome.review_rate:8.4%}  {outcome.block_rate:7.4%}  "
              f"{outcome.fraud_capture_rate:8.4%}  "
              f"{outcome.total_expected_cost - primary_cost:+12,.2f}")

    print_section("STEP 7 — FREEZE AND DOCUMENT")
    print(f"FROZEN: {frozen_config.policy_version} "
          f"T_REVIEW={frozen_review:.6f} T_BLOCK={frozen_block:.6f}")
    print(
        "The thresholds above are now fixed. Nothing below can change them. "
        "TEST labels have not been read by any selection step."
    )
    print("\nMETHODOLOGICAL NUANCE (acknowledged limitation)")
    print(METHOD_NOTE)

    print_header("8/8", "SINGLE TEST EVALUATION OF THE FROZEN POLICY")

    test_labels = split.test[TARGET_COLUMN]

    calibrator = fit_calibrator(
        LOCKED_CALIBRATION_METHOD,
        raw_validation,
        validation_labels,
    )
    calibrated_test = apply_calibrator(calibrator, raw_test)

    raw_test_ranking = evaluate_predictions(test_labels, raw_test)
    calibrated_test_ranking = evaluate_predictions(
        test_labels, calibrated_test
    )
    calibrated_test_calibration = evaluate_calibration(
        test_labels, calibrated_test
    )

    for locked, actual, name in [
        (LOCKED_RAW_TEST_PR_AUC, raw_test_ranking.pr_auc, "raw TEST PR-AUC"),
        (LOCKED_RAW_TEST_ROC_AUC, raw_test_ranking.roc_auc, "raw TEST ROC-AUC"),
        (LOCKED_CALIBRATED_TEST_PR_AUC, calibrated_test_ranking.pr_auc,
         "calibrated TEST PR-AUC"),
        (LOCKED_CALIBRATED_TEST_ROC_AUC, calibrated_test_ranking.roc_auc,
         "calibrated TEST ROC-AUC"),
    ]:
        drift = abs(actual - locked)
        print(f"{name:24s}: locked={locked:.6f} now={actual:.6f} "
              f"drift={drift:.6f}")

        if drift > 1e-5:
            raise AssertionError(
                f"{name} drifted from the locked Phase 3 baseline."
            )

    print("\nAll four locked Phase 3 TEST figures reproduce exactly. "
          "The model, split, features and calibrator were not modified.")

    print_section("MODEL — TEST (final held-out)")
    print(f"TEST rows          : {len(test_labels):,}")
    print(f"TEST fraud         : {int(test_labels.sum()):,}")
    print(f"TEST fraud rate    : {test_labels.mean():.6f}")
    print(f"raw TEST PR-AUC    : {raw_test_ranking.pr_auc:.6f}")
    print(f"raw TEST ROC-AUC   : {raw_test_ranking.roc_auc:.6f}")
    print_model_metrics(
        f"CALIBRATED {LOCKED_CALIBRATION_METHOD.upper()} — TEST",
        calibrated_test_calibration,
    )
    print(f"calibrated TEST PR-AUC  : {calibrated_test_ranking.pr_auc:.6f}")
    print(f"calibrated TEST ROC-AUC : {calibrated_test_ranking.roc_auc:.6f}")

    test_outcome = evaluate_policy(
        frozen_config, test_labels, calibrated_test
    )

    print_section("DECISION POLICY — TEST (final held-out, evaluated once)")
    print_outcome(f"{frozen_config.policy_version} — TEST", test_outcome)

    print("\nThresholds are unchanged after seeing TEST:")
    print(f"  before TEST: T_REVIEW={frozen_review:.6f} T_BLOCK={frozen_block:.6f}")
    print(f"  after TEST : T_REVIEW={test_outcome.review_threshold:.6f} "
          f"T_BLOCK={test_outcome.block_threshold:.6f}")

    if (test_outcome.review_threshold != frozen_review
            or test_outcome.block_threshold != frozen_block):
        raise AssertionError("Thresholds were modified after TEST evaluation.")

    print_section("LEAKAGE AUDIT")
    print("1. Thresholds were selected using VALIDATION only (out-of-fold "
          "calibrated probabilities).")
    print("2. TEST labels were not read by any selection, search or "
          "sensitivity step; they are first read in STEP 8.")
    print("3. No threshold was changed after seeing TEST; the script asserts "
          "this above.")
    print("4. Calibrated probabilities came from the locked Phase 3 pipeline: "
          f"{LOCKED_CALIBRATION_METHOD} selected on out-of-fold VALIDATION "
          "Brier, refitted on all VALIDATION for deployment.")
    print("5. The XGBoost model, 28-feature contract, temporal split, "
          "preprocessing and calibrator were not modified — verified by "
          "reproducing all four locked TEST figures.")
    print("6. Known limitation: the threshold-selection scale (fold-specific "
          "isotonic mappings) and the deployment scale (single refit isotonic "
          "mapping) are not literally identical. The VALIDATION cost is "
          "therefore not a perfectly unbiased estimate of the TEST cost.")

    print(f"\nTotal seconds: {time.perf_counter() - started:.2f}")

    print_section("PHASE 4 DECISION-POLICY SELECTION COMPLETE")


if __name__ == "__main__":
    main()
