from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.data.modeling import (  # noqa: E402
    BEHAVIORAL_FEATURES,
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.xgboost_model import build_xgboost_model  # noqa: E402


DEPTHS = (3, 4, 5, 6)
N_FOLDS = 5
THRESHOLD = 0.5
LOSS_EPSILON = 1e-7
BOOTSTRAP_SEED = 20261116
BOOTSTRAP_REPEATS = 500

# Phase 1 locked chronological partition identity (TRAIN and VALIDATION).
# These are the ordered (TransactionID, TransactionDT, isFraud) SHA-256 digests
# recorded in the Phase 1 rerun metadata. The Phase 2 runner reconstructs the
# split from the behavioral chronology alone and must fail closed if it differs.
PHASE1_TRAIN_SHA256 = "db3d02c445d25bf79f6e304c2d262aafce89678689395b2445fd2f581dcc4357"
PHASE1_VALIDATION_SHA256 = "998af0e9a39163817d747088de53677b6db6a30ef64d6b82e8e605f22e0128f2"


@dataclass(frozen=True)
class ForwardFold:
    fold: int
    train_stop: int
    validation_stop: int


def forward_folds(frame: pd.DataFrame) -> list[ForwardFold]:
    timestamps = frame.TransactionDT
    if frame.empty or timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Folds require nonempty sorted nonmissing timestamps")
    unique = timestamps.drop_duplicates().to_numpy()
    warmup = len(unique) // 2
    edges = np.linspace(warmup, len(unique), N_FOLDS + 1, dtype=int)
    if warmup == 0 or (np.diff(edges) <= 0).any():
        raise ValueError("Too few timestamps for five forward folds")
    row_edges = timestamps.searchsorted(unique[edges - 1], side="right")
    return [ForwardFold(i + 1, int(row_edges[i]), int(row_edges[i + 1])) for i in range(N_FOLDS)]


def pooled_oof(labels: np.ndarray, probabilities: np.ndarray, folds: list[ForwardFold]) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("OOF predictions must align to original TRAIN")
    if not folds or folds[-1].validation_stop != len(labels):
        raise ValueError("OOF folds must cover the end of original TRAIN")
    eligible = np.zeros(len(labels), dtype=bool)
    previous_stop = folds[0].train_stop
    for fold in folds:
        if not 0 < fold.train_stop < fold.validation_stop <= len(labels):
            raise ValueError("Invalid OOF fold bounds")
        if fold.train_stop != previous_stop:
            raise ValueError("OOF scoring blocks must be contiguous and disjoint")
        rows = slice(fold.train_stop, fold.validation_stop)
        eligible[rows] = True
        previous_stop = fold.validation_stop
    if not np.isnan(probabilities[~eligible]).all() or not np.isfinite(probabilities[eligible]).all():
        raise ValueError("OOF coverage must contain exactly the eligible rows")
    return labels[eligible], probabilities[eligible]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def snapshot(roots: list[Path]) -> dict:
    files = set()
    checkouts = {}
    for root in roots:
        checkouts[str(root)] = {"head": git(root, "rev-parse", "HEAD"), "branch": git(root, "branch", "--show-current"), "status": git(root, "status", "--short")}
        for directory in ("reports", "artifacts", "data"):
            for path in (root / directory).rglob("*"):
                if path.is_file() and not (root == ROOT and path.is_relative_to(ROOT / "reports/tree_depth_phase2")):
                    files.add(path.resolve())
        names = git(root, "ls-files", "-z").split("\0") + git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
        for name in names:
            if root == ROOT and name.startswith("reports/tree_depth_phase2/"):
                continue
            path = root / name
            if name and path.is_file():
                files.add(path.resolve())
    return {"checkouts": checkouts, "files": {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(files)}}


def create_output(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run-id")
    output = ROOT / "reports/tree_depth_phase2" / run_id
    if output.resolve() != ROOT.resolve() / "reports/tree_depth_phase2" / run_id:
        raise ValueError("Output resolves outside isolated Phase 2 reports")
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    return output


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or labels.shape != probabilities.shape
        or len(labels) == 0
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError("Labels must be a nonempty aligned binary vector")
    if np.unique(labels).size != 2:
        raise ValueError("Metrics require both fraud and non-fraud cases")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    positive = probabilities >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(labels, positive, labels=[0, 1]).ravel()
    return {
        "rows": len(labels), "fraud_cases": int(labels.sum()), "fraud_rate": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, positive, zero_division=0)),
        "recall": float(recall_score(labels, positive, zero_division=0)),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "brier_score": float(brier_score_loss(labels, probabilities.astype(float))),
        "log_loss": float(log_loss(labels, np.clip(probabilities.astype(float), LOSS_EPSILON, 1 - LOSS_EPSILON))),
        "mean_probability": float(probabilities.mean()), "threshold": THRESHOLD,
    }


def describe(frame: pd.DataFrame) -> dict:
    values = np.ascontiguousarray(frame[["TransactionID", "TransactionDT", TARGET_COLUMN]].to_numpy(dtype="<i8"))
    return {"rows": len(frame), "fraud_cases": int(frame[TARGET_COLUMN].sum()), "dt_min": int(frame.TransactionDT.min()), "dt_max": int(frame.TransactionDT.max()), "id_min": int(frame.TransactionID.min()), "id_max": int(frame.TransactionID.max()), "distinct_timestamps": int(frame.TransactionDT.nunique()), "ordered_id_timestamp_label_sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def load_partition(chronology: pd.DataFrame, *, preceding_rows: int) -> pd.DataFrame:
    behavioral = pd.read_parquet(ROOT / "data/processed/train_behavioral_v1.parquet", columns=["TransactionID", "TransactionDT", *BEHAVIORAL_FEATURES], filters=[("TransactionDT", ">=", int(chronology.TransactionDT.min())), ("TransactionDT", "<=", int(chronology.TransactionDT.max()))])
    transactions = pd.read_csv(ROOT / "data/raw/ieee_cis/train_transaction.csv", usecols=["TransactionID", TARGET_COLUMN, *CATEGORICAL_FEATURES, *CD_FEATURES], skiprows=range(1, preceding_rows + 1), nrows=len(chronology))
    if not np.array_equal(transactions.TransactionID.to_numpy(), chronology.TransactionID.to_numpy()):
        raise ValueError("Raw CSV ordering does not match the locked chronological partition")
    modeling = build_modeling_frame(behavioral, transactions, transactions).data
    if not np.array_equal(modeling.TransactionID.to_numpy(), chronology.TransactionID.to_numpy()):
        raise ValueError("Modeling rows differ from the locked partition")
    return modeling


def enforce_phase1_partition_identity(frame: pd.DataFrame, expected_sha256: str, label: str) -> str:
    """Fail closed unless ``frame`` matches the locked Phase 1 partition digest.

    The digest is the SHA-256 of the ordered (TransactionID, TransactionDT,
    isFraud) int64 matrix, identical to :func:`describe`'s identity hash. This
    is checked *after* the frame is loaded so a silent upstream data change
    aborts before any model is fit.
    """

    actual = describe(frame)["ordered_id_timestamp_label_sha256"]
    if actual != expected_sha256:
        raise ValueError(
            f"{label} partition identity mismatch: reconstructed {actual} "
            f"does not equal locked Phase 1 {expected_sha256}"
        )
    return actual


def fit_pipeline(train: pd.DataFrame, depth: int):
    pipeline = build_xgboost_model(train).model
    pipeline.set_params(classifier__scale_pos_weight=1.0, classifier__max_depth=depth)
    started = time.perf_counter()
    pipeline.fit(train[MODEL_FEATURES], train[TARGET_COLUMN])
    return pipeline, time.perf_counter() - started


def pipeline_record(pipeline, fit_seconds: float) -> dict:
    classifier = pipeline.named_steps["classifier"]
    booster = classifier.get_booster()
    return {
        "parameters": {key: "NaN" if isinstance(value, float) and np.isnan(value) else value for key, value in classifier.get_params(deep=False).items()},
        "fit_seconds": fit_seconds,
        "preprocessor_hash": joblib.hash(pipeline.named_steps["preprocessor"]),
        "n_transformed_columns": int(classifier.n_features_in_),
        "boosted_rounds": booster.num_boosted_rounds(),
        "booster_sha256": hashlib.sha256(booster.save_raw(raw_format="json")).hexdigest(),
        "booster_configuration": json.loads(booster.save_config()),
    }


def score(pipeline, frame: pd.DataFrame, output: Path, name: str, **identity) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    probabilities = pipeline.predict_proba(frame[MODEL_FEATURES])[:, 1]
    elapsed = time.perf_counter() - started
    result = {**identity, **metrics(frame[TARGET_COLUMN].to_numpy(), probabilities), "inference_seconds": elapsed, "prediction_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(), "prediction_file": f"{name}.npz"}
    np.savez_compressed(output / result["prediction_file"], transaction_id=frame.TransactionID.to_numpy(), transaction_dt=frame.TransactionDT.to_numpy(), label=frame[TARGET_COLUMN].to_numpy(), probability=probabilities)
    return probabilities, result


def run_folds(train: pd.DataFrame, folds: list[ForwardFold], output: Path) -> dict:
    records = []
    rows = []
    pooled = []
    preprocessing_by_fold = {}
    for depth in DEPTHS:
        oof = np.full(len(train), np.nan, dtype=np.float32)
        fit_total = inference_total = 0.0
        for fold in folds:
            identity = {"depth": depth, "fold": fold.fold}
            record = {**identity, "status": "RUNNING"}
            print(f"START depth={depth} fold={fold.fold}", flush=True)
            try:
                fit = train.iloc[:fold.train_stop]
                validation = train.iloc[fold.train_stop:fold.validation_stop]
                if not fit.TransactionDT.max() < validation.TransactionDT.min():
                    raise AssertionError("Fold timestamps overlap")
                pipeline, elapsed = fit_pipeline(fit, depth)
                record.update(pipeline_record(pipeline, elapsed))
                record.update(train=describe(fit), validation=describe(validation))
                reference = preprocessing_by_fold.setdefault(fold.fold, record["preprocessor_hash"])
                if record["preprocessor_hash"] != reference:
                    raise AssertionError("Preprocessing differs across depths for the same fold")
                _, train_metrics = score(pipeline, fit, output, f"depth_{depth}_fold_{fold.fold}_train", **identity, partition="fold_train", fit_seconds=elapsed)
                probabilities, validation_metrics = score(pipeline, validation, output, f"depth_{depth}_fold_{fold.fold}_validation", **identity, partition="fold_validation", fit_seconds=elapsed)
                oof[fold.train_stop:fold.validation_stop] = probabilities
                rows.extend([train_metrics, validation_metrics])
                fit_total += elapsed
                inference_total += validation_metrics["inference_seconds"]
                record["status"] = "PASS"
                print(f"PASS depth={depth} fold={fold.fold} fit_seconds={elapsed:.3f} train_AP={train_metrics['average_precision']:.12f} fold_AP={validation_metrics['average_precision']:.12f}", flush=True)
                del pipeline
                gc.collect()
            except Exception as error:
                record.update(status="FAIL", error_type=type(error).__name__, error=str(error))
                records.append(record)
                write_json(output / "fits.json", {"fits": records})
                pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
                raise
            records.append(record)
            write_json(output / "fits.json", {"fits": records})
            pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
        labels, probabilities = pooled_oof(train[TARGET_COLUMN].to_numpy(), oof, folds)
        eligible = train.iloc[folds[0].train_stop:]
        fold_id = np.concatenate([np.full(fold.validation_stop - fold.train_stop, fold.fold) for fold in folds])
        filename = f"depth_{depth}_oof.npz"
        np.savez_compressed(output / filename, transaction_id=eligible.TransactionID.to_numpy(), transaction_dt=eligible.TransactionDT.to_numpy(), label=labels, probability=probabilities, fold=fold_id)
        fold_aps = [
            row["average_precision"]
            for row in rows
            if row.get("depth") == depth and row.get("partition") == "fold_validation"
        ]
        mean_fold_ap = float(np.mean(fold_aps)) if fold_aps else float("nan")
        pooled_metrics = metrics(labels, probabilities)
        result = {"depth": depth, "partition": "pooled_oof", **pooled_metrics, "mean_fold_average_precision": mean_fold_ap, "pooled_oof_average_precision": pooled_metrics["average_precision"], "cumulative_fold_fit_seconds": fit_total, "cumulative_oof_inference_seconds": inference_total, "prediction_file": filename, "prediction_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest()}
        pooled.append(result)
        pd.DataFrame(pooled).to_csv(output / "pooled_oof_metrics.csv", index=False)
        print(f"POOLED depth={depth} OOF_rows={len(labels)} pooled_AP={result['pooled_oof_average_precision']:.12f} mean_fold_AP={mean_fold_ap:.12f}", flush=True)
    ranked = sorted(pooled, key=lambda row: (-row["average_precision"], row["depth"]))
    selected = {"status": "PASS", "selected_depth": ranked[0]["depth"], "selection_metric": "highest pooled TRAIN-contained OOF sklearn average_precision_score", "selection_metric_note": "primary selection remains pooled OOF AP; mean_fold_average_precision is reported alongside for transparency and is not the selection statistic", "ranking": ranked, "runner_up_ap_gap": ranked[0]["average_precision"] - ranked[1]["average_precision"], "within_predeclared_0_005_close_flag": ranked[0]["average_precision"] - ranked[1]["average_precision"] < 0.005, "selection_frozen_before_existing_validation_loaded": True}
    write_json(output / "selection.json", selected)
    print(f"SELECTED depth={selected['selected_depth']} pooled_AP={ranked[0]['average_precision']:.12f} mean_fold_AP={ranked[0]['mean_fold_average_precision']:.12f}", flush=True)
    return selected


def bootstrap_comparisons(output: Path, selection: dict) -> None:
    depth = selection["selected_depth"]
    with np.load(output / f"depth_{depth}_oof.npz", allow_pickle=False) as archive:
        labels = archive["label"]
        probabilities = archive["probability"]
        timestamps = archive["transaction_dt"]
    # Sampling units are contiguous 24-hour blocks of the eligible OOF window,
    # formed from distinct TransactionDT values so tied timestamps stay together.
    # Each sampled block is stored to completion (no truncation at the series
    # end), so resamples always reconstruct the full eligible row set. This is
    # the smallest correction that makes the "n blocks drawn -> full-length
    # resample" invariant hold for an expanding-window OOF series whose first
    # timestamp is far from zero. See the audit C3 note.
    unique_times, group_codes = np.unique(timestamps, return_inverse=True)
    block_ids = (timestamps - unique_times[0]) // 86400
    groups = [np.flatnonzero(block_ids == block) for block in np.unique(block_ids)]
    references = sorted({6, selection["ranking"][1]["depth"]} - {depth})
    results = []
    for reference in references:
        started = time.perf_counter()
        print(f"BOOTSTRAP selected={depth} reference={reference} replicates={BOOTSTRAP_REPEATS} blocks={len(groups)}", flush=True)
        with np.load(output / f"depth_{reference}_oof.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["transaction_dt"], timestamps) or not np.array_equal(archive["label"], labels):
                raise ValueError("Bootstrap reference rows do not align")
            other = archive["probability"]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        differences = []
        attempts = 0
        max_attempts = BOOTSTRAP_REPEATS * 20
        while len(differences) < BOOTSTRAP_REPEATS and attempts < max_attempts:
            attempts += 1
            indices = np.concatenate(
                [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            )
            sampled_labels = labels[indices]
            # AP is not informative when a resample contains only one class.
            # Reject that draw and keep the paired comparison on identical rows.
            if np.unique(sampled_labels).size != 2:
                continue
            differences.append(
                float(
                    average_precision_score(sampled_labels, probabilities[indices])
                    - average_precision_score(sampled_labels, other[indices])
                )
            )
        if len(differences) != BOOTSTRAP_REPEATS:
            raise RuntimeError(
                "Could not obtain enough two-class bootstrap replicates; "
                f"accepted={len(differences)}, attempts={attempts}"
            )
        np.savez_compressed(output / f"bootstrap_depth_{depth}_minus_{reference}.npz", ap_difference=np.array(differences))
        lower, upper = np.quantile(differences, [0.025, 0.975])
        elapsed = time.perf_counter() - started
        results.append({"selected_depth": depth, "reference_depth": reference, "point_ap_difference": float(average_precision_score(labels, probabilities) - average_precision_score(labels, other)), "lower_95_percentile": float(lower), "upper_95_percentile": float(upper), "blocks": len(groups), "block_seconds": 86400, "replicates": BOOTSTRAP_REPEATS, "attempts": attempts, "rejected_single_class_replicates": attempts - BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED, "seconds": elapsed, "method": "paired nonoverlapping 24-hour timestamp-block resampling with replacement; blocks drawn to completion over distinct TransactionDT groups; final calendar block may be partial; single-class draws rejected", "limitation": "conditional on selected fits and chosen blocks; not adjusted for winner selection, temporal drift, seed or retraining uncertainty"})
        print(f"BOOTSTRAP_DONE reference={reference} interval=[{lower:.12f},{upper:.12f}] seconds={elapsed:.3f}", flush=True)
    write_json(output / "oof_bootstrap_diagnostics.json", {"comparisons": results, "does_not_change_selection": True})


def confirm(train: pd.DataFrame, validation_chronology: pd.DataFrame, selected_depth: int, output: Path, phase1: Path) -> None:
    print(f"REFIT selected_depth={selected_depth} full_original_TRAIN", flush=True)
    pipeline, elapsed = fit_pipeline(train, selected_depth)
    record = pipeline_record(pipeline, elapsed)
    record.update(depth=selected_depth, train=describe(train))
    write_json(output / "confirmation_fit.json", record)
    _, train_metrics = score(pipeline, train, output, "selected_full_train", depth=selected_depth, partition="full_train", fit_seconds=elapsed)
    validation = load_partition(validation_chronology, preceding_rows=len(train))
    _, validation_metrics = score(pipeline, validation, output, "selected_existing_validation", depth=selected_depth, partition="existing_validation_retrospective_confirmation", fit_seconds=elapsed)
    pd.DataFrame([train_metrics, validation_metrics]).to_csv(output / "confirmation_metrics.csv", index=False)
    comparisons = [dict(validation_metrics, candidate="phase2_selected", provenance="one prediction call after TRAIN-only OOF selection")]
    for name in ("weight_1", "current_dynamic"):
        source = phase1 / "reports/class_weight_phase1/phase1-20261116-rerun" / f"{name}_validation_predictions.npz"
        with np.load(source, allow_pickle=False) as archive:
            if not np.array_equal(archive["transaction_id"], validation.TransactionID.to_numpy()):
                raise ValueError("Phase 1 reference rows differ from existing VALIDATION")
            if "transaction_dt" in archive.files and not np.array_equal(
                archive["transaction_dt"], validation.TransactionDT.to_numpy()
            ):
                raise ValueError("Phase 1 reference timestamps differ from existing VALIDATION")
            probabilities = np.asarray(archive["probability"])
            if (
                probabilities.shape != (len(validation),)
                or not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError("Phase 1 reference probabilities are invalid")
        comparisons.append({"candidate": f"phase1_{name}", "depth": 6, "provenance": "recomputed metrics from preserved Phase 1 predictions; no new model scoring", **metrics(validation[TARGET_COLUMN].to_numpy(), probabilities), "source_file": str(source), "source_sha256": sha256(source)})
    pd.DataFrame(comparisons).to_csv(output / "phase1_confirmation_comparison.csv", index=False)
    print(f"CONFIRMATION selected_depth={selected_depth} exposed_VALIDATION_AP={validation_metrics['average_precision']:.12f}", flush=True)


def preflight(
    phase1_root: Path,
    protected_roots: list[Path],
) -> list[Path]:
    """Fail before model fitting if inputs or preservation roots are invalid."""
    phase1_root = phase1_root.resolve()
    if not phase1_root.is_dir() or not (phase1_root / ".git").exists():
        raise FileNotFoundError(f"Phase 1 root is not a Git checkout: {phase1_root}")
    if phase1_root == ROOT.resolve():
        raise ValueError("Phase 1 root must be separate from the Phase 2 checkout")

    phase1_dir = phase1_root / "reports/class_weight_phase1/phase1-20261116-rerun"
    required = [
        phase1_dir / "weight_1_validation_predictions.npz",
        phase1_dir / "current_dynamic_validation_predictions.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required preserved Phase 1 predictions are missing: " + ", ".join(missing)
        )

    required_data = [
        ROOT / "data/processed/train_behavioral_v1.parquet",
        ROOT / "data/raw/ieee_cis/train_transaction.csv",
    ]
    missing_data = [str(path) for path in required_data if not path.is_file()]
    if missing_data:
        raise FileNotFoundError(
            "Required Phase 2 data inputs are missing: " + ", ".join(missing_data)
        )

    roots = [ROOT.resolve(), phase1_root]
    roots.extend(path.resolve() for path in protected_roots)
    unique_roots = list(dict.fromkeys(roots))
    for root in unique_roots:
        if not root.is_dir() or not (root / ".git").exists():
            raise FileNotFoundError(f"Protected root is not a Git checkout: {root}")
    return unique_roots



def validate_phase1_predictions(
    phase1_root: Path,
    validation_chronology: pd.DataFrame,
) -> dict[str, str]:
    """Check preserved Phase 1 prediction artifacts against validation row IDs only."""
    phase1_dir = (
        phase1_root.resolve()
        / "reports/class_weight_phase1/phase1-20261116-rerun"
    )
    expected_ids = validation_chronology.TransactionID.to_numpy()
    expected_times = validation_chronology.TransactionDT.to_numpy()
    hashes = {}
    for name in ("weight_1", "current_dynamic"):
        source = phase1_dir / f"{name}_validation_predictions.npz"
        with np.load(source, allow_pickle=False) as archive:
            if not {"transaction_id", "probability"}.issubset(archive.files):
                raise ValueError(f"Phase 1 artifact is missing required arrays: {source}")
            if not np.array_equal(archive["transaction_id"], expected_ids):
                raise ValueError(
                    f"Phase 1 {name} prediction IDs do not match reconstructed VALIDATION"
                )
            if "transaction_dt" in archive.files and not np.array_equal(
                archive["transaction_dt"], expected_times
            ):
                raise ValueError(
                    f"Phase 1 {name} prediction timestamps do not match VALIDATION"
                )
            probabilities = np.asarray(archive["probability"])
            if (
                probabilities.shape != (len(expected_ids),)
                or not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError(
                    f"Phase 1 {name} predictions are not finite aligned probabilities"
                )
        hashes[name] = sha256(source)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--protected-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs, Phase 1 alignment, data loading, folds, and preservation snapshots without fitting models.",
    )
    args = parser.parse_args()

    output = create_output(args.run_id)
    started = time.perf_counter()
    metadata = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "features": MODEL_FEATURES,
        "phase1_root": str(args.phase1_root.resolve()),
        "protocol": {
            "depths": list(DEPTHS),
            "scale_pos_weight": 1,
            "folds": N_FOLDS,
            "warmup_distinct_timestamp_fraction": 0.5,
            "folds_within_original_train_only": True,
            "selection": "highest pooled OOF AP; exact ties prefer shallower depth",
            "close_flag_ap_gap": 0.005,
            "threshold": THRESHOLD,
            "brier_reduction": "float64",
            "log_loss_clip_epsilon": LOSS_EPSILON,
            "calibration": "none",
            "original_validation": (
                "one selected-model prediction call after frozen selection; "
                "previously exposed retrospective confirmation"
            ),
            "historical_test": (
                "only label-free timestamps used to reconstruct original split; "
                "no TEST feature or target frame loaded, no predictions or selection"
            ),
            "bootstrap": (
                "500 paired 24-hour nonoverlapping timestamp-block replicates, "
                "seed20261116, diagnostic only; single-class draws rejected"
            ),
        },
    }
    roots: list[Path] = []
    before = None
    try:
        roots = preflight(args.phase1_root, args.protected_root)
        metadata.update(
            source_head=git(ROOT, "rev-parse", "HEAD"),
            source_branch=git(ROOT, "branch", "--show-current"),
            libraries={
                name: importlib.metadata.version(name)
                for name in (
                    "xgboost", "scikit-learn", "pandas", "numpy", "scipy", "joblib"
                )
            },
        )
        before = snapshot(roots)
        write_json(output / "protected_before.json", before)
        write_json(output / "run_metadata.json", metadata)

        chronology = pd.read_parquet(
            ROOT / "data/processed/train_behavioral_v1.parquet",
            columns=["TransactionID", "TransactionDT"],
        )
        if (
            chronology.empty
            or chronology[["TransactionID", "TransactionDT"]].isna().any().any()
            or chronology.TransactionID.duplicated().any()
            or not chronology.TransactionDT.is_monotonic_increasing
        ):
            raise ValueError(
                "Behavioral chronology must be nonempty, unique by transaction, "
                "complete, and sorted by TransactionDT"
            )
        timestamps = chronology.TransactionDT.drop_duplicates().to_numpy()
        if len(timestamps) < 10:
            raise ValueError("Not enough distinct timestamps to create chronological partitions")

        train_end = timestamps[int(len(timestamps) * 0.70) - 1]
        validation_end = timestamps[int(len(timestamps) * (0.70 + 0.15)) - 1]
        train_chronology = chronology[chronology.TransactionDT <= train_end]
        validation_chronology = chronology[
            (chronology.TransactionDT > train_end)
            & (chronology.TransactionDT <= validation_end)
        ]
        if train_chronology.empty or validation_chronology.empty:
            raise ValueError("Reconstructed TRAIN or VALIDATION partition is empty")
        metadata["phase1_prediction_sha256"] = validate_phase1_predictions(
            args.phase1_root, validation_chronology
        )
        write_json(output / "run_metadata.json", metadata)

        train = load_partition(train_chronology, preceding_rows=0)
        metadata["phase1_train_partition_sha256"] = enforce_phase1_partition_identity(
            train, PHASE1_TRAIN_SHA256, "TRAIN"
        )
        validation_frame = load_partition(validation_chronology, preceding_rows=len(train))
        metadata["phase1_validation_partition_sha256"] = (
            enforce_phase1_partition_identity(
                validation_frame, PHASE1_VALIDATION_SHA256, "VALIDATION"
            )
        )
        del validation_frame
        gc.collect()
        folds = forward_folds(train)
        fold_table = [
            {
                **asdict(fold),
                "train": describe(train.iloc[:fold.train_stop]),
                "validation": describe(
                    train.iloc[fold.train_stop:fold.validation_stop]
                ),
            }
            for fold in folds
        ]
        metadata.update(
            original_train=describe(train),
            original_train_end=int(train_end),
            existing_validation_end=int(validation_end),
            existing_validation_rows=len(validation_chronology),
            warmup_rows=folds[0].train_stop,
            eligible_oof_rows=len(train) - folds[0].train_stop,
            fold_boundaries=fold_table,
        )
        write_json(output / "run_metadata.json", metadata)
        print(json.dumps(fold_table, indent=2), flush=True)

        if args.preflight_only:
            metadata["status"] = "PREFLIGHT_PASS"
            metadata["preflight_only"] = True
            metadata["model_fits_started"] = 0
            metadata["bootstrap_called"] = 0
            metadata["confirmation_called"] = 0
            write_json(output / "run_metadata.json", metadata)
            print("PREFLIGHT_PASS: no model fitting was started", flush=True)
        else:
            selection = run_folds(train, folds, output)
            metadata["selected_depth"] = selection["selected_depth"]
            metadata["bootstrap_called"] = 1
            metadata["confirmation_called"] = 1
            write_json(output / "run_metadata.json", metadata)
            bootstrap_comparisons(output, selection)
            confirm(
                train,
                validation_chronology,
                selection["selected_depth"],
                output,
                args.phase1_root.resolve(),
            )
            metadata["status"] = "PASS"
    except Exception as error:
        metadata.update(
            status="FAIL",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        files_unchanged = False
        checkouts_unchanged = False
        if before is not None:
            try:
                after = snapshot(roots)
                write_json(output / "protected_after.json", after)
                files_unchanged = before["files"] == after["files"]
                checkouts_unchanged = all(
                    before["checkouts"][path] == after["checkouts"][path]
                    for path in before["checkouts"]
                    if path != str(ROOT)
                )
            except Exception as error:
                metadata["after_snapshot_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        metadata.update(
            protected_files_unchanged=files_unchanged,
            protected_other_checkouts_unchanged=checkouts_unchanged,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
        )
        if not files_unchanged or not checkouts_unchanged:
            metadata["status"] = "FAIL"
        write_json(output / "run_metadata.json", metadata)
        print(
            "FINISHED "
            f"status={metadata['status']} "
            f"protected_files_unchanged={files_unchanged} "
            f"protected_other_checkouts_unchanged={checkouts_unchanged} "
            f"seconds={metadata['total_seconds']:.3f}",
            flush=True,
        )
    if metadata["status"] not in {"PASS", "PREFLIGHT_PASS"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
