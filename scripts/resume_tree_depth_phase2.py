"""Resume Phase 2 post-selection work from verified, preserved OOF artifacts.

This script deliberately does not rerun the 20 depth/fold fits. It validates
those artifacts, performs the same paired moving-time-block bootstrap with
timestamp groups precomputed once, then runs the one selected-depth
retrospective VALIDATION confirmation.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts import run_tree_depth_phase2 as runner


BLOCK_SECONDS = 86400


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def precompute_timestamp_groups(timestamps: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build sorted unique timestamps and row-index groups once."""
    values = np.asarray(timestamps)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timestamps must be a nonempty one-dimensional array")
    if pd.isna(values).any() or np.any(values[1:] < values[:-1]):
        raise ValueError("timestamps must be nonmissing and sorted")
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    groups = np.split(np.arange(values.size, dtype=np.int64), boundaries)
    unique = values[np.r_[0, boundaries]]
    if unique.size < 2 or BLOCK_SECONDS >= unique[-1] - unique[0]:
        raise ValueError("block duration must be shorter than timestamp span")
    return unique, groups


def moving_block_indices_precomputed(
    unique: np.ndarray,
    row_groups: list[np.ndarray],
    rng: np.random.Generator,
    *,
    block_duration_seconds: int = BLOCK_SECONDS,
) -> np.ndarray:
    """Match moving_time_block_indices exactly, reusing precomputed groups."""
    if block_duration_seconds < 1:
        raise ValueError("block_duration_seconds must be at least 1")
    n_groups = len(row_groups)
    if n_groups == 0 or len(unique) != n_groups:
        raise ValueError("unique timestamps and row groups must align")
    selected_groups: list[int] = []
    while len(selected_groups) < n_groups:
        start = int(rng.integers(0, n_groups))
        end = int(
            np.searchsorted(
                unique, unique[start] + block_duration_seconds, side="left"
            )
        )
        end = max(end, start + 1)
        take = min(end - start, n_groups - len(selected_groups))
        selected_groups.extend(range(start, start + take))
    return np.concatenate([row_groups[group] for group in selected_groups])


def validate_saved_run(source: Path, train: pd.DataFrame, folds: list) -> dict:
    """Fail closed unless the interrupted run contains complete aligned OOF results."""
    required = (
        "fits.json",
        "fold_metrics.csv",
        "pooled_oof_metrics.csv",
        "selection.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing preserved run artifacts: " + ", ".join(missing))

    fits = json.loads((source / "fits.json").read_text())["fits"]
    expected_pairs = {(depth, fold.fold) for depth in runner.DEPTHS for fold in folds}
    observed_pairs = {
        (int(item["depth"]), int(item["fold"]))
        for item in fits
        if item.get("status") == "PASS"
    }
    if len(fits) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError("Saved fit records are not exactly 20 successful depth/fold fits")
    if any(item.get("status") != "PASS" for item in fits):
        raise ValueError("At least one saved model fit did not pass")

    fold_metrics = pd.read_csv(source / "fold_metrics.csv")
    eligible = train.iloc[folds[0].train_stop:]
    expected_fold_id = np.concatenate([
        np.full(fold.validation_stop - fold.train_stop, fold.fold, dtype=np.int64)
        for fold in folds
    ])
    reference_labels = eligible[runner.TARGET_COLUMN].to_numpy()
    reference_ids = eligible.TransactionID.to_numpy()
    reference_times = eligible.TransactionDT.to_numpy()
    ranking = []
    for depth in runner.DEPTHS:
        path = source / f"depth_{depth}_oof.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing OOF predictions: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["transaction_id"]
            times = archive["transaction_dt"]
            labels = archive["label"]
            probabilities = archive["probability"]
            fold_id = archive["fold"]
        if not (
            np.array_equal(ids, reference_ids)
            and np.array_equal(times, reference_times)
            and np.array_equal(labels, reference_labels)
            and np.array_equal(fold_id, expected_fold_id)
        ):
            raise ValueError(f"Depth {depth} OOF row identity or fold assignment mismatch")
        if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
            raise ValueError(f"Depth {depth} OOF probabilities are invalid")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"Depth {depth} OOF probabilities are outside [0, 1]")

        rows = fold_metrics[
            (fold_metrics["depth"] == depth)
            & (fold_metrics["partition"] == "fold_validation")
        ].sort_values("fold")
        if len(rows) != runner.N_FOLDS or rows["fold"].astype(int).tolist() != list(range(1, runner.N_FOLDS + 1)):
            raise ValueError(f"Depth {depth} does not have exactly five fold-validation metric rows")
        mean_fold_ap = float(rows["average_precision"].mean())
        pooled_ap = float(average_precision_score(labels, probabilities))
        ranking.append({
            "depth": depth,
            "mean_fold_average_precision": mean_fold_ap,
            "pooled_oof_average_precision": pooled_ap,
        })

    recomputed = runner.select_depth(ranking)
    saved = json.loads((source / "selection.json").read_text())
    if int(saved.get("selected_depth", -1)) != int(recomputed["selected_depth"]):
        raise ValueError("Saved selected depth does not match recomputed OOF selection")
    for saved_row in saved.get("ranking", []):
        depth = int(saved_row["depth"])
        current = next(row for row in ranking if row["depth"] == depth)
        if not np.isclose(
            float(saved_row["mean_fold_average_precision"]),
            current["mean_fold_average_precision"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"Saved mean fold AP mismatch for depth {depth}")
        if not np.isclose(
            float(saved_row["pooled_oof_average_precision"]),
            current["pooled_oof_average_precision"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"Saved pooled OOF AP mismatch for depth {depth}")
    return recomputed


def run_bootstrap(source: Path, output: Path, selection: dict, repeats: int = runner.BOOTSTRAP_REPEATS) -> dict:
    selected = int(selection["selected_depth"])
    with np.load(source / f"depth_{selected}_oof.npz", allow_pickle=False) as archive:
        labels = archive["label"].astype(np.int8, copy=False)
        probabilities = archive["probability"]
        timestamps = archive["transaction_dt"]

    unique, row_groups = precompute_timestamp_groups(timestamps)
    runner_up = int(selection["ranking"][1]["depth"])
    references = sorted({6, runner_up} - {selected})
    minimum_valid = int(np.ceil(repeats * 0.90))
    comparisons = []

    for reference in references:
        with np.load(source / f"depth_{reference}_oof.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["transaction_dt"], timestamps):
                raise ValueError("Bootstrap reference timestamps do not align")
            if not np.array_equal(archive["label"], labels):
                raise ValueError("Bootstrap reference labels do not align")
            other = archive["probability"]

        rng = np.random.default_rng(runner.BOOTSTRAP_SEED)
        differences = []
        attempts = 0
        started = time.perf_counter()
        print(
            f"BOOTSTRAP_RESUME selected={selected} reference={reference} "
            f"replicates={repeats} groups={len(unique)}",
            flush=True,
        )
        for replicate in range(repeats):
            attempts += 1
            indices = moving_block_indices_precomputed(
                unique, row_groups, rng, block_duration_seconds=BLOCK_SECONDS
            )
            sampled_labels = labels[indices]
            if np.unique(sampled_labels).size == 2:
                differences.append(float(
                    average_precision_score(sampled_labels, probabilities[indices])
                    - average_precision_score(sampled_labels, other[indices])
                ))
            if (replicate + 1) % 50 == 0:
                print(
                    f"BOOTSTRAP_PROGRESS reference={reference} "
                    f"completed={replicate + 1}/{repeats} valid={len(differences)}",
                    flush=True,
                )

        valid = len(differences)
        if valid < minimum_valid:
            raise RuntimeError(
                f"Bootstrap valid replicates {valid} below required {minimum_valid}"
            )
        lower, upper = np.quantile(differences, [0.025, 0.975])
        np.savez_compressed(
            output / f"bootstrap_depth_{selected}_minus_{reference}.npz",
            ap_difference=np.asarray(differences, dtype=float),
        )
        comparisons.append({
            "selected_depth": selected,
            "reference_depth": reference,
            "point_pooled_oof_ap_difference": float(
                average_precision_score(labels, probabilities)
                - average_precision_score(labels, other)
            ),
            "lower_95_percentile": float(lower),
            "upper_95_percentile": float(upper),
            "distinct_timestamp_groups": int(len(unique)),
            "block_duration_seconds": BLOCK_SECONDS,
            "requested_replicates": repeats,
            "valid_replicates": valid,
            "attempts": attempts,
            "rejected_single_class_replicates": attempts - valid,
            "seed": runner.BOOTSTRAP_SEED,
            "seconds": time.perf_counter() - started,
            "method": "paired_moving_time_block_percentile_bootstrap",
            "sampling": (
                "same existing moving_time_block_indices draw sequence: complete "
                "tied TransactionDT groups, contiguous blocks without circular "
                "wrap-around, final block shortened; timestamp groups precomputed "
                "once for performance; paired indices shared between models"
            ),
            "limitation": (
                "diagnostic conditional on fitted models and selected block "
                "duration; does not adjust for winner selection, training "
                "randomness, retraining uncertainty, or future temporal drift; "
                "resampled row count may vary because timestamp groups have "
                "different row multiplicities"
            ),
        })
        print(
            f"BOOTSTRAP_DONE reference={reference} valid={valid}/{repeats} "
            f"interval=[{lower:.12f},{upper:.12f}]",
            flush=True,
        )

    result = {
        "comparisons": comparisons,
        "minimum_valid_replicates": minimum_valid,
        "does_not_change_selection": True,
        "implementation": "precomputed timestamp groups; identical seeded block draws",
    }
    _write_json(output / "oof_bootstrap_diagnostics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--protected-root", action="append", type=Path, default=[])
    args = parser.parse_args()

    source = (runner.ROOT / "reports/tree_depth_phase2" / args.source_run_id).resolve()
    expected_parent = (runner.ROOT / "reports/tree_depth_phase2").resolve()
    if source.parent != expected_parent or not source.is_dir():
        raise ValueError("Source run must be an existing direct child of Phase 2 reports")
    output = runner.create_output(args.run_id)
    started = time.perf_counter()
    metadata = {
        "status": "RUNNING",
        "resume_of": args.source_run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_called": 0,
        "confirmation_called": 0,
        "model_fits_repeated": 0,
        "protocol": (
            "Validate preserved 20-fold OOF artifacts; same paired 24-hour "
            "moving-time-block bootstrap seed and draw sequence; one retrospective "
            "existing-VALIDATION confirmation; no TEST labels or features."
        ),
    }
    roots = []
    before = None
    try:
        roots = runner.preflight(args.phase1_root, args.protected_root)
        before = runner.snapshot(roots)
        _write_json(output / "protected_before.json", before)

        chronology = pd.read_parquet(
            runner.ROOT / "data/processed/train_behavioral_v1.parquet",
            columns=["TransactionID", "TransactionDT"],
        )
        if (
            chronology.empty
            or chronology[["TransactionID", "TransactionDT"]].isna().any().any()
            or chronology.TransactionID.duplicated().any()
            or not chronology.TransactionDT.is_monotonic_increasing
        ):
            raise ValueError("Behavioral chronology is invalid")
        timestamps = chronology.TransactionDT.drop_duplicates().to_numpy()
        train_end = timestamps[int(len(timestamps) * 0.70) - 1]
        validation_end = timestamps[int(len(timestamps) * 0.85) - 1]
        train_chronology = chronology[chronology.TransactionDT <= train_end]
        validation_chronology = chronology[
            (chronology.TransactionDT > train_end)
            & (chronology.TransactionDT <= validation_end)
        ]

        phase1_hashes = runner.validate_phase1_predictions(
            args.phase1_root, validation_chronology
        )
        train = runner.load_partition(train_chronology, preceding_rows=0)
        runner.enforce_phase1_partition_identity(
            train, runner.PHASE1_TRAIN_SHA256, "TRAIN"
        )
        folds = runner.forward_folds(train)
        selection = validate_saved_run(source, train, folds)
        metadata.update(
            selected_depth=selection["selected_depth"],
            selection=selection,
            source_phase1_prediction_sha256=phase1_hashes,
            source_fits_passed=20,
            source_oof_artifacts_validated=True,
        )
        _write_json(output / "run_metadata.json", metadata)

        metadata["bootstrap_called"] = 1
        _write_json(output / "run_metadata.json", metadata)
        bootstrap = run_bootstrap(source, output, selection)
        metadata["bootstrap_completed"] = True
        metadata["bootstrap_valid_replicates"] = [
            item["valid_replicates"] for item in bootstrap["comparisons"]
        ]
        _write_json(output / "run_metadata.json", metadata)

        metadata["confirmation_called"] = 1
        _write_json(output / "run_metadata.json", metadata)
        runner.confirm(
            train,
            validation_chronology,
            int(selection["selected_depth"]),
            output,
            args.phase1_root.resolve(),
        )
        metadata["status"] = "PASS"
    except Exception as error:
        metadata.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        files_unchanged = False
        checkouts_unchanged = False
        if before is not None:
            try:
                after = runner.snapshot(roots)
                _write_json(output / "protected_after.json", after)
                files_unchanged = before["files"] == after["files"]
                checkouts_unchanged = all(
                    before["checkouts"][path] == after["checkouts"][path]
                    for path in before["checkouts"]
                    if path != str(runner.ROOT)
                )
            except Exception as error:
                metadata["after_snapshot_error"] = f"{type(error).__name__}: {error}"
        metadata.update(
            protected_files_unchanged=files_unchanged,
            protected_other_checkouts_unchanged=checkouts_unchanged,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
        )
        if not files_unchanged or not checkouts_unchanged:
            metadata["status"] = "FAIL"
        _write_json(output / "run_metadata.json", metadata)
        print(
            f"RESUME_FINISHED status={metadata['status']} "
            f"protected_files_unchanged={files_unchanged} "
            f"protected_other_checkouts_unchanged={checkouts_unchanged} "
            f"seconds={metadata['total_seconds']:.3f}",
            flush=True,
        )
    if metadata["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
