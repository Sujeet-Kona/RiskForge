"""Regression tests for the Phase 2 tree-depth runner.

These tests are deliberately free of model fitting and of the real IEEE-CIS
dataset. They exercise the runner's pure control-flow and alignment logic so
that the audited defects (preflight ``UnboundLocalError``, the duplicated
post-selection block, partition-identity enforcement, and the fold/OOF
bookkeeping) cannot silently regress.

Data-dependent checks are skipped when the dataset is absent so the default
suite stays green on a checkout without the (gitignored) IEEE-CIS files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_tree_depth_phase2 as runner  # noqa: E402


# --------------------------------------------------------------------------- #
# forward_folds
# --------------------------------------------------------------------------- #


def _sorted_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": np.arange(n, dtype=np.int64),
            "isFraud": (np.arange(n) % 17 == 0).astype(int),
        }
    )


def test_forward_folds_are_strictly_forward_and_cover_train():
    frame = _sorted_frame(1000)
    folds = runner.forward_folds(frame)

    assert len(folds) == runner.N_FOLDS
    previous_train_size = 0
    for fold in folds:
        fit = frame.iloc[: fold.train_stop]
        validation = frame.iloc[fold.train_stop : fold.validation_stop]
        assert len(fit) > previous_train_size
        previous_train_size = len(fit)
        assert fit["TransactionDT"].max() < validation["TransactionDT"].min()
        assert len(validation) > 0


def test_forward_folds_cover_the_full_training_span():
    frame = _sorted_frame(500)
    folds = runner.forward_folds(frame)

    assert folds[0].train_stop > 0
    assert folds[-1].validation_stop == len(frame)


def test_forward_folds_preserve_timestamp_ties():
    # Two rows per timestamp: no timestamp may be split across a fold boundary.
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(1, 201),
            "TransactionDT": np.repeat(np.arange(100, dtype=np.int64), 2),
            "isFraud": (np.arange(200) % 13 == 0).astype(int),
        }
    )
    folds = runner.forward_folds(frame)

    boundaries = {f.train_stop for f in folds} | {f.validation_stop for f in folds}
    for stop in boundaries:
        if 0 < stop < len(frame):
            assert frame["TransactionDT"].iloc[stop] != frame["TransactionDT"].iloc[stop - 1]


def test_forward_folds_reject_unsorted_timestamps():
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [1, 3, 2, 4],
            "isFraud": [0, 0, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="sorted"):
        runner.forward_folds(frame)


def test_forward_folds_reject_too_few_timestamps():
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(1, 6),
            "TransactionDT": np.arange(5, dtype=np.int64),
            "isFraud": [0, 0, 0, 0, 1],
        }
    )
    with pytest.raises(ValueError):
        runner.forward_folds(frame)


# --------------------------------------------------------------------------- #
# pooled_oof
# --------------------------------------------------------------------------- #


def _toy_oof(n: int = 200) -> tuple[pd.DataFrame, list[runner.ForwardFold]]:
    frame = _sorted_frame(n)
    return frame, runner.forward_folds(frame)


def test_pooled_oof_aligns_labels_and_probabilities():
    frame, folds = _toy_oof()
    oof = np.full(len(frame), np.nan, dtype=np.float32)
    for fold in folds:
        oof[fold.train_stop : fold.validation_stop] = 0.5

    labels, probabilities = runner.pooled_oof(frame["isFraud"].to_numpy(), oof, folds)
    eligible = frame.iloc[folds[0].train_stop :]

    assert len(labels) == len(probabilities) == len(eligible)
    assert np.array_equal(labels, eligible["isFraud"].to_numpy())
    assert np.isfinite(probabilities).all()
    assert np.isnan(oof[: folds[0].train_stop]).all()


def test_pooled_oof_rejects_misaligned_shapes():
    frame, folds = _toy_oof()
    oof = np.full(len(frame) + 5, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="OOF"):
        runner.pooled_oof(frame["isFraud"].to_numpy(), oof, folds)


def test_pooled_oof_rejects_uncovered_tail():
    frame, folds = _toy_oof()
    oof = np.full(len(frame), np.nan, dtype=np.float32)
    for fold in folds[:-1]:
        oof[fold.train_stop : fold.validation_stop] = 0.5
    with pytest.raises(ValueError, match="OOF coverage"):
        runner.pooled_oof(frame["isFraud"].to_numpy(), oof, folds)


# --------------------------------------------------------------------------- #
# create_output
# --------------------------------------------------------------------------- #


def test_create_output_rejects_invalid_run_id():
    with pytest.raises(ValueError, match="Invalid run-id"):
        runner.create_output("../escape")
    with pytest.raises(ValueError, match="Invalid run-id"):
        runner.create_output("")


def test_create_output_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    # A run-id with a slash fails the regex; a plain name succeeds.
    with pytest.raises(ValueError, match="Invalid run-id"):
        runner.create_output("nested/run")


# --------------------------------------------------------------------------- #
# Phase 1 partition identity
# --------------------------------------------------------------------------- #


def _identity_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "TransactionID": np.arange(3_000_000, 3_000_000 + n),
            "TransactionDT": np.sort(rng.integers(1, 10_000, n)),
            "isFraud": (rng.random(n) < 0.1).astype(int),
        }
    )


def test_enforce_phase1_partition_identity_accepts_matching_hash():
    frame = _identity_frame()
    digest = runner.describe(frame)["ordered_id_timestamp_label_sha256"]
    assert (
        runner.enforce_phase1_partition_identity(frame, digest, "TEST") == digest
    )


def test_enforce_phase1_partition_identity_fails_closed_on_mismatch():
    frame = _identity_frame()
    with pytest.raises(ValueError, match="partition identity mismatch"):
        runner.enforce_phase1_partition_identity(frame, "deadbeef" * 8, "TRAIN")


def test_phase1_sha256_constants_are_well_formed():
    for constant in (runner.PHASE1_TRAIN_SHA256, runner.PHASE1_VALIDATION_SHA256):
        assert len(constant) == 64
        int(constant, 16)  # raises if not hexadecimal


# --------------------------------------------------------------------------- #
# Phase 1 prediction alignment
# --------------------------------------------------------------------------- #


def test_validate_phase1_predictions_rejects_misaligned_ids(tmp_path):
    phase1_root = tmp_path / "phase1"
    artifact_dir = phase1_root / "reports/class_weight_phase1/phase1-20261116-rerun"
    artifact_dir.mkdir(parents=True)

    validation = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
        }
    )
    for name in ("weight_1", "current_dynamic"):
        np.savez_compressed(
            artifact_dir / f"{name}_validation_predictions.npz",
            transaction_id=np.array([1, 2, 99]),  # wrong id
            probability=np.array([0.1, 0.2, 0.3]),
        )

    with pytest.raises(ValueError, match="do not match"):
        runner.validate_phase1_predictions(phase1_root, validation)


def test_validate_phase1_predictions_accepts_aligned(tmp_path):
    phase1_root = tmp_path / "phase1"
    artifact_dir = phase1_root / "reports/class_weight_phase1/phase1-20261116-rerun"
    artifact_dir.mkdir(parents=True)

    validation = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
        }
    )
    for name in ("weight_1", "current_dynamic"):
        np.savez_compressed(
            artifact_dir / f"{name}_validation_predictions.npz",
            transaction_id=validation["TransactionID"].to_numpy(),
            transaction_dt=validation["TransactionDT"].to_numpy(),
            probability=np.array([0.1, 0.2, 0.3]),
        )

    hashes = runner.validate_phase1_predictions(phase1_root, validation)
    assert set(hashes) == {"weight_1", "current_dynamic"}
    assert all(len(value) == 64 for value in hashes.values())


# --------------------------------------------------------------------------- #
# Control flow: preflight-only must never reference `selection`,
# run bootstrap, or call confirmation.
# --------------------------------------------------------------------------- #


def _fake_args(run_id: str, preflight_only: bool):
    class Args:
        pass

    args = Args()
    args.run_id = run_id
    args.phase1_root = Path("/nonexistent-phase1")
    args.protected_root = []
    args.preflight_only = preflight_only
    return args


def test_preflight_only_does_not_reference_selection_or_bootstrap(tmp_path, monkeypatch):
    """C1 regression: --preflight-only must finish without touching `selection`.

    We monkeypatch the heavy helpers to explode if called, and drive ``main``
    through the preflight branch. If the control flow references the unbound
    ``selection`` (or runs bootstrap/confirm), the test fails loudly.
    """

    def _explode(*args, **kwargs):
        raise AssertionError("must not be called in preflight-only mode")

    def _make_output(run_id: str) -> Path:
        path = tmp_path / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    monkeypatch.setattr(runner, "create_output", _make_output)
    monkeypatch.setattr(runner, "preflight", lambda phase1_root, protected: [tmp_path])
    monkeypatch.setattr(runner, "snapshot", lambda roots: {"checkouts": {}, "files": {}})
    monkeypatch.setattr(runner, "git", lambda root, *args: "0" * 40)
    monkeypatch.setattr(runner, "run_folds", _explode)
    monkeypatch.setattr(runner, "bootstrap_comparisons", _explode)
    monkeypatch.setattr(runner, "confirm", _explode)
    monkeypatch.setattr(
        runner, "validate_phase1_predictions", lambda *a, **k: {}
    )
    # The synthetic chronology cannot match the real Phase 1 digest, so stub the
    # identity guard for this control-flow test (it has its own dedicated tests).
    monkeypatch.setattr(
        runner,
        "enforce_phase1_partition_identity",
        lambda frame, expected, label: "stubbed-digest",
    )

    # Stub the data-loading path so preflight can complete without the real data.
    chronology = pd.DataFrame(
        {
            "TransactionID": np.arange(1, 1001),
            "TransactionDT": np.sort(np.random.default_rng(0).integers(1, 1000, 1000)),
        }
    )
    monkeypatch.setattr(
        runner.pd, "read_parquet", lambda *a, **k: chronology.copy()
    )
    monkeypatch.setattr(
        runner,
        "load_partition",
        lambda chronology, preceding_rows: chronology.assign(
            isFraud=(chronology.index % 10 == 0).astype(int)
        ),
    )
    monkeypatch.setattr(
        runner.sys, "argv", ["run_tree_depth_phase2.py", "--run-id", "t", "--phase1-root", str(tmp_path), "--preflight-only"]
    )

    runner.main()

    metadata = runner.json.loads((tmp_path / "t" / "run_metadata.json").read_text())
    assert metadata["status"] == "PREFLIGHT_PASS"
    assert metadata["model_fits_started"] == 0
    assert metadata["bootstrap_called"] == 0
    assert metadata["confirmation_called"] == 0
    assert "selected_depth" not in metadata
    assert not (tmp_path / "t" / "selection.json").exists()


def test_full_run_calls_bootstrap_and_confirmation_exactly_once(tmp_path, monkeypatch):
    """C2 regression: the normal path must run bootstrap and confirmation once.

    The audited defect executed the post-selection block unconditionally, so
    ``bootstrap_comparisons`` and ``confirm`` ran twice. This test counts calls
    and asserts each is exactly one.
    """

    calls = {"bootstrap": 0, "confirm": 0}

    def _make_output(run_id: str) -> Path:
        path = tmp_path / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    monkeypatch.setattr(runner, "create_output", _make_output)
    monkeypatch.setattr(runner, "preflight", lambda phase1_root, protected: [tmp_path])
    monkeypatch.setattr(runner, "snapshot", lambda roots: {"checkouts": {}, "files": {}})
    monkeypatch.setattr(runner, "git", lambda root, *args: "0" * 40)
    monkeypatch.setattr(runner, "validate_phase1_predictions", lambda *a, **k: {})
    monkeypatch.setattr(
        runner,
        "enforce_phase1_partition_identity",
        lambda frame, expected, label: "stubbed-digest",
    )
    monkeypatch.setattr(
        runner, "run_folds", lambda train, folds, output: {"selected_depth": 4}
    )

    def _bootstrap(output, selection):
        calls["bootstrap"] += 1

    def _confirm(train, validation_chronology, selected_depth, output, phase1):
        calls["confirm"] += 1

    monkeypatch.setattr(runner, "bootstrap_comparisons", _bootstrap)
    monkeypatch.setattr(runner, "confirm", _confirm)

    chronology = pd.DataFrame(
        {
            "TransactionID": np.arange(1, 1001),
            "TransactionDT": np.sort(np.random.default_rng(0).integers(1, 1000, 1000)),
        }
    )
    monkeypatch.setattr(
        runner.pd, "read_parquet", lambda *a, **k: chronology.copy()
    )
    monkeypatch.setattr(
        runner,
        "load_partition",
        lambda chronology, preceding_rows: chronology.assign(
            isFraud=(chronology.index % 10 == 0).astype(int)
        ),
    )
    monkeypatch.setattr(
        runner.sys, "argv", ["run_tree_depth_phase2.py", "--run-id", "t", "--phase1-root", str(tmp_path)]
    )

    runner.main()

    assert calls == {"bootstrap": 1, "confirm": 1}
    metadata = runner.json.loads((tmp_path / "t" / "run_metadata.json").read_text())
    assert metadata["status"] == "PASS"
    assert metadata["bootstrap_called"] == 1
    assert metadata["confirmation_called"] == 1
    assert metadata["selected_depth"] == 4


# --------------------------------------------------------------------------- #
# Data-dependent checks (skipped when the dataset is absent)
# --------------------------------------------------------------------------- #


def _data_available() -> bool:
    return (
        (ROOT / "data/processed/train_behavioral_v1.parquet").is_file()
        and (ROOT / "data/raw/ieee_cis/train_transaction.csv").is_file()
    )


@pytest.mark.skipif(not _data_available(), reason="IEEE-CIS dataset not present")
def test_load_partition_round_trips_ids_labels_dt():
    chronology = pd.read_parquet(
        ROOT / "data/processed/train_behavioral_v1.parquet",
        columns=["TransactionID", "TransactionDT"],
    )
    chronology = chronology[chronology["TransactionDT"] <= 10452942].reset_index(drop=True)
    frame = runner.load_partition(chronology, preceding_rows=0)

    raw = pd.read_csv(
        ROOT / "data/raw/ieee_cis/train_transaction.csv",
        usecols=["TransactionID", "isFraud"],
        nrows=len(chronology),
    )
    assert np.array_equal(frame["TransactionID"].to_numpy(), raw["TransactionID"].to_numpy())
    assert np.array_equal(frame["isFraud"].to_numpy(), raw["isFraud"].to_numpy())


@pytest.mark.skipif(not _data_available(), reason="IEEE-CIS dataset not present")
def test_reconstructed_train_matches_phase1_partition_hash():
    chronology = pd.read_parquet(
        ROOT / "data/processed/train_behavioral_v1.parquet",
        columns=["TransactionID", "TransactionDT"],
    )
    timestamps = chronology["TransactionDT"].drop_duplicates().to_numpy()
    train_end = timestamps[int(len(timestamps) * 0.70) - 1]
    train_chronology = chronology[chronology["TransactionDT"] <= train_end]
    frame = runner.load_partition(train_chronology, preceding_rows=0)

    digest = runner.enforce_phase1_partition_identity(
        frame, runner.PHASE1_TRAIN_SHA256, "TRAIN"
    )
    assert digest == runner.PHASE1_TRAIN_SHA256