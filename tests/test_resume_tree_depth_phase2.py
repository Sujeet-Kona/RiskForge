"""Tests for the Phase 2 resume and optimized bootstrap path."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import resume_tree_depth_phase2 as resume
from scripts import run_tree_depth_phase2 as runner


def test_precomputed_sampler_matches_existing_helper_exactly():
    timestamps = np.repeat(np.arange(120, dtype=np.int64) * 3600, 2)
    unique, groups = resume.precompute_timestamp_groups(timestamps)

    expected_rng = np.random.default_rng(20261116)
    actual_rng = np.random.default_rng(20261116)
    for _ in range(25):
        expected = runner.moving_time_block_indices(
            pd.Series(timestamps),
            expected_rng,
            block_duration_seconds=86400,
        )
        actual = resume.moving_block_indices_precomputed(
            unique,
            groups,
            actual_rng,
            block_duration_seconds=86400,
        )
        assert np.array_equal(actual, expected)


def test_precomputed_groups_keep_timestamp_ties_together():
    timestamps = np.repeat(np.arange(100, dtype=np.int64) * 3600, 3)
    unique, groups = resume.precompute_timestamp_groups(timestamps)

    assert len(unique) == 100
    assert len(groups) == 100
    assert all(len(group) == 3 for group in groups)
    sampled = resume.moving_block_indices_precomputed(
        unique, groups, np.random.default_rng(7)
    )
    for timestamp in np.unique(timestamps[sampled]):
        positions = np.flatnonzero(timestamps[sampled] == timestamp)
        assert len(positions) % 3 == 0


def test_precomputed_groups_reject_unsorted_or_missing_timestamps():
    with np.testing.assert_raises_regex(ValueError, "sorted"):
        resume.precompute_timestamp_groups(np.array([1, 3, 2]))
    with np.testing.assert_raises_regex(ValueError, "nonmissing"):
        resume.precompute_timestamp_groups(np.array([1.0, np.nan, 3.0]))


def test_resume_bootstrap_writes_paired_diagnostics(tmp_path):
    n = 240
    timestamps = np.arange(n, dtype=np.int64) * 3600
    labels = (np.arange(n) % 13 == 0).astype(np.int8)
    base = np.linspace(0.01, 0.8, n)
    for depth, offset in ((3, 0.0), (4, 0.02), (6, -0.01)):
        np.savez_compressed(
            tmp_path / f"depth_{depth}_oof.npz",
            transaction_dt=timestamps,
            label=labels,
            probability=np.clip(base + offset, 0.001, 0.999),
        )

    selection = {
        "selected_depth": 3,
        "ranking": [{"depth": 3}, {"depth": 4}, {"depth": 6}, {"depth": 5}],
    }
    result = resume.run_bootstrap(tmp_path, tmp_path, selection, repeats=20)

    assert [row["reference_depth"] for row in result["comparisons"]] == [4, 6]
    assert all(row["valid_replicates"] >= 18 for row in result["comparisons"])
    assert all(row["requested_replicates"] == 20 for row in result["comparisons"])
    assert (tmp_path / "bootstrap_depth_3_minus_4.npz").is_file()
    assert (tmp_path / "bootstrap_depth_3_minus_6.npz").is_file()
    assert (tmp_path / "oof_bootstrap_diagnostics.json").is_file()
