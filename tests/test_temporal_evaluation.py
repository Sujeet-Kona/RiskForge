import numpy as np
import pandas as pd
import pytest

from scripts.run_temporal_evaluation import (
    expanding_window_slices,
    moving_time_block_indices,
    time_block_bootstrap,
)


def test_expanding_windows_are_strictly_forward_and_keep_timestamp_ties():
    timestamps = pd.Series(np.repeat(np.arange(100), 3))
    folds = expanding_window_slices(
        timestamps, n_splits=4, test_fraction=0.10
    )

    assert len(folds) == 4
    previous_train_size = 0
    for train_rows, test_rows in folds:
        train_times = timestamps.iloc[train_rows]
        test_times = timestamps.iloc[test_rows]
        assert train_times.max() < test_times.min()
        assert len(train_rows) > previous_train_size
        previous_train_size = len(train_rows)
        assert set(train_times).isdisjoint(set(test_times))
        for timestamp in test_times.unique():
            assert (test_times == timestamp).sum() == 3


def test_expanding_windows_reject_unsorted_timestamps():
    with pytest.raises(ValueError, match="sorted in ascending"):
        expanding_window_slices(pd.Series([1, 3, 2, 4]), n_splits=2)


def test_expanding_windows_reject_invalid_fold_configuration():
    with pytest.raises(ValueError, match="at least 2"):
        expanding_window_slices(pd.Series(range(20)), n_splits=1)
    with pytest.raises(ValueError, match="below 0.5"):
        expanding_window_slices(pd.Series(range(20)), n_splits=4, test_fraction=0.2)


def test_moving_block_sample_preserves_complete_timestamp_groups_and_order():
    timestamps = pd.Series(np.repeat(np.arange(30), [1, 2, 3] * 10))
    indices = moving_time_block_indices(
        timestamps, np.random.default_rng(5), block_length=4
    )
    sampled_times = timestamps.iloc[indices].to_numpy()

    assert len(sampled_times) >= timestamps.nunique()
    for timestamp in np.unique(sampled_times):
        sampled_count = int((sampled_times == timestamp).sum())
        original_count = int((timestamps == timestamp).sum())
        assert sampled_count % original_count == 0

    run_start = 0
    for position in range(1, len(sampled_times) + 1):
        if position == len(sampled_times) or sampled_times[position] != sampled_times[run_start]:
            run_length = position - run_start
            timestamp = sampled_times[run_start]
            original_count = int((timestamps == timestamp).sum())
            assert run_length % original_count == 0
            run_start = position


def test_time_block_bootstrap_is_reproducible_and_paired():
    timestamps = pd.Series(np.repeat(np.arange(200), 2))
    y = np.tile([0, 1], 200)
    base = np.linspace(0.01, 0.9, len(y))
    predictions = {
        "xgboost": np.clip(base + 0.02, 0, 1),
        "random_forest": base,
    }
    kwargs = {
        "replicates": 100,
        "block_length": 10,
        "seed": 23,
    }
    first = time_block_bootstrap(y, timestamps, predictions, **kwargs)
    second = time_block_bootstrap(y, timestamps, predictions, **kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {"average_precision", "roc_auc"}
    assert (first["ci_95_low"] <= first["ci_95_high"]).all()


def test_time_block_bootstrap_rejects_bad_inputs():
    timestamps = pd.Series(np.repeat(np.arange(20), 2))
    y = np.tile([0, 1], 20)
    scores = np.linspace(0.0, 1.0, len(y))
    with pytest.raises(ValueError, match="both classes"):
        time_block_bootstrap(
            np.zeros_like(y), timestamps,
            {"xgboost": scores, "random_forest": scores},
            replicates=100, block_length=5,
        )
