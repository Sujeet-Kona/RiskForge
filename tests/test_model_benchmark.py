import numpy as np
import pandas as pd

from scripts.run_model_benchmark import (
    make_test_groups,
    stratified_bootstrap,
)


def test_test_groups_keep_equal_timestamps_together():
    train = pd.DataFrame(
        {
            "TransactionAmt": np.arange(1, 101, dtype=float),
        }
    )
    test = pd.DataFrame(
        {
            "TransactionDT": [1, 1, 2, 2, 3, 3, 4, 4],
            "TransactionAmt": [10, 20, 30, 40, 50, 60, 70, 80],
            "ProductCD": ["W", "W", "C", "C", "W", "W", "H", "H"],
            "card_txn_count_24h": [0, 1, 2, 3, 0, 1, 2, 3],
        }
    )

    groups = make_test_groups(test, train)

    assert len(groups) == len(test)
    for timestamp in test["TransactionDT"].unique():
        labels = groups.loc[
            test["TransactionDT"] == timestamp, "time_period"
        ].unique()
        assert len(labels) == 1
    assert set(groups["card_history_24h"]) == {
        "no_prior_24h",
        "1_to_2_prior_24h",
        "3plus_prior_24h",
    }


def test_stratified_bootstrap_is_reproducible_and_reports_paired_deltas():
    y = np.array([0] * 80 + [1] * 20)
    logistic = np.linspace(0.01, 0.8, len(y))
    xgboost = np.clip(logistic + np.linspace(0.0, 0.1, len(y)), 0, 1)
    predictions = {
        "logistic_regression": logistic,
        "xgboost": xgboost,
    }

    first = stratified_bootstrap(
        y,
        predictions,
        replicates=100,
        seed=17,
    )
    second = stratified_bootstrap(
        y,
        predictions,
        replicates=100,
        seed=17,
    )

    pd.testing.assert_frame_equal(first, second)
    assert "xgboost_minus_logistic_regression" in set(first["comparison"])
    assert (first["ci_95_low"] <= first["ci_95_high"]).all()
