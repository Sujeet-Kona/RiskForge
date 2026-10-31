from __future__ import annotations

import math
from collections import defaultdict, deque

import pandas as pd


ONE_HOUR = 3_600
ONE_DAY = 86_400


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive relative temporal features from elapsed TransactionDT seconds.

    TransactionDT is treated as elapsed time, not as an absolute calendar
    timestamp. Therefore these features intentionally avoid claiming real
    calendar weekday semantics.
    """
    out = df.copy()

    transaction_dt = out["TransactionDT"]

    out["transaction_hour_bucket"] = (
        (transaction_dt // ONE_HOUR) % 24
    ).astype(int)

    out["transaction_day_index"] = (
        transaction_dt // ONE_DAY
    ).astype(int)

    out["transaction_day_mod7"] = (
        (transaction_dt // ONE_DAY) % 7
    ).astype(int)

    return out


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable log-transformed transaction amount."""
    out = df.copy()

    if (out["TransactionAmt"] < 0).any():
        raise ValueError("TransactionAmt contains negative values.")

    out["amount_log1p"] = out["TransactionAmt"].map(math.log1p)

    return out


def _rolling_history_features(
    df: pd.DataFrame,
    key_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Build strictly point-in-time rolling behavioral features.

    Important guarantees:
    - Only transactions with TransactionDT < current TransactionDT are used.
    - Transactions sharing the current TransactionDT cannot see each other.
    - The lower window boundary is exclusive.
    - Rows must already be sorted by TransactionDT.
    - Rows with incomplete behavioral keys receive zero history.
    """

    required = ["TransactionDT", "TransactionAmt", *key_columns]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for {prefix} history: {missing}"
        )

    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Input must be sorted by TransactionDT in ascending order."
        )

    result = df.copy()

    count_1h = [0] * len(result)
    count_24h = [0] * len(result)
    amount_24h = [0.0] * len(result)

    has_history_1h = [False] * len(result)
    has_history_24h = [False] * len(result)

    # Full 24-hour history for each entity.
    history_24h: dict[tuple, deque[tuple[int, float]]] = defaultdict(deque)

    # Separate 1-hour history lets us maintain the count efficiently.
    history_1h: dict[tuple, deque[tuple[int, float]]] = defaultdict(deque)

    amount_totals: dict[tuple, float] = defaultdict(float)

    transaction_dt = result["TransactionDT"].tolist()
    transaction_amt = result["TransactionAmt"].tolist()

    key_values = result[key_columns].itertuples(
        index=False,
        name=None,
    )

    rows = list(zip(transaction_dt, transaction_amt, key_values))

    start = 0

    while start < len(rows):
        timestamp = rows[start][0]

        end = start + 1

        while end < len(rows) and rows[end][0] == timestamp:
            end += 1

        # ---------------------------------------------------------------
        # Phase 1:
        # Read history for the entire timestamp group.
        #
        # No current-timestamp row is inserted yet.
        # Therefore same-timestamp transactions cannot see each other.
        # ---------------------------------------------------------------
        for row_index in range(start, end):
            _, _, key = rows[row_index]

            if any(pd.isna(value) for value in key):
                continue

            entity = tuple(key)

            events_24h = history_24h[entity]
            events_1h = history_1h[entity]

            cutoff_24h = timestamp - ONE_DAY
            cutoff_1h = timestamp - ONE_HOUR

            # Remove events at or before the 24-hour boundary.
            while events_24h and events_24h[0][0] <= cutoff_24h:
                old_timestamp, old_amount = events_24h.popleft()
                amount_totals[entity] -= old_amount

            # Remove events at or before the 1-hour boundary.
            while events_1h and events_1h[0][0] <= cutoff_1h:
                events_1h.popleft()

            count_24h[row_index] = len(events_24h)
            amount_24h[row_index] = amount_totals[entity]
            count_1h[row_index] = len(events_1h)

            has_history_24h[row_index] = len(events_24h) > 0
            has_history_1h[row_index] = len(events_1h) > 0

        # ---------------------------------------------------------------
        # Phase 2:
        # Add the current timestamp group only after every row has been
        # scored. This prevents same-timestamp leakage.
        # ---------------------------------------------------------------
        for row_index in range(start, end):
            _, amount, key = rows[row_index]

            if any(pd.isna(value) for value in key):
                continue

            entity = tuple(key)
            numeric_amount = float(amount)

            history_24h[entity].append(
                (timestamp, numeric_amount)
            )

            history_1h[entity].append(
                (timestamp, numeric_amount)
            )

            amount_totals[entity] += numeric_amount

        start = end

    result[f"{prefix}_txn_count_1h"] = count_1h
    result[f"{prefix}_txn_count_24h"] = count_24h

    result[f"{prefix}_has_history_1h"] = has_history_1h
    result[f"{prefix}_has_history_24h"] = has_history_24h

    # Floating-point subtraction can leave tiny values such as
    # -1e-12 after old events are removed. These represent zero.
    result[f"{prefix}_amount_sum_24h"] = [
        max(0.0, value) for value in amount_24h
    ]

    return result


def add_card_history(df: pd.DataFrame) -> pd.DataFrame:
    return _rolling_history_features(
        df,
        ["card1"],
        "card",
    )


def add_address_history(df: pd.DataFrame) -> pd.DataFrame:
    return _rolling_history_features(
        df,
        ["addr1", "addr2"],
        "address",
    )


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete v1 point-in-time behavioral feature set."""
    required = [
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "addr1",
        "addr2",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Input must be sorted by TransactionDT in ascending order."
        )

    out = add_temporal_features(df)
    out = add_amount_features(out)
    out = add_card_history(out)
    out = add_address_history(out)

    return out
