"""Request payloads to the locked 28-feature contract.

This is the only place the serving path builds features, and it does not
implement any feature logic of its own: it assembles a frame and hands it to
:func:`ml.features.behavioral.build_behavioral_features`, the same function that
produced the offline training artifact. One inference path, therefore no
training/serving skew.

The point-in-time guarantee is inherited rather than reimplemented. That
function only reads rows with ``TransactionDT < current TransactionDT`` and
scores a whole timestamp group before inserting any of it, so same-timestamp rows
cannot see each other. The premise it needs — a frame whose every historical row
is strictly prior — is asserted here and rejected at the schema layer, never
repaired by filtering.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from app.schemas import HistoryTransaction, TransactionPayload
from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
)
from ml.features.behavioral import build_behavioral_features


# The complete input to build_behavioral_features. Nothing else about a
# historical transaction contributes to any feature of the current one.
BEHAVIORAL_INPUT_COLUMNS = (
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "addr2",
)

# The offline modeling frame's dtypes, restated so the serving frame matches it
# by construction. pandas infers ``object`` for a column whose single value is
# None, where the offline frame carries ``str`` or ``float64``; the fitted
# preprocessor currently tolerates both, but tolerating a difference is not the
# same as not having one. ``astype("str")`` preserves NaN in pandas 3.x rather
# than stringifying it, so gaps stay gaps.
#
# ``card1`` is int64 offline and float64 here, because history rows may omit it
# and a nullable integer column would be the only way to keep int64. The
# behavioral features are counts and sums keyed on equality, and int/float keys
# of the same value group identically, so the features are unchanged.
SERVING_DTYPES: dict[str, str] = {
    "TransactionDT": "int64",
    "TransactionAmt": "float64",
    "card1": "float64",
    "addr1": "float64",
    "addr2": "float64",
    **{feature: "str" for feature in CATEGORICAL_FEATURES},
    **{feature: "float64" for feature in CD_FEATURES},
}


def build_serving_features(
    transaction: TransactionPayload,
    history: Sequence[HistoryTransaction],
) -> pd.DataFrame:
    """Return the single-row frame of the 28 locked model features."""
    _reject_non_prior_history(transaction, history)

    combined = _combined_frame(transaction, history)

    behavioral = build_behavioral_features(combined)

    current = _current_row(behavioral, transaction)

    return current[MODEL_FEATURES]


def _reject_non_prior_history(
    transaction: TransactionPayload,
    history: Sequence[HistoryTransaction],
) -> None:
    """Enforce ``history.TransactionDT < transaction.TransactionDT``.

    The schema already rejects this, and the check is kept here because the
    strict prior-time rule is the project's central leakage invariant: a
    same-timestamp row sorted ahead of the current one would be counted as its
    history, and the resulting score would look perfectly ordinary.
    """
    current = transaction.TransactionDT

    for position, entry in enumerate(history):
        if entry.TransactionDT >= current:
            raise ValueError(
                f"history[{position}].TransactionDT={entry.TransactionDT} must be "
                f"strictly less than transaction.TransactionDT={current}."
            )


def _combined_frame(
    transaction: TransactionPayload,
    history: Sequence[HistoryTransaction],
) -> pd.DataFrame:
    """History plus the current transaction, sorted by elapsed time.

    The current transaction has the strict maximum ``TransactionDT``, so a stable
    sort puts it last and it forms its own timestamp group. That is exactly the
    offline arrangement: every row it can see is genuinely prior.
    """
    rows = [
        {
            "TransactionDT": entry.TransactionDT,
            "TransactionAmt": entry.TransactionAmt,
            "card1": entry.card1,
            "addr1": entry.addr1,
            "addr2": entry.addr2,
        }
        for entry in history
    ]

    rows.append(
        {
            "TransactionDT": transaction.TransactionDT,
            "TransactionAmt": transaction.TransactionAmt,
            "card1": transaction.card1,
            "addr1": transaction.addr1,
            "addr2": transaction.addr2,
        }
    )

    frame = pd.DataFrame(rows, columns=list(BEHAVIORAL_INPUT_COLUMNS))
    frame = frame.astype(
        {column: SERVING_DTYPES[column] for column in BEHAVIORAL_INPUT_COLUMNS}
    )

    return frame.sort_values("TransactionDT", kind="stable").reset_index(drop=True)


def _current_row(
    behavioral: pd.DataFrame,
    transaction: TransactionPayload,
) -> pd.DataFrame:
    """The scored transaction's own row, with its categorical and C/D values.

    The behavioral columns are already computed; the categorical and C/D values
    are properties of this transaction alone and play no part in any historical
    window, so they are attached only to the row being scored.
    """
    payload = {
        feature: getattr(transaction, feature)
        for feature in (*CATEGORICAL_FEATURES, *CD_FEATURES)
    }

    current = behavioral.iloc[[-1]].reset_index(drop=True).assign(**payload)

    return current.astype(
        {
            feature: SERVING_DTYPES[feature]
            for feature in (*CATEGORICAL_FEATURES, *CD_FEATURES)
        }
    )
