"""Serving configuration for the Phase 6 inference API.

Only values the API itself owns live here. The frozen model, feature and policy
identifiers come from the artifact and from :mod:`policy.locked`, so nothing in
this module can drift from what was actually built offline.
"""

from __future__ import annotations

from pathlib import Path

from ml.models.artifact import ARTIFACT_FILENAME


# Relative to the repository root, matching the existing scripts' convention.
DEFAULT_ARTIFACT_PATH = Path("artifacts/serving") / ARTIFACT_FILENAME

DEFAULT_TOP_K = 3

# The largest behavioral window is 24 hours, so a request only ever needs prior
# transactions of its own two entities from inside that window. Measured over
# all 590,540 real IEEE-CIS rows, the most any single transaction needs is
# card_txn_count_24h + address_txn_count_24h = 1864 rows, and no row needs more
# than 2000. The bound is therefore not a truncation risk for real data.
#
# Requests above it are rejected rather than trimmed: silently dropping history
# would change the behavioral features and produce a score that no longer
# matches the offline pipeline for the same transaction.
MAX_HISTORY_TRANSACTIONS = 2000

STATELESS_NOTE = (
    "Phase 6 is stateless. Persistent transaction history is introduced in "
    "Phase 7."
)
