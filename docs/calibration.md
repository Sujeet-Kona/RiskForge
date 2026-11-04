# Calibration validation methodology

RiskForge separates raw model ranking from probability calibration. The
classifier is fitted on the chronological TRAIN partition; candidate
calibration methods are compared using raw model scores and labels from
VALIDATION only. TEST is reserved for the final, single evaluation.

## Forward-chaining validation

Calibration cross-validation uses expanding chronological windows:

1. Split VALIDATION into ordered folds while keeping equal `TransactionDT`
   values together.
2. Treat fold 0 as a warm-up period. It has no earlier calibration observations,
   so it is not assigned out-of-fold calibrated probabilities.
3. For each fold from 1 onward, fit the calibrator using only rows in earlier
   folds and score the current fold.
4. Never fit a calibrator for an earlier fold using labels from a later fold.

The returned out-of-fold probability series therefore contains `NaN` for the
initial warm-up fold. Calibration metrics, reliability tables, ranking metrics,
and policy-threshold searches must use only rows with a non-null out-of-fold
probability. Comparing metrics over the full validation set while retaining
warm-up rows would be invalid.

This is expanding-window validation, not shuffled K-fold and not leave-one-fold-
out validation. It reflects the intended temporal deployment direction: past
labels may calibrate later observations, but future labels cannot calibrate
earlier observations.

## Deployment refit and remaining limitation

After method selection, the selected calibrator is refitted on all VALIDATION
rows for the frozen serving artifact. The TEST partition is then transformed
with that one final calibrator.

Threshold selection uses forward-scored validation rows, while the final
serving calibrator is fitted on all validation rows. These are not identical
calibration procedures; consequently, the validation threshold-search cost is
not an unbiased estimate of deployed policy cost. The frozen TEST evaluation is
the final held-out check for the resulting model/calibrator/policy combination.

The thresholds currently stored in `policy/locked.py` predate this corrected
forward-chaining validation protocol. They must be compared with the new output
from `scripts/select_decision_policy.py` on the local IEEE-CIS data. If the
selected thresholds change, update the locked policy constants, policy version,
related tests and serving artifact together; do not silently keep the old lock
and describe it as newly validated.

The locked serving-artifact builder currently specifies isotonic calibration.
After changing the validation protocol, rerun
`PYTHONPATH=. python scripts/evaluate_calibration.py` on the local IEEE-CIS
dataset and inspect the new method-selection result before treating isotonic as
the empirically preferred method. Do not infer that the previous calibration
metrics or threshold results remain unchanged.

## Reproduction

The dataset and generated artifacts are local and gitignored. With the required
IEEE-CIS files and generated behavioral parquet present:

```bash
PYTHONPATH=. python scripts/evaluate_calibration.py
PYTHONPATH=. python scripts/select_decision_policy.py
PYTHONPATH=. python scripts/build_serving_artifact.py
```

The unit tests use synthetic chronological data to verify the invariants:
same-timestamp groups remain intact, each scored fold is independent of its own
labels, future-fold labels cannot affect earlier scored predictions, and the
warm-up fold remains unscored.
