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
forward-chaining validation protocol. The output from
`scripts/select_decision_policy.py` is an experimental candidate, stamped
`policy_version = candidate`; it does not modify or replace the frozen serving
policy. The active serving policy remains `policy-001` with
`T_REVIEW = 0.075` and `T_BLOCK = 0.110`.

A candidate must not be promoted automatically because it has the lowest cost
under the project's synthetic assumptions. Promotion requires a separate,
documented review of the assumptions, threshold stability, calibration-scale
mismatch, operational constraints and held-out evaluation. Only after that
review should the locked policy version, constants, tests and serving artifact
be changed together. Until then, keep `policy-001` as the active serving policy
and do not describe its thresholds as revalidated under the corrected protocol.

The corrected protocol was rerun on the local IEEE-CIS dataset. Across
70,334 forward-chaining OOF VALIDATION rows (17,679 warm-up rows excluded):

| Method | Brier (selection metric) | Log loss | ECE | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sigmoid | 0.025368 | 0.104336 | 0.010320 | 0.426233 | 0.876543 |
| Isotonic | 0.024703 | 0.108550 | 0.004678 | 0.422378 | 0.874798 |

Isotonic was selected because Brier score is the predeclared method-selection
criterion and isotonic achieved the lower OOF Brier score. This is not a claim
that isotonic wins every metric: sigmoid had lower OOF log loss and higher
ranking metrics. The selection criterion must be stated when presenting this
result.

The selected isotonic calibrator was refitted on all VALIDATION rows for the
serving artifact. On the 88,081-row held-out TEST partition (3,065 fraud cases),
the calibrated results were PR-AUC 0.438919, ROC-AUC 0.864850, Brier 0.024204,
log loss 0.102851 and ECE 0.003553. Raw XGBoost TEST PR-AUC and ROC-AUC were
0.449254 and 0.865318 respectively. These are benchmark results, not estimates
of future payment-network performance.

The serving-artifact builder therefore continues to use isotonic calibration,
consistent with the recorded OOF Brier selection. The active serving policy
remains `policy-001`; the experimental candidate threshold search is separate
and has not been promoted. Full results and the threshold-selection caveat are
recorded in [model evaluation](model-evaluation.md).

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
