# Temporal robustness evaluation

This evaluation is an additional experiment. It does not replace the locked
chronological 70/15/15 benchmark, its saved metrics, the selected calibrator,
or the serving policy.

## Run

After following the dataset and environment setup in this document:

```bash
PYTHONPATH=. python scripts/run_temporal_evaluation.py
```

Optional controls:

```bash
PYTHONPATH=. python scripts/run_temporal_evaluation.py \
  --folds 4 \
  --test-fraction 0.10 \
  --block-duration-seconds 86400 \
  --bootstrap-replicates 500
```

The script uses the existing 28-feature point-in-time modeling frame. It fits
fixed XGBoost and Random Forest candidates on an expanding history and scores
the next chronological window. Boundaries are defined over distinct
`TransactionDT` values, so tied timestamps remain together.

It also reruns the locked TRAIN/TEST split and computes paired moving-time-block
bootstrap intervals for XGBoost minus Random Forest on TEST. Sampling units are
complete timestamp groups. Contiguous blocks span the configured
`TransactionDT` duration, preserve local order, and never split simultaneous
transactions.

## Outputs

- `reports/temporal_robustness/walk_forward_folds.csv`: metrics and sample
  sizes for every model/fold.
- `reports/temporal_robustness/walk_forward_summary.csv`: mean, standard
  deviation, minimum and maximum across folds.
- `reports/temporal_robustness/test_time_block_bootstrap.csv`: paired
  percentile intervals for Average Precision and ROC-AUC differences.
- `reports/temporal_robustness/test_calibration_segments.csv`: raw and
  isotonic-calibrated probability quality by chronological and amount segment.
- `reports/temporal_robustness/test_calibration_reliability_bins.csv`: bin-level
  predicted-versus-observed rates for reliability plots.
- `reports/temporal_robustness/run_metadata.json`: fold, block and seed
  configuration plus interpretation boundaries.

## Relationship to the locked benchmark

The walk-forward folds and the locked 70/15/15 benchmark are **different
experiments with different training windows**. Their numbers must not be
compared as though they measured the same thing.

The expanding window starts at 60% of distinct timestamps and grows by one
10% test window per fold. The locked benchmark trains on exactly 414,446 rows
and selects the calibrator and thresholds on the next 88,013. Those two
schedules diverge from fold 3 onward:

| Fold | `train_rows` | Relationship to the locked partitions |
| --- | ---: | --- |
| 1 | 355,520 | within locked TRAIN only |
| 2 | 414,446 | within locked TRAIN only — identical row count to locked TRAIN |
| 3 | 473,187 | extends 58,741 rows into the locked VALIDATION period |
| 4 | 531,878 | extends 117,432 rows past locked TRAIN, including **29,419 rows of the locked TEST partition** (33.4% of its 88,081 rows) |

So fold 4 is *not* an out-of-sample estimate with respect to the locked TEST
set: a third of that set is inside its training window. Fold 4's Average
Precision is therefore not comparable to the locked TEST figure of 0.449254,
and the walk-forward mean across folds mixes four different train/test
relationships.

This is a property of expanding-window evaluation on a single finite dataset,
not a defect in the implementation — but it is the reason the walk-forward
results are reported separately and never substituted for the locked metrics.

## Evidence status

The five files under `reports/temporal_robustness/` are **historical generated
artifacts**. They were produced by `scripts/run_temporal_evaluation.py` against
the gitignored IEEE-CIS inputs; regenerating them requires that dataset and
refits both candidates on every fold.

What has been verified without re-running:

- **Configuration agreement.** `run_metadata.json` states 4 splits, a 0.1 test
  fraction, a 0.6 initial train fraction, 500 bootstrap replicates, an 86,400
  second block duration, and seed 20261107. These match the code defaults at
  `scripts/run_temporal_evaluation.py:35-38` exactly.
- **Expanding-window arithmetic.** The folds are self-consistent: the four
  `test_rows` values sum to 235,020, and 590,540 − 235,020 = 355,520, which is
  fold 1's `train_rows`. Each subsequent fold's `train_rows` equals the
  previous fold's `train_rows + test_rows` (414,446 → 473,187 → 531,878). No
  fold contradicts another.
- **Strict-forward ordering.** In every fold `train_end_transaction_dt` is less
  than `test_start_transaction_dt`, so no fold trains on data from its own test
  window.

What has **not** been verified: that re-running the script today reproduces
these exact floats. That requires the dataset and is an open reproduction task,
not a completed one.

## Interpretation

Walk-forward windows are repeated historical simulations on the same dataset,
not independent datasets. The expanding-window result is sensitive to the
chosen start fraction and window sizes. The block-bootstrap interval is
conditional on the fixed TEST holdout and chosen elapsed-time duration per block; it is not a guarantee about future
periods.

The existing decision-policy script already reports sensitivity to assumed
review effectiveness and review-volume caps. The calibration script already
prints reliability tables for raw VALIDATION, forward-scored calibrated
VALIDATION, raw TEST and calibrated TEST. The benchmark already reports
descriptive TEST slices by time, amount, ProductCD and card-history group.
Those existing evaluations are intentionally retained rather than duplicated.
