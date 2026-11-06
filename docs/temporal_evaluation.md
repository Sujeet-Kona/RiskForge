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
  --block-length 50 \
  --bootstrap-replicates 500
```

The script uses the existing 28-feature point-in-time modeling frame. It fits
fixed XGBoost and Random Forest candidates on an expanding history and scores
the next chronological window. Boundaries are defined over distinct
`TransactionDT` values, so tied timestamps remain together.

It also reruns the locked TRAIN/TEST split and computes paired moving-time-block
bootstrap intervals for XGBoost minus Random Forest on TEST. Sampling units are
complete timestamp groups; contiguous blocks preserve local order and avoid
splitting simultaneous transactions.

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

## Interpretation

Walk-forward windows are repeated historical simulations on the same dataset,
not independent datasets. The expanding-window result is sensitive to the
chosen start fraction and window sizes. The block-bootstrap interval is
conditional on the fixed TEST holdout and chosen number of timestamp groups per
block; it is not a guarantee about future periods.

The existing decision-policy script already reports sensitivity to assumed
review effectiveness and review-volume caps. The calibration script already
prints reliability tables for raw VALIDATION, forward-scored calibrated
VALIDATION, raw TEST and calibrated TEST. The benchmark already reports
descriptive TEST slices by time, amount, ProductCD and card-history group.
Those existing evaluations are intentionally retained rather than duplicated.
