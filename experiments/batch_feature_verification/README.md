# Batch-relative feature verification

This supplementary paired experiment holds HistGB parameters, the other features and user folds fixed, and toggles the six batch-relative features.

| Configuration | Mean five-fold MAP@200 |
|---|---:|
| Without batch-relative features | 0.726710 |
| With batch-relative features | 0.767775 |
| Paired improvement | +0.041064 |

The feature block improves all five folds. All 15,220 candidate identities and fold assignments match the published four-model comparison. AP@200 was independently recalculated from the saved predictions and matches both means.

This is a separate supplementary run on macOS, with Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.6.1 and one CPU thread. The four-model comparison retains its original HistGB score of 0.767729. The difference between full-model runs has not been attributed to a specific cause. Compute the feature gain using the two results in this directory.

## Records

- `results.json`: full precision means, fold scores, parameters, source hashes and recorded dependency versions.
- `folds.csv`: user-event pairs and fold IDs.
- `without_batch_fold_scores.csv` and `with_batch_fold_scores.csv`: five fold scores for each configuration.
- `without_batch_oof.csv` and `with_batch_oof.csv`: held-out predictions and labels.
- `compare.py`: AI-assisted paired experiment wrapper; project source modules are unchanged.

## Run

From the repository root, use a Python 3.12 environment and install `experiments/batch_feature_verification/requirements.txt`. Put the competition data in `data/raw/` following the root README, then run:

```text
python experiments/batch_feature_verification/compare.py
```

The script accepts uncompressed CSV files or `.csv.gz` for the three large data tables. New results are written to `rerun_results/`, preserving the published records. Compared with the original local wrapper, the packaged script only changes input and output paths and supports both compressed and uncompressed data.
