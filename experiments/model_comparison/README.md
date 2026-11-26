# Four-model comparison

This directory contains the saved experiment records used in the Group 25 final report. The source package is `model_comparison_report_package_20261126`, dated November 26, 2026.

## Results

| Model | Mean MAP@200 | Sample fold SD |
|---|---:|---:|
| HistGradientBoosting | **0.767729** | 0.009476 |
| Random Forest | 0.764186 | 0.011295 |
| RF+LR reference adaptation | 0.748328 | 0.014215 |
| MLP | 0.666460 | 0.016109 |

The reported mean is the arithmetic mean of five user-disjoint fold scores. Each fold averages AP@200 over its users, including zero for users without positive labels. `user_weighted_map200` in the summary instead averages all users across folds.

All models use the same candidate pairs, user folds and fold-specific features. RF uses training-fold median imputation. LR and MLP additionally use training-fold standardization. The RF+LR ranking score is `0.69 * RF + 0.57 * LR`, using published combination weights with the Group 25 features. MLP reached its 150-iteration limit in every fold; warnings are preserved in the fold records.

## Files

| File | Contents |
|---|---|
| [summary.csv](summary.csv) | Mean MAP, sample fold SD, timing and user-weighted MAP |
| [fold_scores.csv](fold_scores.csv) | Model scores, timings and warnings for each fold |
| [config.json](config.json) | Original model settings, dependency versions and source hashes |
| [folds.csv](folds.csv) | Candidate identities and user-grouped fold assignments |
| [oof.csv](oof.csv) | Held-out candidate scores, labels and folds |
| [user_scores.csv](user_scores.csv) | AP@200 for each user and model |
| `feature_columns_fold_0.json` through `feature_columns_fold_4.json` | Feature columns used in each fold |
| [compare.py](compare.py) | Comparison entry point |
| [requirements.txt](requirements.txt) | Dependency snapshot supplied with the comparison package |

The saved records are copied unchanged from the supplied package. The six source hashes in `config.json` match the repository's `src/` files. For repository packaging, `compare.py` now reads those modules from `src/` and writes new runs to `rerun_results/`. Model settings and evaluation logic are unchanged.

## Environment

The main project's development environment is described in the root README. This comparison package has its own recorded environment in `config.json` and dependency snapshot in `requirements.txt`. These original records are preserved. The comparison script uses two CPU threads.

## Run the comparison

Place the five competition CSV files in `data/raw/` as described in the root README. Use a separate environment for the comparison dependencies. From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv-comparison
.\.venv-comparison\Scripts\Activate.ps1
python -m pip install -r experiments/model_comparison/requirements.txt
python experiments/model_comparison/compare.py
```

The saved package records Python 3.13.15. The command above requires Python 3.13 to be installed. A new run writes to `experiments/model_comparison/rerun_results/`, leaving the reported records intact.

The comparison harness was prepared with AI assistance. Team responsibilities and tool usage are described in the final report.
