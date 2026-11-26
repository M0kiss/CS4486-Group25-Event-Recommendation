# CS4486 Group 25 — Event Recommendation

This project ranks candidate events for each user in the [Kaggle Event Recommendation Engine Challenge](https://www.kaggle.com/competitions/event-recommendation-engine-challenge). A histogram-based gradient boosting classifier predicts interest probabilities, which are used to order each user's candidate events. Evaluation uses MAP@200 with five disjoint user folds.

## Team

| Member | Main responsibilities |
|---|---|
| Li Siyuan | Dataset research, feature extraction and model construction |
| Xue Jinyuan | Dataset research, preprocessing and report preparation |
| Xie Xu | Model comparison, parameter optimization and presentation |

## Method

- Retain the earliest exposure for each user–event pair: 15,398 training records become 15,220 pairs covering 2,034 users.
- Combine contextual features, social response statistics, one-hot encoded content-cluster membership, user interest profiles and batch-relative features.
- Apply log compression and L2 normalization to the 101-dimensional content vectors before clustering and profile construction. KMeans `predict()` assigns a cluster ID, which is encoded into 20 one-hot columns for the classifier.
- Fit content clustering and locale encoding within each training fold.
- Train HistGradientBoostingClassifier and rank validation candidates by predicted interest, breaking score ties by event ID.
- Compute AP@200 per user, average within each fold, and report the mean across five folds. The split and model seed is 25.

## Repository layout

The project code is in `src/`; saved model comparison records are in `experiments/model_comparison/`.

```text
README.md
requirements.txt
src/
    data.py
    feature.py
    model.py
    validation.py
    interest_profile.py
    geography.py
experiments/
    model_comparison/    # Comparison script, settings and saved results
data/
    raw/                 # Locally downloaded CSV files; not included in Git
```

## Data preparation

Obtain the dataset from the competition's Data page using your Kaggle account and follow its access terms. Extract these files into `data/raw/`:

```text
train.csv
users.csv
events.csv
event_attendees.csv
user_friends.csv
```

These five files are required by the cross-validation entry point. The competition test file is not needed for this evaluation. Keep raw data outside the uploaded repository.

## Environment and installation

The team's main development environment is Windows 11 Home Chinese Edition (Build 26300), x86_64 (AMD64), with Python 3.12.14. The supplied package versions are NumPy 2.3.5, pandas 3.0.1 and scikit-learn 1.6.1. Training uses the CPU; GPU/CUDA is not used.

Create and activate an isolated environment in PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run cross-validation

From the repository root, with the environment activated:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
python src/data.py
```

The program prints preprocessing progress, each fold's MAP@200 and the five-fold mean. The current entry point computes fold results and out-of-fold predictions in memory; it does not automatically save a model, predictions or a Kaggle submission file.

## Model settings

| Setting | Value |
|---|---:|
| Boosting iterations | 300 |
| Learning rate | 0.05 |
| Maximum leaf nodes | 31 |
| Minimum samples per leaf | 60 |
| L2 regularization | 10 |
| Early stopping | Disabled |
| Content clusters | 20 |
| KMeans initializations | 10 |
| KMeans maximum iterations | 300 |
| Random seed | 25 |
| Advanced geographic matching | Disabled (`advancedgeo_on=False`) |

Settings and feature switches are defined near the beginning of `src/data.py`.

## Results reported in the final report

The final HistGB mean five-fold MAP@200 is **0.767729**, shown as **0.7677** in the presentation. The result is recorded in [summary.csv](experiments/model_comparison/summary.csv), with individual fold scores in [fold_scores.csv](experiments/model_comparison/fold_scores.csv).

The four-model comparison uses the same candidate pairs, user folds and feature matrix within each fold:

| Model | Mean five-fold MAP@200 | Sample fold SD |
|---|---:|---:|
| HistGradientBoosting | **0.767729** | 0.009476 |
| Random Forest | 0.764186 | 0.011295 |
| RF+LR reference adaptation | 0.748328 | 0.014215 |
| MLP | 0.666460 | 0.016109 |

HistGB scored higher than Random Forest in three of five folds and higher than the RF+LR adaptation in all five folds. The adaptation uses the project features with reference weights of 0.69 for RF and 0.57 for LR. The MLP used hidden layers (64, 32) and reached its 150-iteration limit in every fold.

These results come from the team's `model_comparison_report_package_20261126`. The [comparison directory](experiments/model_comparison/README.md) provides the script, original configuration, fold assignments and held-out predictions. Its README explains how to run the four-model comparison with its recorded dependencies. The main `python src/data.py` command runs HistGB cross-validation using the main project environment described above. All reported scores are cross-validation MAP@200.

## Implementation index

| Report topic | Source and function |
|---|---|
| Candidate deduplication | `src/data.py`: `load_candidate()` |
| User and event metadata | `src/data.py`: `read_userfeatures()`, `read_eventfeatures()` |
| User folds | `src/validation.py`: `make_fold()` |
| AP@200 | `src/validation.py`: `ap()` |
| Content normalization and clustering | `src/feature.py`: `content_feature()`, `cluster_feature()` |
| Social responses | `src/feature.py`: `response_feature()`, `friend_response()` |
| Interest profiles | `src/interest_profile.py`: `build_profile_features()` |
| Batch-relative features | `src/feature.py`: `query_relative_feature()` |
| Training and validation | `src/model.py`: `run_cv()` |
| Four-model comparison and results | [Comparison materials](experiments/model_comparison/README.md) |

Attendance and friendship data are snapshots. User separation prevents overlap between training and validation users, while snapshot timing remains a limitation of the evaluation.

## References and AI assistance

- [Kaggle competition](https://www.kaggle.com/competitions/event-recommendation-engine-challenge)
- [HistGradientBoostingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [KMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Andrei Olariu's RF+LR reference implementation](https://github.com/andreiolariu/kaggle-event-recommendation/blob/master/model.py)

OpenAI Codex supported technical explanations, debugging, report drafting and editing, layout, and supplementary comparison and verification scripts. Member responsibilities and detailed contributions are described in the final report.
