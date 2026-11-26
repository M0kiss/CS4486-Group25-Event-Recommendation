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
- Combine contextual features, social response statistics, normalized content features, content-cluster distances, user interest profiles and batch-relative features.
- Fit content clustering and locale encoding within each training fold.
- Train HistGradientBoostingClassifier and rank validation candidates by predicted interest, breaking score ties by event ID.
- Compute AP@200 per user, average within each fold, and report the mean across five folds. The split and model seed is 25.

## Repository layout

The following layout is used by the run instructions. Place the six project Python modules inside `src/`.

```text
README.md
src/
    data.py
    feature.py
    model.py
    validation.py
    interest_profile.py
    geography.py
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
python -m pip install numpy==2.3.5 pandas==3.0.1 scikit-learn==1.6.1
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

Settings and feature switches are defined near the beginning of `src/data.py`.

## Results reported in the final report

| Configuration | Mean five-fold MAP@200 |
|---|---:|
| B0: Context baseline | 0.687183 |
| B1: Context + social/content | 0.728832 |
| B2: B1 + profiles | 0.728242 |
| F: B2 + batch features | 0.768746 |
| A: Full model without profiles | 0.766471 |

These values are the saved controlled-comparison results used in the report. The main entry point runs the full model. The comparative configurations were evaluated with a supplementary experiment wrapper; they are not all executed by the command above. Scores are cross-validation measurements, not Kaggle leaderboard scores.

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

Attendance and friendship data are snapshots. User separation prevents overlap between training and validation users, while snapshot timing remains a limitation of the evaluation.

## References and AI assistance

- [Kaggle competition](https://www.kaggle.com/competitions/event-recommendation-engine-challenge)
- [HistGradientBoostingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [KMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

OpenAI Codex supported technical explanations, debugging, report drafting and editing, layout, and supplementary comparison and verification scripts. Member responsibilities and detailed contributions are described in the final report.
