# Video Game Player Behavior Analysis

A functional college project using Pandas, scikit-learn, Streamlit, Plotly and joblib. The local dataset and measured results are described below.

## What does this model do? (simple explanation)

Imagine a teacher showing a computer player records with the correct answers already written beside them: **Low**, **Medium**, or **High engagement**. The computer learns patterns in those examples. For a new player, you enter their activity and profile details, and it estimates which of those three labels fits best.

For example, you can enter how often someone plays, their average session duration, level, and achievements. The model returns one engagement label and estimated probabilities for all three labels. This is an illustration of the inputs, not a guaranteed prediction: frequent play does not automatically mean the model will return High.

- **Features** are the clues: seven numeric inputs and three categories listed below.
- **Target** is the answer the model learns: the existing `EngagementLevel` column. We do not invent new labels.
- **Training** means learning from examples. **Validation** means comparing model settings. **Testing** means grading the selected model on held-out examples.
- **Logistic Regression** combines weighted clues. **Decision Tree** learns a series of questions. **Random Forest** combines many trees.
- **Accuracy** is the fraction of test answers that match the dataset labels. **Macro F1** balances missed and incorrect predictions and gives each engagement class equal weight.

This classifies the engagement label in this dataset. It does not predict whether a player will quit, explain why they behave that way, or prove that a retention suggestion works. The dataset's label creation rules are unknown: if labels were calculated from the same activity inputs, the model may mainly reproduce those rules.

## Setup and run

Run from this project directory using Python 3.9 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Upload a CSV in the sidebar, or place it at `data/online_gaming_behavior_dataset.csv`. You can also enter another local CSV path. The original CSV is never modified. Open **Model Results → Train models** to train explicitly; normal reruns and exploration filters do not retrain. **Predict Engagement** becomes available after matching training.

Command-line training uses the same functions:

```sh
.venv/bin/python core.py data/online_gaming_behavior_dataset.csv
.venv/bin/python smoke_check.py
```

Only add `--duration-minutes` when the source confirms that `AvgSessionDuration` is measured in minutes. Add `--source "Your source and provenance notes"` to save provenance with the model. Match the duration checkbox to that training setting in the dashboard. Files are saved as `artifacts/model.joblib` and `artifacts/model.json`. Only load trusted local artifacts; pickle/joblib files can execute code. Uploaded serialized models are not supported.

## Data and definitions

Expected target: `EngagementLevel`, exactly `Low`, `Medium`, or `High` after whitespace trimming. Required inputs: `Age`, `Gender`, `Location`, `GameGenre`, `PlayTimeHours`, `InGamePurchases`, `SessionsPerWeek`, `AvgSessionDuration`, `PlayerLevel`, `AchievementsUnlocked`. Optional `PlayerID` is used only for splitting. Extra source columns are retained but excluded from modeling.

The requested source is the Kaggle “Predict Online Gaming Behavior Dataset,” expected to have around 40,000 rows. The local CSV contains 40,034 records. Its original duration column is named `AvgSessionDurationMinutes`, which is normalized to `AvgSessionDuration`. No source URL or authoritative data dictionary is available; provenance, synthetic status, sampling and label construction remain unverified. No Kaggle URL has been guessed. Overview shows actual loaded counts, inferred types, category values, missing cells and numeric ranges. Record source details in the sidebar before training.

Numeric fields must be finite and nonnegative. Age, session counts, player level and achievement counts must be whole numbers. Age is interpreted as years; verify against the source. `InGamePurchases` is accepted only as 0/1 purchase participation, never monetary spending. Invalid numeric cells become missing during cleaning and are rejected during prediction. No speculative upper bounds are imposed. Missing/invalid target rows are dropped, never imputed. Duplicate predictor records retain their first occurrence, even if IDs or labels differ. Cleaning reports actual counts. Blank categories become missing; unknown categories during prediction are supported.

Features: `AchievementsPerLevel = AchievementsUnlocked / PlayerLevel`, with zero level yielding a missing value. When minutes are explicitly confirmed, also use `EstimatedWeeklySessionHours = SessionsPerWeek * AvgSessionDuration / 60`. Missing values propagate to training-fitted imputers. `PlayTimeHours` has an unresolved time window and is not compared with estimated weekly hours.

## Methodology and limits

Stratified 60/20/20 train/validation/test splits use seed 42. Repeated player identities use stratified group folds instead, preserving player separation with approximate proportions. Identical predictor records are deduplicated before splitting. Splits must contain all three classes; insufficient data produces an actionable error. With no reliable player identifier, unobserved repeated people cannot be detected.

Training pipelines fit numeric median imputation, scaling, categorical mode imputation and unknown-safe one-hot encoding only on training records. All-missing columns use scikit-learn's retained-column fallback. A fixed search compares a dummy baseline, 6 Logistic Regression settings, 9 Decision Tree settings, and 8 Random Forest settings by validation macro F1. The original settings are included as explicit baselines. All 24 candidate results and the best settings per family are saved; ties keep the first candidate. The search is fixed before the final test evaluation. The winner is selected before its one test evaluation; it is not refitted on validation data. Results show accuracy, macro precision/recall/F1, class scores, counts and actual-versus-predicted confusion matrix. Validation permutation importance uses three repeats and up to 2,000 rows. The pipeline, metrics, classes, definitions and raw-CSV SHA-256 are saved. A changed CSV or duration setting disables the previous artifact.

Metrics are not evidence of generalization beyond the dataset. Repeated training on the same dataset is deterministic, not an independent experiment. Unknown label construction may make the prediction task circular; synthetic data or sampling bias may limit external validity. Demographic predictors may encode unfair associations. Probabilities may be uncalibrated. Engagement is not future churn. Dashboard observations are associations; retention recommendations are hypotheses requiring subsequent testing. No results are fabricated when data is absent.

`smoke_check.py` uses explicitly synthetic fixtures in temporary storage, never project findings. It checks cleaning, group separation, training, missing values, unseen categories, invalid input, persistence and dashboard pages.

## Measured results — 21 September 2026

| Model | Original validation macro F1 | Best validation macro F1 | Best validation accuracy |
|---|---:|---:|---:|
| Dummy baseline | 0.217416 | 0.217416 | 48.40% |
| Logistic Regression | 0.812506 | 0.812529 | 81.92% |
| Decision Tree | 0.895367 | 0.896232 | 90.05% |
| Random Forest | 0.890614 | 0.909040 | 91.32% |

Validation selected Random Forest (100 trees, max_depth=14, min_samples_leaf=3, max_features=1.0). Final test accuracy: **91.38%**. Final test macro F1: **0.9089** on 8,007 records. The earlier saved Decision Tree scored 89.85% accuracy and 0.8933 macro F1.

This is an observed improvement on the same fixed split, not proof of better performance on new real-world data. The test set was evaluated in an earlier project run; this run kept it out of training and tuning and evaluated only the validation-selected winner. No tuning followed the final test result. Independent confirmation needs fresh data.

See [the full training and leakage report](artifacts/training_report.md) for the search budget, checks, per-class scores and limitations. All candidate results and split hashes are in `artifacts/model.json`. Previous outputs are preserved in `artifacts/before_bounded_tuning/`.
