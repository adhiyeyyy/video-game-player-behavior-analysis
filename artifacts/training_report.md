# Training and leakage report — 21 September 2026

## What the model does

The model learns from player examples already labeled Low, Medium, or High engagement. Given another player's activity and profile details, it estimates their engagement label. It does not predict future quitting or establish what causes engagement.

## Dataset and leakage checks

- Actual local CSV: 40,034 source rows and 40,034 usable rows; zero missing source cells, invalid numeric cells, duplicate predictor rows, or repeated player IDs.
- Target: EngagementLevel; Medium 19,374, High 10,336, Low 10,324. Targets are never inputs or imputed.
- Inputs: Age, PlayTimeHours, InGamePurchases, SessionsPerWeek, AvgSessionDuration, PlayerLevel, AchievementsUnlocked, Gender, Location, GameGenre.
- PlayerID and GameDifficulty remain excluded. This preserves the previous feature set and project scope.
- AvgSessionDurationMinutes is normalized to AvgSessionDuration. The optional weekly-hours feature remains disabled, matching the previous run. AchievementsPerLevel remains enabled; zero levels become missing.
- Numeric imputation, scaling, category imputation and encoding are fitted separately on training records for every candidate. Feature ratios operate within each row and use no target values.
- No shared row indices, identical predictor records, or known player identities across splits. Runtime checks reject overlap. Split index hashes are stored with metrics for reproducibility.
- Split sizes: 24,020 training, 8,007 validation, 8,007 test. Seed 42 and the previous deterministic split are preserved.

## Fixed experiment

The search was specified before evaluating the new winner on the test set. It ran once on the real CSV. The original settings are included, allowing direct validation comparisons.

- Dummy baseline: always predict the most common training label.
- Logistic Regression: C in {0.1, 1, 10}; class_weight in {None, balanced}; 6 candidates, max_iter=1000.
- Decision Tree: max_depth in {4, 6, 10}; min_samples_leaf in {5, 20, 50}; 9 candidates.
- Random Forest: max_depth in {10, 14}; min_samples_leaf in {3, 10}; max_features in {sqrt, 1.0}; 8 candidates, each with 100 trees.
- All models use seed 42 where supported. The total budget is 24 candidates including the dummy.
- Highest validation macro F1 wins; exact ties retain the first candidate. No refitting on validation records. Only the selected pipeline predicts test records, once at the end of this run. Permutation importance uses validation records only.

| Model | Original validation macro F1 | Best validation macro F1 | Best validation accuracy |
|---|---:|---:|---:|
| Dummy baseline | 0.217416 | 0.217416 | 48.40% |
| Logistic Regression | 0.812506 | 0.812529 | 81.92% |
| Decision Tree | 0.895367 | 0.896232 | 90.05% |
| Random Forest | 0.890614 | 0.909040 | 91.32% |

## Final held-out results

Selected model: **Random Forest**. Settings: 100 trees, max_depth=14, min_samples_leaf=3, max_features=1.0, random_state=42.

| Metric | Previous saved Decision Tree | New validation-selected Random Forest |
|---|---:|---:|
| Test accuracy | 89.8464% | 91.3825% |
| Test macro F1 | 0.893266 | 0.908937 |

Observed accuracy difference: 1.5362 percentage points. Observed macro F1 difference: 0.015671. These are measured differences on this split, not a claim of statistical significance or external validity.

| Class | Precision | Recall | F1 | Test records |
|---|---:|---:|---:|---:|
| Low | 0.9119 | 0.8920 | 0.9018 | 2065 |
| Medium | 0.9127 | 0.9466 | 0.9293 | 3875 |
| High | 0.9182 | 0.8742 | 0.8957 | 2067 |

## Honest limitations

The test set was already evaluated in the earlier project version; it is not a never-before-seen benchmark. This run excludes it from training, hyperparameter selection, and feature importance. No additional tuning was performed after seeing the final score. Fresh independently collected data is needed to confirm the improvement.

The source URL, label construction, sampling method and synthetic status remain unverified. Activity-derived labels could make the task circular even when software leakage checks pass. There is no timestamp-based future evaluation or observed churn target. Demographic inputs can encode bias, probabilities are uncalibrated, and one fixed validation split can favor some settings by chance. The bounded search does not establish the globally best model. Retention suggestions remain hypotheses.

## Validation and saved outputs

`smoke_check.py` checks cleaning, group separation, all 24 candidate fits using only training indices, validation-only predictions before a single final test prediction, best-candidate selection, unseen categories, invalid inputs, persistence, and all dashboard pages including How the Model Works. Synthetic smoke data never replaces the real-data model.

Real training command: `.venv/bin/python core.py data/online_gaming_behavior_dataset.csv`.

Current pipeline and all measured candidate results: `artifacts/model.joblib` and `artifacts/model.json`. Previous outputs are preserved in `artifacts/before_bounded_tuning/`. The README and Streamlit explanation describe the task in student-friendly language. Existing black-and-white styling is preserved.

CSV SHA-256: `63e63057d2984a0630d9af5457df564ac35af5ec0d9d7d82dad6a8e7097f4e15`.
