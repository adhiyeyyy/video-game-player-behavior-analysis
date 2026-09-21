# Functional base handoff

## Entry points

- `app.py`: Streamlit navigation, sidebar loading, charts, training action and prediction form. UI edits belong here.
- `core.py`: shared CSV validation/cleaning, feature transformation, leakage-safe splitting, four-model training, saving and prediction. CLI uses this same path.
- `smoke_check.py`: one runnable check with explicitly synthetic data and temporary artifacts.
- `requirements.txt`: minimal direct dependencies. README contains exact setup and launch commands.

## Scope and verification

The base covers Overview, Exploration, Model Results, Predict Engagement and Retention Insights. No authentication, database, API, deployment, notebook, tuning framework or custom animations were added.

Verified on 2026-09-21 using Python 3.9.6: the focused smoke check passes synthetic training and prediction, cleaning, grouped and stratified splits, missing values, unseen categories, invalid input rejection, artifact round-trip, all five dashboard pages, prediction form submission and unit-setting mismatch handling. Streamlit starts on localhost:8501 and `/_stcore/health` returns `ok`. The bare-mode ScriptRunContext warning comes from Streamlit's test harness.

No real dataset was present; real-data training and scientific findings cannot be verified yet. Test artifacts were temporary and were not saved as project findings. The installed environment is `.venv`; setup commands are in README.

## Unresolved data issues

Supply the actual CSV and original data dictionary/source. Confirm provenance, synthetic status, target construction, duration units and the PlayTimeHours time window. Inspect Overview before training. Age is currently interpreted as whole years; purchases as 0/1 participation. Demographic predictors and unknown sampling restrict responsible interpretation. No model artifact from synthetic checks should be used as project output.

## Prioritized visual polish for Cursor/Antigravity

1. Improve chart label readability and form labels while preserving data units and uncertainty text.
2. Refine spacing and narrow-screen layout; keep contrast and keyboard accessibility.
3. Harmonize chart sizing and number formatting; preserve class colors, supporting counts and percentage denominators.

Keep model functions, split behavior and explicit training unchanged during UI work. Run the focused smoke check after changes.
