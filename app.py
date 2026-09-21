"""Basic Streamlit dashboard. Keep data/model behavior in core.py."""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from core import (ARTIFACT, CATEGORICAL, CLASSES, FEATURES, NUMERIC, TARGET,
                  load_data, predict_player, save_artifact, train_model)

st.set_page_config(page_title='Video Game Player Behavior Analysis', layout='wide')
COLORS = {'Low': '#A3A3A3', 'Medium': '#626262', 'High': '#171717'}
LABELS = {'Age': 'Age (years)', 'PlayTimeHours': 'Play time (hours)',
          'InGamePurchases': 'Purchase participation (0=no, 1=yes)',
          'SessionsPerWeek': 'Sessions per week', 'AvgSessionDuration': 'Average session duration',
          'PlayerLevel': 'Player level', 'AchievementsUnlocked': 'Achievements unlocked',
          'GameGenre': 'Game genre', 'Gender': 'Gender', 'Location': 'Location'}
st.markdown('''
<style>
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 2rem; }
.metric-card { background: #ffffff; border: 1px solid #e5e5e5; border-radius: 12px; padding: 20px; }
.metric-label { color: #555555; font-size: .95rem; margin-bottom: 8px; }
.metric-value { color: #111111; font-size: 2rem; font-weight: 700; }
[data-testid="stCaptionContainer"] { color: #666666; }
.stButton > button, .stFormSubmitButton > button { background: #111111; color: #ffffff; border-radius: 8px; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background: #333333; color: #ffffff; }
</style>
''', unsafe_allow_html=True)
st.title('Video Game Player Behavior Analysis')
st.caption('College Data Science Project · Player Engagement Classification')


def metric_card(panel, label, value):
    panel.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def chart(figure, title):
    """Apply the same restrained presentation to existing chart data."""
    figure.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white',
                         font={'color': '#262626', 'family': 'Arial, sans-serif', 'size': 13},
                         colorway=list(COLORS.values()), margin={'l': 20, 'r': 20, 't': 20, 'b': 20},
                         legend_title_text='', height=360)
    figure.update_xaxes(gridcolor='#eeeeee', zerolinecolor='#dddddd')
    figure.update_yaxes(gridcolor='#eeeeee', zerolinecolor='#dddddd')
    with st.container(border=True):
        st.markdown(f'**{title}**')
        st.plotly_chart(figure, theme=None, config={'displayModeBar': False})


@st.cache_data
def read_csv(raw):
    return load_data(raw)


def distribution(frame):
    counts = frame[TARGET].value_counts().reindex(CLASSES, fill_value=0).rename_axis('Engagement').reset_index(name='Players')
    counts['Percent'] = (100 * counts['Players'] / len(frame)).round(1) if len(frame) else 0
    st.caption(f'Percentage denominator: {len(frame):,} cleaned records.')
    chart(px.bar(counts, x='Engagement', y='Players', color='Engagement',
                 color_discrete_map=COLORS, hover_data=['Percent']), 'Engagement distribution')


upload = st.sidebar.file_uploader('Upload CSV', type=['csv'])
local_path = st.sidebar.text_input('Local CSV path', 'data/online_gaming_behavior_dataset.csv')
try:
    raw = upload.getvalue() if upload else Path(local_path).read_bytes() if Path(local_path).is_file() else None
except OSError as exc:
    st.error(f'Cannot read CSV: {exc}')
    st.stop()
if raw is None:
    st.info('Upload a CSV in the sidebar, or place your dataset at the local path shown there.')
    st.write('Required columns:', ', '.join(FEATURES + [TARGET]))
    st.caption('No dataset is bundled. No project results are available until real data is loaded and training is requested.')
    st.stop()
try:
    original, data, cleaning, fingerprint = read_csv(raw)
except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
    st.error(f'CSV validation failed: {exc}')
    st.stop()

page = st.sidebar.radio('Navigation', ['Overview', 'How the Model Works', 'Exploration', 'Model Results', 'Predict Engagement', 'Retention Insights'])
content = st.container()
with st.expander('Dataset settings', expanded=False):
    source = st.text_input('Source / provenance notes',
        'User-supplied CSV; Kaggle provenance, synthetic status, and data definitions are unverified.')
    minutes = st.checkbox('Source confirms AvgSessionDuration is minutes', value=False)
    st.caption('PlayTimeHours time window is unresolved. Purchases represent participation (0/1), not money.')
with content:
    artifact = None
    if ARTIFACT.is_file():
        try:
            candidate = joblib.load(ARTIFACT)  # Trusted local output only; no serialized uploads.
            if candidate['metrics']['dataset_fingerprint'] != fingerprint:
                st.warning('Saved model belongs to a different CSV. Train on this dataset to enable results and prediction.')
            elif candidate['metrics']['duration_minutes_confirmed'] != minutes:
                st.warning('Duration-unit setting differs from the trained model. Restore the setting or train again.')
            else:
                artifact = candidate
        except Exception as exc:
            st.warning(f'Local model could not be loaded. Retrain it. ({type(exc).__name__})')

    if page == 'Overview':
        st.subheader('Dataset Overview')
        st.write('Predict Low, Medium, or High player engagement from activity and profile details. Open How the Model Works for a simple explanation.')
        cols = st.columns(3)
        metric_card(cols[0], 'Source Records', f'{len(original):,}')
        metric_card(cols[1], 'Cleaned Records', f'{len(data):,}')
        metric_card(cols[2], 'Missing Values', f'{cleaning["missing_source_cells"]:,}')
        st.caption(f'Dataset fingerprint · SHA-256: {fingerprint}')
        distribution(data)
        st.subheader('Cleaning Summary')
        summary_rows = [
            ('Source rows', f'{cleaning["source_rows"]:,}'),
            ('Usable rows', f'{cleaning["usable_rows"]:,}'),
            ('Missing source cells', f'{cleaning["missing_source_cells"]:,}'),
            ('Dropped invalid targets', f'{cleaning["dropped_missing_or_invalid_targets"]:,}'),
            ('Duplicate records removed', f'{cleaning["dropped_duplicate_predictor_records"]:,}'),
            ('Missing predictor cells', f'{cleaning["missing_predictor_cells_after_cleaning"]:,}'),
            ('Handling decisions', cleaning['decisions'] + ' PlayerID is excluded from predictors.'),
        ]
        st.dataframe(pd.DataFrame(summary_rows, columns=['Item', 'Value']),
                     width='stretch', hide_index=True)
    elif page == 'How the Model Works':
        st.subheader('How the Model Works')
        st.write('This model estimates whether a player belongs to the Low, Medium, or High engagement group in this dataset.')
        st.markdown("""
**1. Give it examples.** Each dataset row describes a player and already has an engagement label.
The model learns patterns from these examples; it does not create the correct answers itself.

**2. Give it clues.** Inputs include age, play time, purchase participation, sessions per week,
average session duration, player level, achievements, gender, location, and game genre.
PlayerID, GameDifficulty, and the answer column are excluded from the inputs.

**3. Compare three ways of learning.** Logistic Regression combines weighted clues.
A Decision Tree asks a series of questions. A Random Forest combines many trees.
We also compare a simple baseline that always guesses the most common training label.

**4. Practise, choose, then test.** About 60% of records teach the models, 20% help choose
settings, and 20% grade the selected model. Settings are chosen using validation macro F1,
which gives Low, Medium, and High equal importance. Accuracy means the fraction of correct answers.

**5. Try a player.** Open Predict Engagement, enter their details, and submit.
For example, enter their sessions per week, session duration, level, and achievements.
You receive one predicted label plus estimated probabilities. These are not guarantees.

**What this cannot tell you.** Low engagement does not mean someone will quit.
The model does not prove what causes engagement or whether reminders will help.
If the original labels were calculated from the activity inputs, it may mainly learn those rules.
The label creation rules and whether the data is synthetic remain unverified.
""")
        st.caption('The existing test set has been evaluated in an earlier project run. It is excluded from fitting and tuning in this run; a fresh external dataset is needed for independent confirmation.')
    elif page == 'Exploration':
        st.subheader('Exploration')
        st.markdown('**Filter players**')
        filtered = data.copy()
        columns = st.columns(3)
        for panel, field in zip(columns, ['GameGenre', 'Location', 'Gender']):
            options = sorted(data[field].dropna().unique())
            chosen = panel.multiselect(LABELS[field], options, default=options)
            filtered = filtered[filtered[field].isin(chosen)]
        ages = data['Age'].dropna()
        if not ages.empty and ages.min() < ages.max():
            age_range = st.slider('Age (whole years)', int(ages.min()), int(ages.max()), (int(ages.min()), int(ages.max())))
            filtered = filtered[filtered['Age'].between(*age_range)]
        st.caption(f'{len(filtered):,} of {len(data):,} cleaned records. Missing filter categories/ages are excluded when those filters apply.')
        if filtered.empty:
            st.info('No records match these filters.')
        else:
            distribution(filtered)
            for field, label in [('PlayTimeHours', 'Play time (hours; time window unverified)'),
                                 ('SessionsPerWeek', 'Sessions per week'),
                                 ('AvgSessionDuration', 'Average session duration' + (' (minutes)' if minutes else ' (units unverified)'))]:
                chart(px.histogram(filtered, x=field, color=TARGET, barmode='overlay',
                                   color_discrete_map=COLORS, labels={field: label}, nbins=30), label)
            purchases = filtered.groupby(['InGamePurchases', TARGET], dropna=False).size().reset_index(name='Records')
            chart(px.bar(purchases, x='InGamePurchases', y='Records', color=TARGET,
                         color_discrete_map=COLORS, labels={'InGamePurchases': LABELS['InGamePurchases']}),
                  'Purchase participation by engagement')
    elif page == 'Model Results':
        st.subheader('Model Results')
        st.caption('Explicit training only. Exploration filters never affect training or evaluation.')
        if st.button('Train models', type='primary', disabled=data.empty):
            try:
                with st.spinner('Comparing 24 fixed candidates, then evaluating the validation-selected winner…'):
                    artifact = train_model(data, fingerprint, minutes, source)
                    save_artifact(artifact)
                st.success('Training complete. Model and metrics saved locally.')
            except (ValueError, OSError) as exc:
                st.error(f'Training failed: {exc}')
        if artifact:
            metrics = artifact['metrics']
            report = metrics['test_report']
            panels = st.columns(3)
            panels[0].metric('Selected Model', metrics['selected_model'])
            panels[1].metric('Test Accuracy', f'{report["accuracy"]:.2%}')
            panels[2].metric('Macro F1', f'{report["macro avg"]["f1-score"]:.3f}')
            st.caption('Final scores use the held-out test set, excluded from fitting and tuning. Model selection uses validation macro F1. This test set was also evaluated in an earlier project run.')
            st.markdown('**Model comparison · validation set**')
            st.dataframe(pd.DataFrame(metrics['validation_comparison']).assign(
                parameters=lambda table: table['parameters'].map(str) if 'parameters' in table else '').rename(columns={
                'model': 'Model', 'macro_f1': 'Macro F1', 'accuracy': 'Accuracy', 'samples': 'Samples'}),
                hide_index=True, column_config={'Macro F1': st.column_config.NumberColumn(format='%.3f'),
                                               'Accuracy': st.column_config.NumberColumn(format='%.3f')})
            chart(px.imshow(metrics['confusion_matrix'], x=CLASSES, y=CLASSES, text_auto=True,
                                      labels={'x': 'Predicted engagement', 'y': 'Actual engagement', 'color': 'Records'},
                                      color_continuous_scale=['#f5f5f5', '#171717']), 'Confusion matrix · test set')
            importance = pd.DataFrame(metrics['feature_importance']).sort_values('mean')
            chart(px.bar(importance, x='mean', y='feature', error_x='std', orientation='h',
                         color_discrete_sequence=['#383838'],
                         labels={'mean': 'Validation macro F1 decrease after permutation', 'feature': 'Input feature'}),
                  'Feature importance · validation set')
            st.caption('Importance uses validation data, three repeats, at most 2,000 records. Correlated inputs can share importance; negative values are possible.')
            with st.expander('Tuning and leakage checks'):
                st.write('A fixed, bounded search includes the original model settings. Higher scores are not guaranteed.')
                if 'tuning_results' in metrics:
                    st.dataframe(pd.DataFrame(metrics['tuning_results']).assign(
                        parameters=lambda table: table['parameters'].map(str)), hide_index=True)
                    audit = metrics['leakage_audit']
                    st.table(pd.DataFrame([
                        ('Rows shared across splits', str(audit['row_overlap'])),
                        ('Identical predictors shared across splits', str(audit['identical_predictor_overlap'])),
                        ('Known players shared across splits', str(audit['known_player_overlap'])),
                        ('Preprocessing learned from', audit['preprocessing_fit']),
                        ('Excluded columns', ', '.join(audit['excluded_columns'])),
                    ], columns=['Check', 'Result']))
                else:
                    st.write('Retrain to generate the bounded-search audit.')
                st.write('Unknown label construction can still make the task circular. These checks cannot establish real-world validity.')
            with st.expander('Evaluation details'):
                st.write(metrics['split_method'])
                st.table(pd.DataFrame.from_dict(metrics['split_counts'], orient='index', columns=['Records']))
                st.dataframe(pd.DataFrame(metrics['split_class_counts']).fillna(0))
                details = st.columns(2)
                details[0].metric('Macro precision', f'{report["macro avg"]["precision"]:.3f}')
                details[1].metric('Macro recall', f'{report["macro avg"]["recall"]:.3f}')
                st.dataframe(pd.DataFrame({key: report[key] for key in CLASSES}).T)
                st.table(pd.DataFrame(metrics['feature_definitions'].items(), columns=['Feature', 'Definition']).set_index('Feature'))
                st.caption('Training source: ' + metrics['source'])
        else:
            st.info('No matching trained artifact. Use Train models above, or the README command.')
    elif page == 'Predict Engagement':
        st.subheader('Predict Engagement')
        if not artifact:
            st.info('Prediction is disabled. Train a matching model on Model Results first.')
        else:
            with st.form('player'):
                values = {}
                panels = st.columns(3)
                for i, field in enumerate(NUMERIC):
                    median = data[field].median()
                    default = float(median) if pd.notna(median) else 0.0
                    label = LABELS[field]
                    if field == 'InGamePurchases':
                        values[field] = panels[i % 3].selectbox(label, [0, 1])
                    elif field in ['Age', 'SessionsPerWeek', 'PlayerLevel', 'AchievementsUnlocked']:
                        values[field] = panels[i % 3].number_input(label, min_value=0, value=int(default), step=1)
                    else:
                        values[field] = panels[i % 3].number_input(label, min_value=0.0, value=default)
                for i, field in enumerate(CATEGORICAL):
                    values[field] = panels[i].text_input(LABELS[field], value=str(data[field].dropna().iloc[0]) if data[field].notna().any() else '')
                submitted = st.form_submit_button('Predict engagement')
            st.caption('Purchase participation is binary. PlayTimeHours time window is unverified. Average session duration units: ' + ('minutes' if minutes else 'unverified; use the same units as the CSV') + '.')
            if submitted:
                try:
                    prediction, probabilities = predict_player(artifact, values)
                    st.success(f'Predicted Engagement Level: {prediction}')
                    st.dataframe(pd.DataFrame({'Class': list(probabilities), 'Model probability': list(probabilities.values())}),
                                 hide_index=True, column_config={'Model probability': st.column_config.NumberColumn(format='%.3f')})
                    st.caption('Probabilities are model estimates, not guarantees, and may be uncalibrated. Low engagement does not establish future churn.')
                except ValueError as exc:
                    st.error(str(exc))
    elif page == 'Retention Insights':
        st.subheader('Retention Insights')
        if data.empty:
            st.info('No cleaned records are available.')
        else:
            st.markdown('### Key Observations')
            counts = data[TARGET].value_counts().reindex(CLASSES, fill_value=0)
            st.write(f'Low engagement: {counts["Low"]:,}/{len(data):,} records ({counts["Low"] / len(data):.1%}). This is not an observed churn rate.')
            st.write('Session patterns by engagement (means and non-missing supporting counts)')
            st.dataframe(data.groupby(TARGET)[['SessionsPerWeek', 'AvgSessionDuration', 'PlayTimeHours']].agg(['mean', 'count']))
            known = data.dropna(subset=['InGamePurchases'])
            if len(known):
                st.write(f'Purchase participation: {int(known["InGamePurchases"].sum()):,}/{len(known):,} records with known participation ({known["InGamePurchases"].mean():.1%}).')
            st.markdown('### Player Segments')
            st.caption('Genre engagement counts')
            st.dataframe(pd.crosstab(data['GameGenre'], data[TARGET]).reindex(columns=CLASSES, fill_value=0))
            st.markdown('### Suggested Retention Actions')
            st.info('Hypotheses to test: optional session reminders or onboarding improvements may support engagement. Evaluate them in a subsequent controlled study, using the supporting counts to choose experiments. These associations do not establish what changes engagement or retention.')
            st.caption('Sampling, synthetic status, and target construction are unverified. Demographic predictors can encode bias; this college demonstration is not validated for decisions about people.')
