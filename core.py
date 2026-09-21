"""Shared cleaning, training and prediction; no dashboard dependencies."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = 'EngagementLevel'
CLASSES = ['Low', 'Medium', 'High']
NUMERIC = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek',
           'AvgSessionDuration', 'PlayerLevel', 'AchievementsUnlocked']
CATEGORICAL = ['Gender', 'Location', 'GameGenre']
FEATURES = NUMERIC + CATEGORICAL
ARTIFACT = Path('artifacts/model.joblib')


def validate_features(frame, strict=False):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    result = frame.copy()
    invalid = {}
    for col in NUMERIC:
        values = pd.to_numeric(result[col], errors='coerce')
        bad = result[col].notna() & (values.isna() | ~np.isfinite(values) | (values < 0))
        if col == 'InGamePurchases':
            bad |= values.notna() & ~values.isin([0, 1])
        if col in ['Age', 'SessionsPerWeek', 'PlayerLevel', 'AchievementsUnlocked']:
            bad |= values.notna() & (values % 1 != 0)
        invalid[col] = int(bad.sum())
        result[col] = values.mask(bad).astype(float)
    for col in CATEGORICAL:
        result[col] = result[col].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        result[col] = result[col].replace('', np.nan)
    if strict and any(invalid.values()):
        raise ValueError(f'Invalid numeric input: { {k: v for k, v in invalid.items() if v} }. '
                         'Use finite nonnegative numbers, whole counts, and purchase participation 0 or 1.')
    return result, invalid


def load_data(raw):
    frame = pd.read_csv(io.BytesIO(raw))
    if TARGET not in frame:
        raise ValueError('Missing column: EngagementLevel')
    original = frame.copy()
    if 'AvgSessionDuration' not in frame and 'AvgSessionDurationMinutes' in frame:
        frame = frame.rename(columns={'AvgSessionDurationMinutes': 'AvgSessionDuration'})
    frame, invalid = validate_features(frame)
    frame[TARGET] = frame[TARGET].astype('string').str.strip()
    bad_target = ~frame[TARGET].isin(CLASSES)
    frame = frame.loc[~bad_target].copy()
    # Identical predictor records cannot cross splits, even with different IDs/labels.
    duplicates = frame.duplicated(subset=FEATURES, keep='first')
    frame = frame.loc[~duplicates].reset_index(drop=True)
    summary = {
        'source_rows': len(original), 'usable_rows': len(frame),
        'missing_source_cells': int(original.isna().sum().sum()),
        'dropped_missing_or_invalid_targets': int(bad_target.sum()),
        'dropped_duplicate_predictor_records': int(duplicates.sum()),
        'invalid_numeric_cells_set_missing': invalid,
        'missing_predictor_cells_after_cleaning': int(frame[FEATURES].isna().sum().sum()),
        'decisions': 'Targets outside Low/Medium/High are dropped, never imputed. '
                     'Invalid numeric values become missing. Duplicate predictor rows retain the first record. '
                     'Numeric medians and categorical modes are fitted on training data only. '
                     'Additional columns are preserved but excluded from predictors.',
    }
    return original, frame, summary, hashlib.sha256(raw).hexdigest()


def engineer(frame, duration_minutes=False):
    result = frame[FEATURES].copy()
    result['AchievementsPerLevel'] = result['AchievementsUnlocked'] / result['PlayerLevel'].replace(0, np.nan)
    if duration_minutes:
        result['EstimatedWeeklySessionHours'] = result['SessionsPerWeek'] * result['AvgSessionDuration'] / 60
    return result


def split_data(frame):
    y = frame[TARGET]
    if set(y) != set(CLASSES) or y.value_counts().min() < 5:
        raise ValueError('Training requires all three engagement classes and at least five records per class.')
    groups = frame['PlayerID'].astype('string').str.strip() if 'PlayerID' in frame else None
    repeated = groups is not None and groups.dropna().loc[lambda s: s != ''].duplicated().any()
    if repeated:
        groups = groups.map(lambda value: 'player:' + value if pd.notna(value) and value else None)
        groups = groups.fillna(pd.Series([f'missing:{i}' for i in frame.index], index=frame.index))
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        rest, test = next(outer.split(frame, y, groups))
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=42)
        train_rel, val_rel = next(inner.split(frame.iloc[rest], y.iloc[rest], groups.iloc[rest]))
        train, val = rest[train_rel], rest[val_rel]
        method = 'Stratified group split by PlayerID; proportions are approximate.'
    else:
        rest, test = train_test_split(np.arange(len(frame)), test_size=.2, stratify=y, random_state=42)
        train, val = train_test_split(rest, test_size=.25, stratify=y.iloc[rest], random_state=42)
        method = 'Stratified 60/20/20 train/validation/test split.'
    for indices in (train, val, test):
        if set(y.iloc[indices]) != set(CLASSES):
            raise ValueError('Cannot form splits containing all classes while preserving player separation. Supply more independent players.')
    return train, val, test, method


def train_model(frame, fingerprint, duration_minutes=False, source='Unverified local CSV'):
    train, val, test, method = split_data(frame)
    # Audit split boundaries before fitting. Unknown label provenance remains a limitation.
    splits = {'train': train, 'validation': val, 'test': test}
    row_sets = [set(idx) for idx in splits.values()]
    predictor_hashes = pd.util.hash_pandas_object(frame[FEATURES], index=False)
    feature_sets = [set(predictor_hashes.iloc[idx]) for idx in splits.values()]
    for sets in (row_sets, feature_sets):
        if any(sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError('Leakage detected: overlapping rows or identical predictors across splits.')
    player_overlap = None
    if 'PlayerID' in frame:
        players = frame['PlayerID'].astype('string').str.strip().replace('', pd.NA)
        player_sets = [set(players.iloc[idx].dropna()) for idx in splits.values()]
        player_overlap = any(player_sets[i] & player_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)])
        if player_overlap:
            raise ValueError('Leakage detected: player identities overlap across splits.')
    audit = {
        'target': TARGET, 'target_classes': CLASSES, 'input_features': FEATURES,
        'excluded_columns': sorted(set(frame.columns) - set(FEATURES) - {TARGET}),
        'row_overlap': False, 'identical_predictor_overlap': False,
        'known_player_overlap': player_overlap,
        'preprocessing_fit': 'Training partition only, independently for each candidate.',
        'label_construction': 'Unverified; activity-derived labels could make classification circular.',
        'split_index_sha256': {name: hashlib.sha256(np.asarray(idx, dtype='<i8').tobytes()).hexdigest()
                               for name, idx in splits.items()},
    }
    numeric = NUMERIC + ['AchievementsPerLevel']
    definitions = {'AchievementsPerLevel': 'AchievementsUnlocked / PlayerLevel; zero level becomes missing.'}
    if duration_minutes:
        numeric += ['EstimatedWeeklySessionHours']
        definitions['EstimatedWeeklySessionHours'] = 'SessionsPerWeek * AvgSessionDuration / 60; minutes confirmed by user.'
    models = {
        'Dummy baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=3,
                                                n_jobs=-1, random_state=42),
    }
    # Fixed budget: 1 dummy + 6 logistic + 9 tree + 8 forest candidates.
    # Original settings come first and are always retained as baselines.
    grids = {
        'Dummy baseline': [{}],
        'Logistic Regression': list(ParameterGrid({'C': [0.1, 1.0, 10.0], 'class_weight': [None, 'balanced']})),
        'Decision Tree': list(ParameterGrid({'max_depth': [4, 6, 10], 'min_samples_leaf': [5, 20, 50]})),
        'Random Forest': list(ParameterGrid({'max_depth': [10, 14], 'min_samples_leaf': [3, 10],
                                           'max_features': ['sqrt', 1.0]})),
    }
    comparison, fitted, tuning = [], {}, []
    x, y = frame[FEATURES], frame[TARGET]
    for name, model in models.items():
        preprocess = ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                  ('scale', StandardScaler())]), numeric),
            ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                                   ('encode', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
        ], sparse_threshold=1.0)
        pipeline = Pipeline([('features', FunctionTransformer(engineer, kw_args={'duration_minutes': duration_minutes})),
                             ('preprocess', preprocess), ('model', model)])
        keys = list(grids[name][0])
        original = {key: model.get_params()[key] for key in keys}
        candidates = [original] + [params for params in grids[name] if params != original]
        best = None
        for number, params in enumerate(candidates):
            candidate = clone(pipeline).set_params(**{'model__' + key: value for key, value in params.items()})
            candidate.fit(x.iloc[train], y.iloc[train])
            prediction = candidate.predict(x.iloc[val])
            row = {'model': name, 'macro_f1': f1_score(y.iloc[val], prediction, average='macro'),
                   'accuracy': accuracy_score(y.iloc[val], prediction), 'samples': len(val),
                   'parameters': params, 'original_settings': number == 0}
            tuning.append(row)
            if best is None or row['macro_f1'] > best['macro_f1']:
                best, fitted[name] = row, candidate
        comparison.append(best)
    winner = max(comparison, key=lambda row: row['macro_f1'])['model']
    pipeline = fitted[winner]
    importance = permutation_importance(pipeline, x.iloc[val], y.iloc[val], scoring='f1_macro',
                                       n_repeats=3, max_samples=min(len(val), 2000), random_state=42)
    # One final evaluation, after model selection; no refit or test-driven tuning.
    prediction = pipeline.predict(x.iloc[test])
    report = classification_report(y.iloc[test], prediction, labels=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        'selected_model': winner, 'validation_comparison': comparison, 'test_report': report,
        'confusion_matrix': confusion_matrix(y.iloc[test], prediction, labels=CLASSES).tolist(),
        'split_method': method, 'split_counts': {'train': len(train), 'validation': len(val), 'test': len(test)},
        'split_class_counts': {name: y.iloc[idx].value_counts().to_dict()
                               for name, idx in [('train', train), ('validation', val), ('test', test)]},
        'feature_importance': [{'feature': col, 'mean': float(mean), 'std': float(std)}
                               for col, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
        'dataset_fingerprint': fingerprint, 'class_order': list(pipeline.classes_),
        'confusion_class_order': CLASSES, 'feature_definitions': definitions,
        'duration_minutes_confirmed': duration_minutes, 'source': source,
        'tuning_results': tuning, 'leakage_audit': audit,
        'selection_protocol': 'Fixed 24-candidate search; validation macro F1 only; first candidate wins ties. '
                              'No train+validation refit. One test prediction for the selected pipeline per run. '
                              'Repeated runs do not provide independent test evidence.',
    }
    return {'pipeline': pipeline, 'metrics': metrics}


def save_artifact(artifact, path=ARTIFACT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    path.with_suffix('.json').write_text(json.dumps(artifact['metrics'], indent=2))


def predict_player(artifact, values):
    frame, _ = validate_features(pd.DataFrame([values]), strict=True)
    pipeline = artifact['pipeline']
    return str(pipeline.predict(frame[FEATURES])[0]), dict(zip(pipeline.classes_, pipeline.predict_proba(frame[FEATURES])[0]))


if __name__ == '__main__':
    # Use the importable module for pickled transformation functions, even from CLI.
    import core as shared
    parser = argparse.ArgumentParser(description='Train on a local CSV; never modifies the source.')
    parser.add_argument('csv', type=Path)
    parser.add_argument('--duration-minutes', action='store_true', help='Only set when source confirms minutes.')
    parser.add_argument('--source', default='Unverified local CSV; provenance and synthetic status unknown')
    args = parser.parse_args()
    _, data, summary, fingerprint = load_data(args.csv.read_bytes())
    artifact = shared.train_model(data, fingerprint, args.duration_minutes, args.source)
    save_artifact(artifact)
    print(json.dumps({'cleaning': summary, 'selected_model': artifact['metrics']['selected_model'],
                      'test_report': artifact['metrics']['test_report']}, indent=2))
