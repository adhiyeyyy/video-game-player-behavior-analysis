"""Synthetic test fixtures only. These scores are never project findings."""
import tempfile
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from sklearn.pipeline import Pipeline

from core import (CLASSES, FEATURES, load_data, predict_player, save_artifact,
                  split_data, train_model)


def main():
    rng = np.random.default_rng(42)
    n = 150
    fixture = pd.DataFrame({
        'PlayerID': np.repeat(np.arange(n // 2), 2),
        'Age': rng.integers(18, 60, n), 'Gender': rng.choice(['A', 'B'], n),
        'Location': rng.choice(['North', 'South'], n), 'GameGenre': rng.choice(['Action', 'Puzzle'], n),
        'PlayTimeHours': rng.uniform(0, 20, n), 'InGamePurchases': rng.integers(0, 2, n),
        'SessionsPerWeek': rng.integers(0, 15, n), 'AvgSessionDuration': rng.uniform(10, 120, n),
        'PlayerLevel': rng.integers(0, 30, n), 'AchievementsUnlocked': rng.integers(0, 50, n),
        'EngagementLevel': np.repeat(np.resize(CLASSES, n // 2), 2),
    })
    fixture.loc[0, 'Age'] = np.nan
    fixture.loc[1, 'Gender'] = None
    fixture.loc[2, 'PlayTimeHours'] = -2
    fixture.loc[3, 'EngagementLevel'] = None
    fixture = pd.concat([fixture, fixture.iloc[[5]]], ignore_index=True)
    raw = fixture.to_csv(index=False).encode()
    _, data, summary, fingerprint = load_data(raw)
    raw = fixture.rename(columns={'AvgSessionDuration': 'AvgSessionDurationMinutes'}).to_csv(index=False).encode()
    original, aliased_data, _, fingerprint = load_data(raw)
    pd.testing.assert_frame_equal(data, aliased_data)
    assert 'AvgSessionDurationMinutes' in original
    assert summary['dropped_duplicate_predictor_records'] == 1
    assert summary['dropped_missing_or_invalid_targets'] == 1
    assert summary['invalid_numeric_cells_set_missing']['PlayTimeHours'] == 1
    train, val, test, _ = split_data(data)
    groups = [set(data.iloc[idx].PlayerID) for idx in (train, val, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    plain_splits = split_data(data.drop(columns='PlayerID'))[:3]
    assert sum(map(len, plain_splits)) == len(data)
    assert not (set(plain_splits[0]) & set(plain_splits[2]))
    fit_indices, prediction_indices = [], []
    original_fit, original_predict = Pipeline.fit, Pipeline.predict

    @wraps(original_fit)
    def tracked_fit(self, x, y=None, **kwargs):
        if 'features' in self.named_steps:
            fit_indices.append(set(x.index))
        return original_fit(self, x, y, **kwargs)

    @wraps(original_predict)
    def tracked_predict(self, x, **kwargs):
        if 'features' in self.named_steps:
            prediction_indices.append(set(x.index))
        return original_predict(self, x, **kwargs)

    with patch.object(Pipeline, 'fit', tracked_fit), patch.object(Pipeline, 'predict', tracked_predict):
        artifact = train_model(data, fingerprint, duration_minutes=True, source='SYNTHETIC SMOKE TEST ONLY')
    assert len(fit_indices) == 24 and all(indices == set(train) for indices in fit_indices)
    assert prediction_indices[-1] == set(test)
    assert all(indices <= set(val) for indices in prediction_indices[:-1])
    metrics = artifact['metrics']
    assert len(metrics['tuning_results']) == 24
    assert metrics['selected_model'] == max(metrics['validation_comparison'], key=lambda row: row['macro_f1'])['model']
    for best in metrics['validation_comparison']:
        family = [row for row in metrics['tuning_results'] if row['model'] == best['model']]
        assert sum(row['original_settings'] for row in family) == 1
        assert best['macro_f1'] == max(row['macro_f1'] for row in family)
    assert metrics['leakage_audit']['known_player_overlap'] is False
    player = data.iloc[0][FEATURES].to_dict()
    player.update(Gender='Never seen', PlayerLevel=0, Age=np.nan)
    prediction, probabilities = predict_player(artifact, player)
    assert prediction in CLASSES and set(probabilities) == set(CLASSES)
    assert abs(sum(probabilities.values()) - 1) < 1e-8
    try:
        predict_player(artifact, dict(player, SessionsPerWeek=-1))
        raise AssertionError('Invalid prediction accepted')
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.joblib'
        save_artifact(artifact, path)
        assert predict_player(joblib.load(path), player)[0] == prediction
        assert path.with_suffix('.json').is_file()
        csv = Path(directory) / 'fixture.csv'
        csv.write_bytes(raw)
        with patch('core.ARTIFACT', path):
            app = AppTest.from_file('app.py', default_timeout=30).run()
            assert not app.exception
            app.sidebar.text_input[0].set_value(str(csv)).run()
            app.checkbox[0].set_value(True).run()
            assert not app.exception
            for page in ['How the Model Works', 'Exploration', 'Model Results', 'Predict Engagement', 'Retention Insights']:
                app.sidebar.radio[0].set_value(page).run()
                assert not app.exception, page
                if page == 'Predict Engagement':
                    app.button[0].click().run()
                    assert not app.exception and app.success
            app.checkbox[0].set_value(False).run()
            app.sidebar.radio[0].set_value('Predict Engagement').run()
            assert not app.button and app.warning
    print('PASS: synthetic training/prediction, cleaning, group separation, unseen categories, invalid input, persistence, dashboard pages. No project metrics generated.')


if __name__ == '__main__':
    main()
