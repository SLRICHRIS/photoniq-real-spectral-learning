import json
from pathlib import Path
import pytest
from photoniq.train import run_experiment


def test_full_pipeline_on_fake_fixture_only(dummy_source_archive, tmp_path):
    out = tmp_path / 'result'
    run_experiment(dummy_source_archive, out, epochs=3, patience=2,
                   seed=42, n_boot=20, device='cpu')
    for name in ('test_metrics.csv', 'data_audit.json', 'run_metadata.json',
                 'conformal_evaluation.json', 'split_manifest.csv',
                 'test_predictions.csv', 'photoniq_cnn.pt', 'ridge_baseline.joblib'):
        assert (out/name).exists(), name
    meta = json.loads((out / 'run_metadata.json').read_text())
    assert meta['independent_films'] == 24
    assert sum(meta['split_film_counts'].values()) == 24
    assert not (out / 'real_data_accuracy_verified.json').exists()
