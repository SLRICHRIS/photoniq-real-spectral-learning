"""Synthetic fixtures only: original experimental data is never copied to Git."""
import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'advanced_benchmark.py'
# In a developer/local checkout, tests live in tests/, tool is in tools/.


def module():
    spec = importlib.util.spec_from_file_location('photoniq_advanced', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_archive(path, target_shift=0):
    rng = np.random.default_rng(31)
    n = 120
    sample = [f'S_{i:04d}' for i in range(n)]
    date = [f'2026-10-{i//12+1:02d}' for i in range(n)]
    targets = 220 + rng.normal(0, 25, size=n) + target_shift
    wavelengths = np.linspace(300, 1000, 45)
    spectra = np.clip(rng.normal(0.4, 0.1, size=(45, n)), 0, 1)
    spectra[7, 0] = np.nan
    db = pd.DataFrame({'Sample':sample,'Date':date,
                       'Perovskite_thickness_nm':targets,
                       'Perovskite_ODint_300_900_nm':targets * 1000})
    curve = pd.DataFrame(spectra, columns=sample)
    curve.insert(0, 'Wavelength_nm', wavelengths)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('Dataset/blade_perovskite_database.csv',db.to_csv(index=False))
        z.writestr('Dataset/curvedata_opticaldensity.csv',curve.to_csv(index=False))


def test_no_target_leakage_into_raw_spectral_features(tmp_path):
    mod = module()
    a, b = tmp_path / 'original.zip', tmp_path / 'relabelled.zip'
    fixture_archive(a,0)
    fixture_archive(b,900)
    db_a, X_a,y_a,group_a,wl_a,audit_a = mod.load_data(a)
    db_b, X_b,y_b,group_b,wl_b,audit_b = mod.load_data(b)
    np.testing.assert_array_equal(X_a, X_b)
    np.testing.assert_allclose(y_b - y_a, 900)
    assert np.isfinite(X_a).all()
    assert audit_a['valid_joined_films'] == 120
    assert audit_a['missing_spectral_values_filled'] == 1


def test_grouped_cross_validation_date_isolation(tmp_path):
    mod = module()
    archive = tmp_path/'real_format_synthetic_content.zip'
    fixture_archive(archive)
    _, X,y,groups,_,_ = mod.load_data(archive)
    encountered = []
    for train,test in GroupKFold(n_splits=5).split(X,y,groups):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        encountered.extend(test.tolist())
    assert sorted(encountered) == list(range(len(y)))


def test_statistical_summary_matches_known_values():
    mod = module()
    out=mod.evaluate(np.array([10.,20.,30.]),np.array([10.,25.,25.]))
    assert out['MAE_nm'] == pytest.approx(10/3)
    assert out['RMSE_nm'] == pytest.approx(np.sqrt(50/3))
    assert 0 <= out['R2'] <= 1
