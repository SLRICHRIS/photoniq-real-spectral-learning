import zipfile
from pathlib import Path
import numpy as np
import pytest
from photoniq.data import (sample_key, position_key, unpack_source, safe_extract,
                           parse_thickness_workbook, load_experimental, grouped_split)


def test_sample_keys_and_positions():
    assert sample_key('10_20_75_130_TOP') == '10_20_75_130'
    assert sample_key('10-20-75-130') == '10_20_75_130'
    assert position_key('10_20_75_130_TOP') == 'TOP'
    assert position_key('sample_MID_measurement') == 'MID'
    assert position_key('sample_TOP_BOT') is None


def test_real_parser_contract_on_synthetic_fixture(dummy_source_archive, tmp_path):
    raw = tmp_path / 'extract'
    unpack_source(dummy_source_archive, raw)
    data = load_experimental(raw)
    assert data.X.shape == (72, 128)
    assert len(np.unique(data.group)) == 24
    assert set(np.unique(data.position)) == {'TOP', 'MID', 'BOT'}
    assert data.audit['original_data_type'] == 'EXPERIMENTAL'  # schema designation; fixture itself is synthetic
    assert all(np.isfinite(data.X.ravel()))
    # No y/film-thickness leaked into feature construction: perturb ONLY labels and assert X unchanged.
    book = next(raw.rglob('*.xlsx'))
    import openpyxl
    workbook = openpyxl.load_workbook(book)
    sheet = workbook.active
    # workbook row 3 contains top measurements; col B is first specimen.
    sheet.cell(3, 2).value = 1400
    workbook.save(book)
    altered = load_experimental(raw)
    np.testing.assert_allclose(data.X, altered.X)
    assert not np.array_equal(data.y, altered.y)


def test_group_splits_prevent_leakage(dummy_source_archive, tmp_path):
    unpack_source(dummy_source_archive, tmp_path / 'data')
    data = load_experimental(tmp_path / 'data')
    one = grouped_split(data.group, seed=7)
    two = grouped_split(data.group, seed=7)
    for key in one:
        np.testing.assert_array_equal(one[key], two[key])
    sets = [set(data.group[idx]) for idx in one.values()]
    assert not sets[0].intersection(sets[1])
    assert not sets[1].intersection(sets[2])
    assert not sets[0].intersection(sets[2])


def test_secure_zip_extract_refuses_path_traversal(tmp_path):
    z = tmp_path / 'bad.zip'
    with zipfile.ZipFile(z, 'w') as writer:
        writer.writestr('../leak.txt', 'bad')
    with pytest.raises(ValueError, match='Unsafe ZIP member'):
        safe_extract(z, tmp_path / 'out')
