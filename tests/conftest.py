"""SYNTHETIC FIXTURE for testing software only. NOT a dataset or empirical result."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import zipfile

@pytest.fixture
def dummy_source_archive(tmp_path):
    source = tmp_path / 'fake_source'
    uv = source / 'Restructured_Data' / 'UVVis'
    uv.mkdir(parents=True)
    rng = np.random.default_rng(5)
    ids = [f'{5+i}_20_75_130' for i in range(24)]
    # The authors' XLSX transposes 4 rows to columns Sample/TOP/MID/BOT.
    # First row is a discarded metadata row; positions are the following three rows.
    matrix = [['sample'] + ids,
              ['TOP'] + [150 + i * 11 for i in range(24)],
              ['MID'] + [152 + i * 11 for i in range(24)],
              ['BOT'] + [154 + i * 11 for i in range(24)]]
    xls = uv / 'Thickness Machine Learning_Perovskite_Blade.xlsx'
    with pd.ExcelWriter(xls, engine='openpyxl') as writer:
        pd.DataFrame([['metadata'] * len(matrix[0])], dtype=object).to_excel(
            writer, index=False, header=False, startrow=0)
        pd.DataFrame(matrix, dtype=object).to_excel(writer, index=False, header=False, startrow=1)
    w = np.linspace(400, 900, 101)
    for i, key in enumerate(ids):
        for j, pos in enumerate(('TOP', 'MID', 'BOT')):
            signal = 0.4 + .2 * np.sin(w / (52 + i) + i * .1) + .002 * j + rng.normal(0, .001, len(w))
            file = uv / f'{key}_{pos}_uvvis.txt'
            file.write_text('Instrument header\nWavelength Intensity\n' +
                            '\n'.join(f'{wl:.3f} {x:.7f}' for wl, x in zip(w, signal)))
    archive = tmp_path / 'source_test_fixture_only.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in source.rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(source))
    return archive
