"""Strict experimental dataset parsing and film-group splits.

The target (film thickness) must NEVER be used to compute model features.
The source dataset belongs to Harth & Vesce and is not redistributed here.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SOURCE_URL = 'https://data.mendeley.com/datasets/zt2j5j4v44/3'
SOURCE_DOI = '10.17632/zt2j5j4v44.3'
MAX_ARCHIVE_BYTES = 1_500_000_000


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, folder: Path) -> None:
    """Refuse absolute/traversal paths, symlinks, or excessive uncompressed data."""
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        total = 0
        for info in z.infolist():
            path = (folder / info.filename).resolve()
            if path != folder and folder not in path.parents:
                raise ValueError(f'Unsafe ZIP member: {info.filename}')
            # POSIX zip external attributes: symbolic-link mode.
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f'Symlink refused in ZIP: {info.filename}')
            total += info.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError('Extracted archive exceeds size safety cap')
        z.extractall(folder)


def unpack_source(archive: Path, output: Path) -> list[Path]:
    """Extract original ZIP and at most two layers of nested source ZIPs."""
    archive = Path(archive)
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise ValueError('Provide the actual Mendeley Download All ZIP file.')
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    safe_extract(archive, output)
    seen: set[str] = set()
    for _ in range(2):
        nested = [p for p in output.rglob('*') if p.suffix.lower() == '.zip' and str(p) not in seen]
        if not nested:
            break
        for p in nested:
            seen.add(str(p))
            if zipfile.is_zipfile(p):
                safe_extract(p, p.parent / (p.stem + '_unpacked'))
    return sorted(p for p in output.rglob('*') if p.is_file())


def sample_key(text: str) -> str:
    text = str(text).strip().lower().replace('-', '_').replace(' ', '_')
    found = re.search(r'(?<!\d)(\d+_\d+_\d+_\d+(?:_\d+)?)(?!\d)', text)
    if found:
        return found.group(1)
    return re.sub(r'[^a-z0-9]+', '_', re.sub(r'top|middle|mid|bottom|bot', '', text)).strip('_')


def position_key(filename: str) -> str | None:
    u = str(filename).upper()
    found = [p for p, terms in {
        'TOP': ('TOP',), 'MID': ('MID', 'MIDDLE'), 'BOT': ('BOT', 'BOTTOM')
    }.items() if any(term in u for term in terms)]
    return found[0] if len(found) == 1 else None


def parse_thickness_workbook(path: Path) -> pd.DataFrame:
    """Exact published 4-row, transposed Excel layout. No guessed row labels."""
    raw = pd.read_excel(path, header=None, skiprows=1, nrows=4, engine='openpyxl')
    if raw.shape[0] != 4:
        raise ValueError(f'Unsupported thickness workbook layout: {path.name}; shape={raw.shape}')
    frame = raw.T
    frame.columns = ['sample', 'TOP', 'MID', 'BOT']
    frame = frame.iloc[1:]  # published source skips first transposed header row
    records = []
    for _, row in frame.iterrows():
        sample_id = sample_key(row['sample'])
        if not sample_id:
            continue
        for position in ('TOP', 'MID', 'BOT'):
            value = pd.to_numeric(row[position], errors='coerce')
            if np.isfinite(value) and 5 <= value <= 5000:
                records.append(dict(sample_id=sample_id, position=position,
                                    thickness_nm=float(value), label_source=path.name))
    if not records:
        raise ValueError(f'No usable thickness labels in {path}')
    return pd.DataFrame.from_records(records)


def parse_spectrum(path: Path, min_points: int = 30) -> np.ndarray | None:
    """Read experimental UV–Vis TXT; first 2 lines are instrument headers."""
    points = []
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines()[2:]:
        parts = re.split(r'[\s,;]+', line.strip())
        if len(parts) < 2:
            continue
        try:
            w, intensity = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if np.isfinite(w) and np.isfinite(intensity):
            points.append((w, intensity))
    if len(points) < min_points:
        return None
    data = pd.DataFrame(points, columns=['wavelength', 'signal']).groupby('wavelength', as_index=False).mean()
    arr = data.to_numpy(dtype=np.float64)
    if len(arr) < min_points or arr[0, 0] < 180 or arr[-1, 0] > 2500 or np.ptp(arr[:, 0]) < 80:
        return None
    return arr


@dataclass
class ExperimentalData:
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    position: np.ndarray
    path: list[str]
    wavelengths: np.ndarray
    audit: dict


def load_experimental(source_dir: Path, grid_points: int = 128, min_match_fraction: float = .65) -> ExperimentalData:
    root = Path(source_dir)
    workbooks = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.xlsx'
                 and 'thickness' in p.name.lower()]
    if not workbooks:
        raise FileNotFoundError('Thickness XLSX workbooks missing; use published original data archive.')
    labels = pd.concat([parse_thickness_workbook(p) for p in sorted(workbooks)], ignore_index=True)
    # Conflicting labels for the same film/position are disallowed; never silently discard.
    grouped = labels.groupby(['sample_id', 'position']).thickness_nm
    if (grouped.max() - grouped.min() > 0.01).any():
        raise ValueError('Conflicting thickness labels detected for the same sample/position.')
    labels = labels.drop_duplicates(subset=['sample_id', 'position'])

    # Only files explicitly located in a UV-Vis directory are considered spectra.
    # Do NOT search PL data or other instrument signals as fallbacks.
    raw_files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.txt']
    uv_files = [p for p in raw_files if position_key(p.stem) is not None and
                any('uvvis' in str(part).lower().replace('_', '').replace('-', '') for part in p.parts)]
    if not uv_files:
        raise FileNotFoundError('No TOP/MID/BOT TXT files under UVVis directory; inspect source archive layout.')
    spectra = []
    for path in sorted(uv_files):
        arr = parse_spectrum(path)
        if arr is not None:
            spectra.append(dict(sample_id=sample_key(path.stem), position=position_key(path.stem),
                                path=str(path.relative_to(root)), array=arr))
    if not spectra:
        raise ValueError('All selected UV–Vis files failed numeric validation')
    spec = pd.DataFrame(spectra)
    matched = spec.merge(labels, on=['sample_id', 'position'], how='inner', validate='many_to_one')
    coverage = len(matched) / len(spec)
    if coverage < min_match_fraction:
        raise ValueError(f'Too few matched spectra: {len(matched)}/{len(spec)} ({coverage:.1%}); inspect IDs and workbooks.')
    if matched.sample_id.nunique() < 15:
        raise ValueError('Fewer than 15 distinct labelled films; grouped evaluation would be unreliable.')
    matched = matched.sort_values(['sample_id', 'position', 'path']).reset_index(drop=True)
    lo = max(float(arr[0, 0]) for arr in matched['array'])
    hi = min(float(arr[-1, 0]) for arr in matched['array'])
    if hi - lo < 80:
        raise ValueError('Insufficient shared wavelength span for interpolation.')
    grid = np.linspace(lo, hi, grid_points, dtype=np.float32)
    X = np.stack([np.interp(grid, arr[:, 0], arr[:, 1]) for arr in matched['array']]).astype('float32')
    y = matched.thickness_nm.to_numpy(dtype='float32')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite feature/target values after preprocessing')
    audit = dict(dataset_doi=SOURCE_DOI, spectra_found=len(spec), spectra_matched=len(matched),
                 match_fraction=round(coverage, 5), films=int(matched.sample_id.nunique()),
                 workbooks=[p.name for p in workbooks],
                 missing_spectra=int(len(spec) - len(matched)),
                 wavelength_min_nm=round(lo, 4), wavelength_max_nm=round(hi, 4),
                 instrument_signal_units='original UV-Vis signal units; not thickness-normalised',
                 target_units='nm', original_data_type='EXPERIMENTAL')
    return ExperimentalData(X, y, matched.sample_id.to_numpy(dtype=str),
                            matched.position.to_numpy(dtype=str), matched.path.tolist(), grid, audit)


def grouped_split(groups: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    groups = np.asarray(groups)
    indices = np.arange(len(groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    tv, test = next(splitter.split(indices, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1)
    t, v = next(second.split(tv, groups=groups[tv]))
    splits = {'train': tv[t], 'val': tv[v], 'test': test}
    sets = [set(groups[idx]) for idx in splits.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), 'Group leakage'
    assert all(len(g) >= 3 for g in sets), 'Insufficient independent films in one split'
    return splits
