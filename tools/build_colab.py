"""Generate the SELF-CONTAINED Colab notebook from source files; do not hand-edit notebook."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'notebooks' / 'PhotonIQ_Real_Experimental_Training.ipynb'


def markdown(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def main():
    cells = [
        markdown('''# PhotonIQ — Experimental Spectral Learning · v1.0

**Train a mask-aware 1D CNN on real optoelectronic thin-film measurements.** This is a reproducible research notebook, not a website. **No synthetic training data or synthetic fallback.**

**Source**: M. Harth and L. Vesce, *Perovskite-Thin-Films* (Mendeley Data, v3, 2023). [DOI: 10.17632/zt2j5j4v44.3](https://doi.org/10.17632/zt2j5j4v44.3). CC BY 4.0. Corresponding study: *Optoelectronic perovskite film characterization via machine vision* (Solar Energy, 2023).

**Outputs**: source checksum, audited scan-to-thickness joins, group-disjoint train/val/test manifest, train-only scaling, masked CNN training, mean/Ridge baselines, MAE/RMSE/R² and group-bootstrap intervals, validation-film conformal intervals, saved model checkpoint and complete exported results.

**Important**: film thickness is *only a target*. It is never used to divide or otherwise construct optical input features. The model is empirical, **not** claimed physics-informed. `TOP/MID/BOT` scans from the same physical film must stay in the same split. Results are unverified until this notebook runs on the publisher's original experimental ZIP.'''),
        code('%pip -q install "numpy>=1.24,<3" "pandas>=2,<3" "scikit-learn>=1.3,<2" "openpyxl>=3.1,<4" "joblib>=1.3,<2" "nbformat>=5.9,<6"'),
        markdown('''## 1 · Acquire the original data

Download the experimental ZIP using **Download All** at [Mendeley Data v3](https://data.mendeley.com/datasets/zt2j5j4v44/3), and upload the unchanged ZIP in the cell below. You may instead supply a path to a copy in Google Drive. The source archive is not committed to Git. The code verifies ZIP format and SHA-256, and refuses malformed or missing experimental files.

If you use a Google Drive ZIP, mount Drive and set `ARCHIVE_IN_DRIVE` to the full filename. This notebook has no fabricated replacement dataset.'''),
        code('''from pathlib import Path
ARCHIVE_IN_DRIVE = ""  # Optional e.g. /content/drive/MyDrive/Datasets/Perovskite-Thin-Films.zip
if ARCHIVE_IN_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    archive_path = Path(ARCHIVE_IN_DRIVE)
else:
    from google.colab import files
    print('Upload ORIGINAL Mendeley Download All ZIP (v3):')
    uploads = files.upload()
    names = [name for name in uploads if name.lower().endswith('.zip')]
    if len(names) != 1:
        raise ValueError('Upload exactly ONE source ZIP from Mendeley Data; do not upload synthetic test fixtures.')
    archive_path = Path(names[0]).resolve()
if not archive_path.is_file():
    raise FileNotFoundError(archive_path)
print('Source ZIP selected:', archive_path.name, '| bytes:', archive_path.stat().st_size)'''),
        markdown('''## 2 · Install the versioned PhotonIQ source

These cells are **generated verbatim from the GitHub repository's `src/` modules** by `tools/build_colab.py`. They make this notebook fully usable even before the repository is public. To change methods, edit `src/` and regenerate the notebook; do not manually diverge this file.'''),
        code("from pathlib import Path\nPath('/content/photoniq_runtime/photoniq').mkdir(parents=True, exist_ok=True)"),
    ]
    files = [ROOT / 'src' / 'photoniq' / name for name in
             ('__init__.py', 'data.py', 'model.py', 'metrics.py', 'train.py')]
    for path in files:
        relative = path.relative_to(ROOT/'src')
        content = path.read_text(encoding='utf-8').rstrip() + '\n'
        cells.append(nbf.v4.new_code_cell(f'%%writefile /content/photoniq_runtime/{relative}\n' + content))
    cells += [
        markdown('''## 3 · Audit, train, evaluate

Default: independent-film 60/20/20 train/validation/test split, 128 wavelength bins, up to 150 epochs with validation early stopping, CPU/GPU auto selection. The test split is never used to choose architecture or epoch. Small datasets may produce wide confidence intervals. To adjust training for a quick dry run, set `EPOCHS = 3`; do **not** publish that as a scientific result.'''),
        code('''import sys
from pathlib import Path
sys.path.insert(0, '/content/photoniq_runtime')
from photoniq.train import run_experiment

EPOCHS = 150
PATIENCE = 20
SEED = 42
BOOTSTRAP_ITERATIONS = 1000
run_dir = Path('/content/PhotonIQ_Experimental_Results')
report = run_experiment(archive_path, run_dir, epochs=EPOCHS, patience=PATIENCE,
                        seed=SEED, n_boot=BOOTSTRAP_ITERATIONS, device='auto')
print('Independent films:', report['metadata']['independent_films'])
print('Dataset SHA256:', report['metadata']['dataset_sha256'])'''),
        markdown('''## 4 · Inspect independent test performance

Interpret the CNN alongside conventional baselines; don't declare it superior unless results support that. The sparse-channel runs use uniformly spaced *fixed* wavelength masks, NOT an adaptive acquisition policy. The conformal interval assumes exchangeability at the film level and is not a certificate of accuracy on other materials or instruments.'''),
        code('''import pandas as pd
from IPython.display import display
metrics = pd.read_csv(run_dir/'test_metrics.csv')
manifest = pd.read_csv(run_dir/'split_manifest.csv')
print('Group-level distribution (FILMS, not scans):')
display(manifest.groupby('split').sample_id.nunique())
display(metrics)
print('Group-conformal evaluation:')
print((run_dir/'conformal_evaluation.json').read_text())'''),
        code('''import matplotlib.pyplot as plt
preds = pd.read_csv(run_dir/'test_predictions.csv')
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(preds.thickness_nm, preds.cnn_full_prediction_nm, alpha=.7)
lo = min(preds.thickness_nm.min(), preds.cnn_full_prediction_nm.min())
hi = max(preds.thickness_nm.max(), preds.cnn_full_prediction_nm.max())
ax.plot([lo,hi], [lo,hi], '--', color='black', lw=1)
ax.set(xlabel='Measured film thickness (nm)', ylabel='CNN prediction (nm)',
       title='Held-out experimental films — full UV–Vis spectrum')
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(run_dir/'experimental_test_scatter.png', dpi=180)
plt.show()'''),
        markdown('''## 5 · Export all results

The ZIP contains predictions, split manifest, metrics, dataset audit, conformal analysis, model weights, and environment metadata. **It does not include the source's raw experimental data**. Before publishing any GitHub performance table, inspect the run audit and independent film counts; retrain if the parser reports inadequate matches.'''),
        code('''from google.colab import files
import shutil
bundle_path = shutil.make_archive('/content/PhotonIQ_Experimental_Results', 'zip', root_dir=run_dir)
print('Export:', bundle_path)
files.download(bundle_path)'''),
        markdown('''### Responsible interpretation

- Not enough evidence to claim a physics-informed method, active wavelength selection, or cross-material generalization.
- Group-disjoint test scans are still only from the same published experimental collection; external validation remains future work.
- Report per-film (not only per-row) accuracy, confidence intervals, source DOI, and data SHA256.
- If the CNN performs worse than Ridge, report that result.

For the methodology, see `docs/DATASET.md`, `docs/MODEL_CARD.md`, and `docs/EXPERIMENT_PROTOCOL.md` in the GitHub-ready project package.''')
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'colab': {'name': OUTPUT.name, 'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    })
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    nbf.write(nb, OUTPUT)
    print('Wrote:', OUTPUT)


if __name__ == '__main__':
    main()
