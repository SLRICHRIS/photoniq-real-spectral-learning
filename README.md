# PhotonIQ

### Experimental spectral learning for perovskite thin-film thickness

**Physics-respecting research discipline, real measured spectra, reproducible Colab training.** PhotonIQ learns a regression relationship between experimental UV–Vis spectral measurements and independently measured thin-film thickness. It compares a compact **mask-aware 1D CNN** with conventional regression baselines, evaluates missing-wavelength robustness, and reports uncertainty at the **physical-film** level.

> **Status: Research implementation verified on a software-only synthetic test fixture; the actual Mendeley experimental ZIP has not yet been trained here. There is no verified real-data accuracy to report.** The distributed notebook requires you to upload the original experimental dataset before training. No simulation-generated measurements are used to train the real-data model.

## Course context

**Course:** Optoelectronic Materials & Devices (23ECE353)  
**Course faculty:** [Dr. Suma H.](https://www.amrita.edu/faculty/suma-h/), Assistant Professor, Department of Electronics & Communication Engineering  
**Institution:** Amrita Vishwa Vidyapeetham, School of Engineering, Amritapuri

PhotonIQ is a student-developed software research project undertaken in connection with this course. The faculty listing acknowledges the teaching context; it does not imply institutional endorsement, formal supervision, or experimental validation.

## Start here: Google Colab

1. Download [`notebooks/PhotonIQ_Real_Experimental_Training.ipynb`](notebooks/PhotonIQ_Real_Experimental_Training.ipynb).
2. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and select the `.ipynb`.
3. Obtain the original **Download All** ZIP from [Harth–Vesce experimental dataset v3](https://data.mendeley.com/datasets/zt2j5j4v44/3). The notebook prompts for this exact file, or you can set a Google Drive path.
4. Run the notebook top-to-bottom. It produces audited training metadata, trained CNN weights, baseline predictions, held-out test metrics and a downloadable results ZIP.

**No website, sensor, microcontroller, or semiconductor fabrication is required.** Colab may use a GPU, but this architecture also runs on CPU.

## What is the scientific question?

*Can a small deep-learning model recover film thickness from measured optical spectra when only some wavelength measurements are available, without relying on target leakage or artificially inflated scan-level accuracy?*

We do **not** pre-claim novelty or model superiority. A competing Ridge regressor might be better on a small experimental collection; PhotonIQ reports this transparently.

## Results and verification

**Results status (28 October 2026):** The software validation pipeline has passed, but **no model metrics have been produced from the original experimental Mendeley archive**. The table below deliberately separates confirmed engineering checks from scientific measurements. No synthetic fixture scores are presented as experimental results.

### Software validation results

| Verification | Result | Evidence |
|---|---|---|
| GitHub Actions CI | **Passed** | [Latest verified workflow run](https://github.com/SLRICHRIS/photoniq-real-spectral-learning/actions/runs/40689364441) |
| Python automated tests | **8/8 passed in local testing** | [Test suite](tests/) and CI workflow |
| End-to-end training smoke test | **Passed on synthetic software-test fixture** | [Integration test](tests/test_integration.py) |
| Colab notebook/source consistency | **Passed by repository checks** | [Notebook checker](tools/check_notebook.py) |
| Training on the actual perovskite measurements | **Not yet executed** | [Real-data Colab notebook](notebooks/PhotonIQ_Real_Experimental_Training.ipynb) |

These checks demonstrate that the code can execute its intended workflow on test inputs. They **do not** demonstrate successful film-thickness prediction on real experimental specimens.

### Experimental model results — pending real-data training

The predeclared comparison uses **held-out physical films**, with top/middle/bottom scans from any one film kept in the same split. All numbers below remain unreported until the original experimental dataset has been processed and the evaluation artifacts are inspected.

| Method | Held-out MAE (nm) | Held-out RMSE (nm) | Held-out R² |
|---|---|---|---|
| Training-set mean baseline | Not yet measured | Not yet measured | Not yet measured |
| Ridge regression (full spectrum) | Not yet measured | Not yet measured | Not yet measured |
| Mask-aware CNN — 16 wavelengths | Not yet measured | Not yet measured | Not yet measured |
| Mask-aware CNN — 32 wavelengths | Not yet measured | Not yet measured | Not yet measured |
| Mask-aware CNN — 64 wavelengths | Not yet measured | Not yet measured | Not yet measured |
| Mask-aware CNN — 128 wavelengths | Not yet measured | Not yet measured | Not yet measured |

**Additional required reporting:** independent film counts per split; clustered 95% bootstrap interval for MAE; conformal interval half-width and test-film coverage; source archive SHA-256; random seed; spectral match rate; and any excluded samples. Sparse-channel results correspond to **fixed masks, not adaptive wavelength selection**.

### Producing publishable experimental results

1. Download the **original experimental dataset ZIP** from [Mendeley Data](https://data.mendeley.com/datasets/zt2j5j4v44/3).
2. Run [PhotonIQ in Google Colab](notebooks/PhotonIQ_Real_Experimental_Training.ipynb) with that archive, or run the CLI command in [Run locally](#run-locally).
3. Inspect `data_audit.json`, `split_manifest.csv`, and `run_metadata.json` for specimen alignment, sample counts and reproducibility.
4. Transfer the *actual* model values from `test_metrics.csv` and `conformal_evaluation.json` into this section only after confirming successful real-data training. Do not substitute the synthetic test-fixture output.

The original raw experimental files and trained weights are intentionally excluded from Git version control. Until this evaluation completes, **the central thickness-prediction hypothesis remains untested on real data**.

### Data and methodology at a glance

```text
Published real measurements (CC BY 4.0)
        │
        ├── UV–Vis TXT files (TOP, MID, BOT)
        └── independently measured thickness (XLSX)
        │
        ▼
Strict source parser + ID/position join + diagnostics
        │
        ▼
Grouped film-level splits: train / validation / untouched test
        │
        ▼
Train-only wavelength-intensity standardization
        │
        ├── Mean baseline
        ├── Ridge regression baseline
        └── Mask-aware 1D CNN (random missing wavelength inputs)
        │
        ▼
Validation-based early stopping + grouped conformal calibration
        │
        ▼
Independent test film evaluation + cluster bootstrap intervals
        │
        ▼
Model, metrics, split manifest, original archive SHA256
```

| Design choice | Why it matters |
|---|---|
| **Real measured UV–Vis, not synthetic training spectra** | Optical input and thickness labels have experimental provenance |
| **Same-film grouping** | TOP/MID/BOT measurements of one specimen cannot straddle train/test |
| **Thickness never included in input engineering** | Prevents target leakage through features such as thickness-normalized UV–Vis |
| **Ridge and mean baselines** | Tests whether a CNN is genuinely warranted |
| **Wavelength masking (16, 32, 64, 128)** | Tests inference under sparse fixed spectral acquisition |
| **Groupwise conformal calibration** | Intervals are calibrated using validation **films**, not correlated scan rows |
| **Cluster bootstrap over films** | More informative uncertainty estimate than treating individual scans as independent |
| **Archive checksum and manifest** | Allows results to be traced to one source archive and one fixed split |

### Honest boundaries

PhotonIQ v1 is **an empirical neural spectral regression system**, not yet a physics-informed thin-film solver, an active wavelength selection algorithm, or a validated industrial measurement device. The sparse experiment uses *fixed uniformly spaced masks*, not adaptive optical acquisition. Experimental evaluation on another perovskite material, laboratory, or instrument is **future work**. These boundaries are documented in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Run locally

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_notebook.py

# Validate source file structure, sample joins, and source checksum:
python -m photoniq audit --archive /path/to/Mendeley_Download_All.zip

# Train and evaluate with real original measurements:
python -m photoniq train --archive /path/to/Mendeley_Download_All.zip \
  --output runs/experiment_seed42 --epochs 150 --seed 42
```

Running `audit` does not train. Running `train` creates a deterministic group-level split, trains on training films, selects the checkpoint using validation films and evaluates the untouched test films. It writes `data_audit.json`, `split_manifest.csv`, `test_metrics.csv`, `test_predictions.csv`, `conformal_evaluation.json`, `learning_curve.csv`, `photoniq_cnn.pt`, `ridge_baseline.joblib`, and `run_metadata.json` to `runs/`. Files in `runs/` and original data are ignored by Git.

### Folder structure

```text
PhotonIQ/
├── notebooks/PhotonIQ_Real_Experimental_Training.ipynb
├── src/photoniq/
│   ├── data.py           # original data parsing, safe extraction, group split
│   ├── model.py          # mask-aware 1D CNN
│   ├── train.py          # real-data experiment + baseline training
│   ├── metrics.py        # film bootstrap + conformal calibration
│   └── __main__.py       # audit/train CLI
├── tests/                # synthetic software-only fixtures, leakage tests
├── tools/                # rebuild/validate self-contained Colab notebook
├── docs/
│   ├── DATASET.md
│   ├── MODEL_CARD.md
│   └── EXPERIMENT_PROTOCOL.md
├── .github/workflows/ci.yml
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

## Reproducibility and experimental integrity

- The original data source is cited by DOI and **not** copied into this repository.
- `tools/build_colab.py` generates a **self-contained notebook** containing the exact current Python modules. CI checks that its embedded source remains identical to Git source.
- CI uses **only synthetic fixtures to test the software**; fake fixture metrics must not be presented as experimental findings.
- The output contains a **SHA256 of the actual input archive**, dataset join summary, per-split independent film counts, hardware/runtime versions, and a per-specimen manifest.
- Before publishing results, review the match fraction, unexpected missing samples, source layout, prediction residuals, and test-film size. A high score on one small experiment is not general proof.

## References and credit

**Experimental data:** Harth, M. & Vesce, L. (2023). *Perovskite-Thin-Films*, Version 3. Mendeley Data. [DOI: 10.17632/zt2j5j4v44.3](https://doi.org/10.17632/zt2j5j4v44.3). Dataset license: **CC BY 4.0**.

**Study:** Harth, M., Vesce, L., Kouroudis, I., et al. (2023), *Optoelectronic perovskite film characterization via machine vision*, Solar Energy, 262, 111840. [DOI: 10.1016/j.solener.2023.111840](https://doi.org/10.1016/j.solener.2023.111840). Original authors' [code repository](https://github.com/Galdramadhur/Perovskite-Thin-Film-Characterization).

Please cite both the source dataset and the original study when reusing their data. The MIT license in this repository applies only to **PhotonIQ-written code**; it does not replace the source dataset's attribution requirement.

## Future research, in order

1. Run and verify the original experimental data archive and its sample/label alignment.
2. Repeat evaluation across multiple grouped splits; report distribution of model vs baseline differences, not just a favorable seed.
3. Test cross-batch and cross-instrument generalization with independent measurements, if available.
4. Introduce actual adaptive wavelength selection and a properly verified optical-physics loss, then run ablations against the honest v1 baseline.

**Repository name suggestion:** `photoniq-real-spectral-learning`.
