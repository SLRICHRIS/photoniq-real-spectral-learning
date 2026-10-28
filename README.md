# PhotonIQ

### Experimental spectral learning for perovskite thin-film thickness

**Physics-respecting research discipline, real measured spectra, reproducible Colab training.** PhotonIQ learns a regression relationship between experimental UV–Vis spectral measurements and independently measured thin-film thickness. It compares a compact **mask-aware 1D CNN** with conventional regression baselines, evaluates missing-wavelength robustness, and reports uncertainty at the **physical-film** level.

> **Status:** A real-data baseline and 1D CNN experiment was successfully trained and evaluated on the [Zenodo perovskite optical-density dataset](https://zenodo.org/records/19237616) (517 measured films). The separate Mendeley experiment described in the Colab notebook remains pending because automated source retrieval returned HTTP 401. Do not mix the results of these two datasets.

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

## Results — real experimental optical-density spectra

**Completed 28 October 2026** on the independently published [Zenodo dataset (DOI: 10.5281/zenodo.19237616)](https://zenodo.org/records/19237616). Source archive SHA-256: `c79b616d2c998977557cf72017c00f10b60a953570fe939932d025b208a2eb62`.

**Experiment:** 517 films; 351 wavelength channels spanning 300–1000 nm. Fabrication-date-grouped split: **343 train / 69 validation / 105 test**, representing **12 / 4 / 4** distinct fabrication dates. Test specimens and their dates were not used in training or selecting the CNN checkpoint.

| Model | Test MAE (nm) ↓ | Test RMSE (nm) ↓ | Test R² ↑ |
|---|---:|---:|---:|
| Training-set mean | 165.58 | 210.03 | −0.107 |
| Ridge regression | 81.49 | 114.82 | 0.669 |
| **1D CNN** | **50.91** | **74.49** | **0.861** |

**In this single split**, the CNN attained a **37.5% lower MAE** than Ridge. Its selected checkpoint was from epoch **17** (training stopped after 37 epochs). The metrics were independently recalculated from the saved test predictions, matching the GitHub Actions log.

**Uncertainty and scope:** The test set contains only **four independent fabrication dates**. A descriptive 20,000-resample date-cluster bootstrap of the saved predictions estimated a CNN MAE range of **33.91–67.19 nm** (2.5th–97.5th percentiles). With only four groups, this is **not a strong inferential confidence guarantee**. One random seed and one split do not establish generalization or statistically proven superiority. The inputs were measured optical-density spectra only; thickness and integrated descriptors derived using thickness were excluded. The model is **not physics-informed**, and sparse/adaptive wavelength inference was **not** evaluated here.

**[Detailed Zenodo experimental results](results/zenodo_2026/README.md)** · [Exact numerical metrics](results/zenodo_2026/test_metrics.csv) · [Run metadata](results/zenodo_2026/run_metadata.json) · [Original GitHub Actions run and full artifacts](https://github.com/SLRICHRIS/photoniq-real-spectral-learning/actions/runs/40701586597)

### Original Harth–Vesce Mendeley experiment

The training implementation in [`src/photoniq/`](src/photoniq/) and [Google Colab notebook](notebooks/PhotonIQ_Real_Experimental_Training.ipynb) target a **different** experiment using the original Mendeley archive. That source returned **HTTP 401 Unauthorized** in automated retrieval; this model has **not yet been trained on the Mendeley measurements**. None of the Zenodo results above should be attributed to the Mendeley dataset. The original dataset's training results remain unreported.

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

**Repository:** [`SLRICHRIS/photoniq-real-spectral-learning`](https://github.com/SLRICHRIS/photoniq-real-spectral-learning).
