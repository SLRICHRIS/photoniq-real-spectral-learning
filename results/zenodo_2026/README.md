# Real experimental benchmark — Zenodo 2026

**Scientific scope:** Regression of measured film thickness (nm) from measured optical-density spectra of blade-coated perovskite films. **Source:** [Zenodo record 19237616](https://zenodo.org/records/19237616), CC BY 4.0. This is **not** the separate Mendeley benchmark.

## Data and split

- Source SHA-256: `c79b616d2c998977557cf72017c00f10b60a953570fe939932d025b208a2eb62`
- 517 experimental films, 351 spectral wavelengths (300–1000 nm).
- Date-grouped train/validation/test: 343/69/105 films; 12/4/4 fabrication dates.
- Training seed 42. Only training-set statistics used for feature and target normalization.
- CNN selected by validation loss at epoch 17; stopped after 37 epochs.

## Held-out test accuracy

| Method | MAE (nm) | RMSE (nm) | R² |
|---|---:|---:|---:|
| Training mean | 165.58 | 210.03 | −0.107 |
| Ridge (full spectrum) | 81.49 | 114.82 | 0.669 |
| **1D CNN (full spectrum)** | **50.91** | **74.49** | **0.861** |

**Relative improvement:** 37.5% lower MAE than Ridge for this one split.

## Uncertainty and scientific caveats

A separate post-training verification recalculated the accuracy metrics from the 105 saved test predictions, matching the stored scores. A 20,000-resample **fabrication-date-clustered bootstrap** of these predictions gave the following descriptive 2.5th–97.5th percentile intervals:

| Quantity | Descriptive bootstrap range |
|---|---:|
| CNN MAE | 33.91–67.19 nm |
| Ridge MAE | 60.76–95.08 nm |
| CNN MAE reduction versus Ridge | 18.4–49.3% |

The test data comprise only **four fabrication dates**. These intervals are unstable estimates, not calibrated guarantees of performance on new instruments, dates, compositions, or laboratories. No repeated-seed study, cross-material benchmark, independent external validation or hypothesis test is claimed.

Original inputs: only `curvedata_opticaldensity.csv` spectral values. Excluded as features: `Perovskite_thickness_nm` (target) and derived `Perovskite_ODint_*` fields (avoid target-derived feature contamination). No synthetic samples were used for model fitting.

## Reproducibility

- [Successful training workflow](https://github.com/SLRICHRIS/photoniq-real-spectral-learning/actions/runs/40701586597) (28 October 2026).
- [Download original run artifacts](https://github.com/SLRICHRIS/photoniq-real-spectral-learning/actions/runs/40701586597/artifacts/12822126620) (includes predictions, manifest, learning curve, trained `.pt` checkpoint and metadata; artifacts may expire).
- [Training script](../../tools/train_zenodo.py) · [Workflow definition](../../.github/workflows/train_zenodo.yml).
- [Exact metrics](test_metrics.csv) and [metadata](run_metadata.json) are copied from the completed run.

Raw original Zenodo files and trained weights are **not committed to this repository**. The original Mendeley Harth–Vesce benchmark remains a distinct uncompleted experiment due to an HTTP 401 dataset download error.
