# Dataset card · Experimental Perovskite UV–Vis

**Original dataset**: Harth, Milan & Vesce, Luigi (2023), *Perovskite-Thin-Films*, Version 3, Mendeley Data. DOI: [10.17632/zt2j5j4v44.3](https://doi.org/10.17632/zt2j5j4v44.3).

**License**: CC BY 4.0 (original data). This repository does **not** bundle or claim ownership of the original dataset. Users separately obtain the **Download All** ZIP from the publisher.

**Paper**: *Optoelectronic perovskite film characterization via machine vision*, Solar Energy 262 (2023), 111840, DOI: [10.1016/j.solener.2023.111840](https://doi.org/10.1016/j.solener.2023.111840). Authors' reference code: [Perovskite-Thin-Film-Characterization](https://github.com/Galdramadhur/Perovskite-Thin-Film-Characterization).

## Target and inputs

- **Inputs**: experimental UV–Vis spectral intensity as supplied in the source TXT files (`TOP`, `MID`, `BOT` positions). Signal units are retained as source units; not called absorbance or reflectance without proof.
- **Targets**: experimentally measured film thickness in nanometres from the original Excel workbooks.
- **Sample-level grouping**: different optical scans/positions on the *same physical film* must remain in the same split.
- **Materials**: triple-cation lead bromide-iodide perovskite films, not SiO₂/Si. Do not apply this model to SiO₂ without separate calibration and validation.

## What preprocessing does

The parser reads the published source's four-row transposed thickness Excel format and UV–Vis TXT files (two instrument header lines), joins records by original sample ID and position, rejects ambiguous labels, averages duplicate wavelength readings, then interpolates to a common 128-bin grid. Spectral interpolation is **not** experimental data augmentation. The fitted standardizer and target scale use *training films only*.

**Critical safeguard:** The original authors computed a UV–Vis-derived AUAC metric after dividing spectral intensity by thickness, because their work involved different prediction targets. PhotonIQ MUST NOT reuse that thickness-divided feature when predicting thickness: it would leak ground truth.

## Validity limits

- Dataset describes a specific laboratory, fabrication setup, materials, and instruments; generalization to other devices, materials, and laboratories is untested.
- Repeated spatial positions are correlated; effective sample size is independent films, not number of spectral rows.
- No physical forward optical model is used in this baseline. The project is **not yet physics-informed**.
- The repository contains *no* experimental results until the source archive is supplied and training finishes.
- Small sample sizes can make uncertainty estimates imprecise. Never compare models solely on one favorable random split.
