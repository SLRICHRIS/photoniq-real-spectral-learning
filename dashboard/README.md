# PhotonIQ — Interactive Research Dashboard

A **client-side, dependency-free** visual companion to [PhotonIQ's real-data five-fold research benchmark](https://github.com/SLRICHRIS/photoniq-real-spectral-learning/tree/main/results/zenodo_2026/advanced_cv).

## Open

Open `dashboard/index.html` in a browser. No Python server, build tool, package installation, external API or internet connection is required to display the dashboard. External links to the GitHub sources require internet access. A public GitHub Pages URL is **not** claimed unless Pages has been activated and verified separately.

## Sections

- **Overview:** measured-film counts, data provenance, headline results and research pipeline summary.
- **Pipeline explorer:** inspect each of seven grounded stages, inputs, outputs and leakage safeguards.
- **Model results:** interactive metric (MAE/RMSE/R²) and model family selectors for six wavelength budgets, and the complete exact result table.
- **Error diagnostics:** thickness, date, substrate, humidity, and blade-temperature breakdowns from the out-of-fold experiment.
- **Reproducibility:** dataset DOI and SHA256, frozen run metadata, links to source code, numeric artifacts and independent verifier.

## Source of truth

`dashboard/data/snapshot.js` is **generated** from the committed files under `results/zenodo_2026/advanced_cv/`: `cross_validation_summary.csv`, `per_fold_metrics.csv`, `error_diagnostics.csv`, `run_metadata.json`, `data_audit.json`, `verification.json`, `cluster_bootstrap.json`. Its content is frozen as of **28 October 2026** and is clearly labelled as a **saved experimental result** rather than a live model, sensor or CI monitor.

Never paste fabricated accuracy scores into this snapshot. To update it, rerun the original benchmark, validate metrics using `tools/verify_advanced_results.py`, regenerate the snapshot from those exact outputs, then commit the new data and documentation together.

The **unexecuted Mendeley/Colab training implementation is distinct** from the successfully evaluated Zenodo experiment. No industrial deployment, physics-informed solver, automated active sensing or external laboratory validation is implied.

## Quick check

`node dashboard/tests/smoke.cjs`

The test checks the supplied research snapshot against key numerical and scientific consistency requirements; it does not replace Python training or browser QA.
