# Model card · PhotonIQ 1.0

**Model:** MaskAwareSpectralCNN, 1D convolutional regressor with two channels (scaled UV–Vis intensity and observation mask), outputs one number (film thickness, nm). Training masks are random subsets; final checkpoint selected by grouped validation loss.

**Intended use:** research benchmark for real-data thin-film spectroscopy; experimentation with reduced spectral sampling.

**Not intended for:** manufacturing acceptance decisions, medical devices, measurements of other materials without retraining, published claims of accuracy before independently testing.

**Data provenance:** experimental Harth–Vesce Mendeley dataset v3, original source ZIP provided by user. Raw data are not shipped in the repository.

**Evaluation:** grouped independent-film split, Ridge and mean baselines, MAE/RMSE/R² and film macro MAE, cluster-bootstrap uncertainty, validation-film conformal coverage.

**Known limits:** small dataset, no external domain validation, possible correlated fabrication batches, not a physics-informed network, no experimental active-wavelength acquisition.

**Current model status:** NO public trained real-data checkpoint or verified test metrics included. After running Colab, the dated output contains actual results. Publish the generated results with source hash only after checking data provenance and leakage.
