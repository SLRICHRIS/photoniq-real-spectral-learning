# Predeclared experimental protocol

**Problem:** supervised regression from measured UV–Vis signal to independently measured film thickness (nm).

**Primary evaluation:** grouped train/validation/test split by physical sample ID at approximately 60/20/20, seed 42; group-disjoint. The held-out *test films* are used once after model training and validation-based early stopping. No synthetic spectra are part of fitting.

**Predeclared baselines:** train-mean and standardized Ridge(alpha=10.0); neural baseline: compact mask-aware 1D CNN with 16–32–32 channels, AdamW lr=0.001, weight_decay=0.01, SmoothL1Loss, dropout=0.15, early stopping on full-spectrum validation loss.

**Primary metric:** MAE in nanometres, shown with independent-film cluster-bootstrap 95% interval; secondary RMSE, film-macro-MAE, R². Always display number of *independent films* and scans.

**Sparse experiment:** test CNN with fixed uniformly spaced 16/32/64/128 channels. This measures tolerance to *missing spectral samples*. It does **not** constitute an adaptive active-sensing algorithm. The mask is an explicit model input; unmeasured values are replaced by zero after training-only scaling.

**Calibration:** 90% nominal groupwise conformal interval using maximum error across positions for each *validation film*, with finite-sample quantile. Test film coverage is reported descriptively. Groupwise exchangeability may fail under material or instrument shift. With too few validation films, width is undefined rather than pretending coverage.

**Randomness:** fixed numpy/PyTorch seeds, deterministic cuDNN mode and stored split manifest. Hardware libraries may still introduce small numerical variation.

**Primary risks:** small data, correlated positions, spurious instrument-specific spectral artifacts, incomplete source matches, fabrication-condition confounding, out-of-distribution inputs. If the CNN underperforms Ridge, publish that result.

**Future work (NOT implemented):** experimental external validation, physics-informed loss with justified thin-film dispersion model, nested grouped cross-validation, pre-registered adaptive wavelength acquisition, domain shift experiments, formal reproducibility release with source archive checksums.

**Do not claim:** "physics guided", "generalizes across semiconductor materials", "real-time instrument-ready", or "high accuracy" until supported by appropriate experiments.
