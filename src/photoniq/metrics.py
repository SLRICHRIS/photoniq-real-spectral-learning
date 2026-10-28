"""Experimental metrics and film-level bootstrap uncertainty."""
from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    groups = np.asarray(groups)
    errors = np.abs(y - pred)
    film_errors = [float(np.mean(errors[groups == g])) for g in np.unique(groups)]
    return dict(MAE_nm=float(mean_absolute_error(y, pred)),
                RMSE_nm=float(np.sqrt(mean_squared_error(y, pred))),
                film_macro_MAE_nm=float(np.mean(film_errors)),
                R2=float(r2_score(y, pred)) if len(y) > 1 else float('nan'),
                scans=int(len(y)), independent_films=int(len(film_errors)))


def film_bootstrap_mae_ci(y: np.ndarray, pred: np.ndarray, groups: np.ndarray,
                          n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Resample independent films (not scan rows); report scan-weighted MAE distribution."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        draw = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([np.where(groups == g)[0] for g in draw])
        values.append(float(np.abs(y[chosen] - pred[chosen]).mean()))
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def film_conformal_width(y_val: np.ndarray, pred_val: np.ndarray, groups: np.ndarray,
                         alpha: float = .1) -> tuple[float, dict]:
    """Group maximum-error conformal width; large if calibration set is small.

    Groupwise calibration gives an interval covering all *positions* on a new film
    under film-level exchangeability; it is not a guaranteed calibrated interval
    under arbitrary domain shift. No test labels influence the width.
    """
    unique = np.unique(groups)
    residuals = np.abs(y_val - pred_val)
    scores = np.sort(np.array([residuals[groups == g].max() for g in unique]))
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float('inf'), dict(calibration_films=n, nominal_coverage=1-alpha,
                                  finite_sample_width_available=False,
                                  note='Too few calibration films for a finite distribution-free width')
    return float(scores[rank - 1]), dict(calibration_films=n,
                                         nominal_coverage=1-alpha,
                                         finite_sample_width_available=True)
