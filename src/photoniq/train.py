"""Reproducible training/evaluation on real thin-film measurements.

No test-informed tuning. No synthetic fallback. No target-derived feature computation.
"""
from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import __version__
from .data import file_sha256, grouped_split, load_experimental, unpack_source
from .metrics import film_bootstrap_mae_ci, film_conformal_width, score
from .model import MaskAwareSpectralCNN, mask_spectra


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, os.cpu_count() or 2))


def save_json(path: Path, data: dict | list):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding='utf-8')


def fit_cnn(X: np.ndarray, y: np.ndarray, splits: dict,
            epochs: int = 150, patience: int = 20, seed: int = 42,
            device: str = 'cpu', batch_size: int = 16) -> tuple:
    # X: training-standardized spectra; y: training-standardized targets
    net = MaskAwareSpectralCNN().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss(beta=0.6)
    tr, va = splits['train'], splits['val']
    rng = np.random.default_rng(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                        batch_size=batch_size, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    validation_x = torch.from_numpy(X[va]).to(device)
    validation_y = torch.from_numpy(y[va]).to(device)
    best = float('inf'); state = None; idle = 0; history = []
    for epoch in range(1, epochs + 1):
        net.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(mask_spectra(xb, rng=rng))
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        net.eval()
        with torch.no_grad():
            vpred = net(mask_spectra(validation_x, keep=X.shape[1]))
            vloss = float(loss_fn(vpred, validation_y).item())
        history.append(dict(epoch=epoch, train_loss=np.mean(losses), val_loss=vloss))
        if vloss + 1e-5 < best:
            best, idle = vloss, 0
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            idle += 1
        if idle >= patience:
            break
    assert state is not None
    net.load_state_dict(state)
    return net, pd.DataFrame(history), best


def cnn_predict(net, X: np.ndarray, indices: np.ndarray, device: str,
                ymean: float, ystd: float, keep: int | None = None) -> np.ndarray:
    net.eval()
    x = torch.from_numpy(X[indices]).to(device)
    with torch.no_grad():
        result = net(mask_spectra(x, keep=keep or X.shape[1])).detach().cpu().numpy()
    return result * ystd + ymean


def run_experiment(archive: str | Path, output: str | Path, *,
                   epochs: int = 150, patience: int = 20,
                   seed: int = 42, n_boot: int = 1000,
                   device: str = 'auto', grid_points: int = 128,
                   min_match_fraction: float = .65) -> dict:
    archive = Path(archive).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    real_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if real_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but no GPU is available')
    t0 = time.perf_counter()
    print('Dataset: EXPERIMENTAL Mendeley Harth–Vesce v3 (no synthetic fallback)')
    print('Source archive SHA256:', file_sha256(archive))
    files = unpack_source(archive, output / 'extracted_original_dataset')
    dataset = load_experimental(output / 'extracted_original_dataset', grid_points, min_match_fraction)
    print('Files:', len(files), '| Matched scans:', len(dataset.y),
          '| Independent films:', len(np.unique(dataset.group)))
    splits = grouped_split(dataset.group, seed)
    for name, ix in splits.items():
        print(name, ':', len(ix), 'scans /', len(np.unique(dataset.group[ix])), 'independent films')

    # Film-level split is fixed before fit; fit scaler/target scaling on TRAIN ONLY.
    scaler = StandardScaler().fit(dataset.X[splits['train']])
    X = scaler.transform(dataset.X).astype('float32')
    ymean = float(dataset.y[splits['train']].mean())
    ystd = float(dataset.y[splits['train']].std())
    if ystd < 1e-6:
        raise ValueError('Insufficient training target variation')
    yscaled = ((dataset.y - ymean) / ystd).astype('float32')

    split_by_index = np.empty(len(dataset.y), dtype=object)
    for name, ix in splits.items():
        split_by_index[ix] = name
    manifest = pd.DataFrame(dict(sample_id=dataset.group, position=dataset.position,
                                 thickness_nm=dataset.y, source_path=dataset.path,
                                 split=split_by_index))
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    save_json(output / 'data_audit.json', dataset.audit)

    # Baselines: hyperparameters fixed in protocol; never chosen using test performance.
    mean_baseline = DummyRegressor(strategy='mean').fit(X[splits['train']], dataset.y[splits['train']])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(dataset.X[splits['train']], dataset.y[splits['train']])

    net, history, best_val = fit_cnn(X, yscaled, splits, epochs, patience, seed, real_device)
    history.to_csv(output / 'learning_curve.csv', index=False)

    # Calibrate on validation films, using maximum position error per physical film.
    val_ix = splits['val']; test_ix = splits['test']
    val_pred = cnn_predict(net, X, val_ix, real_device, ymean, ystd)
    width, conformal_info = film_conformal_width(dataset.y[val_ix], val_pred,
                                                  dataset.group[val_ix], alpha=.1)
    truth = dataset.y[test_ix]; test_groups = dataset.group[test_ix]
    models = {'Training-mean baseline': mean_baseline.predict(X[test_ix]),
              'Ridge baseline': ridge.predict(dataset.X[test_ix])}
    for keep in [16, 32, 64, grid_points]:
        if keep <= grid_points:
            models[f'Mask-aware CNN ({keep}/{grid_points} wavelengths)'] = (
                cnn_predict(net, X, test_ix, real_device, ymean, ystd, keep=keep))

    rows = []; predictions = manifest.iloc[test_ix].copy()
    for name, pred in models.items():
        s = score(truth, pred, test_groups)
        low, high = film_bootstrap_mae_ci(truth, pred, test_groups, n_boot, seed)
        rows.append(dict(model=name, **s, MAE_95pct_cluster_bootstrap_low_nm=low,
                         MAE_95pct_cluster_bootstrap_high_nm=high))
        if 'wavelengths)' in name and name.endswith(f'({grid_points}/{grid_points} wavelengths)'):
            predictions['cnn_full_prediction_nm'] = pred
        if name == 'Ridge baseline':
            predictions['ridge_prediction_nm'] = pred
    results = pd.DataFrame(rows)
    results.to_csv(output / 'test_metrics.csv', index=False)
    predictions.to_csv(output / 'test_predictions.csv', index=False)

    full = models[f'Mask-aware CNN ({grid_points}/{grid_points} wavelengths)']
    if np.isfinite(width):
        predictions['cnn_interval_low_nm'] = full - width
        predictions['cnn_interval_high_nm'] = full + width
        position_coverage = float(np.mean(np.abs(truth-full) <= width))
        film_coverage = float(np.mean([
            bool(np.all(np.abs(truth[test_groups == group] - full[test_groups == group]) <= width))
            for group in np.unique(test_groups)]))
    else:
        position_coverage = None
        film_coverage = None
    conformal_info.update(dict(half_width_nm=width if np.isfinite(width) else None,
                               test_position_coverage=position_coverage,
                               test_film_coverage=film_coverage,
                               note='Groupwise exchangeability assumption; not valid under unrestricted domain shift'))
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    save_json(output / 'conformal_evaluation.json', conformal_info)

    torch.save(dict(model_state=net.cpu().state_dict(), architecture='MaskAwareSpectralCNN',
                    feature_grid_nm=dataset.wavelengths.tolist(),
                    feature_mean=scaler.mean_.tolist(), feature_scale=scaler.scale_.tolist(),
                    target_mean_nm=ymean, target_sd_nm=ystd, training_seed=seed,
                    dataset_source_sha256=file_sha256(archive)), output / 'photoniq_cnn.pt')
    import joblib
    joblib.dump(ridge, output / 'ridge_baseline.joblib')
    metadata = dict(software_version=__version__, dataset_doi=dataset.audit['dataset_doi'],
                    dataset_type='experimentally measured UV-Vis + measured thickness',
                    dataset_sha256=file_sha256(archive), source_licence='CC BY 4.0',
                    seed=seed, device=real_device, python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                    independent_films=int(len(set(dataset.group))), total_scans=len(dataset.y),
                    split_film_counts={k:len(set(dataset.group[v])) for k,v in splits.items()},
                    training_epochs=int(len(history)), best_validation_loss=float(best_val),
                    seconds=float(time.perf_counter()-t0),
                    limitations=['Single experimental dataset; domain shift not tested',
                                 'Small independent-film counts; uncertainty intervals may be wide',
                                 'Model is empirical, not physics-informed',
                                 'Sparse masks represent observed channels, not active wavelength selection'])
    save_json(output / 'run_metadata.json', metadata)
    print(results.to_string(index=False, float_format=lambda a: f'{a:.4f}'))
    print('Outputs:', output)
    return dict(data=dataset.audit, metrics=results.to_dict(orient='records'),
                conformal=conformal_info, metadata=metadata)
