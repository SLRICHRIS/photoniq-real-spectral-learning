import numpy as np
import torch
from photoniq.model import MaskAwareSpectralCNN, mask_spectra
from photoniq.metrics import film_conformal_width, film_bootstrap_mae_ci, score


def test_masked_model_dimensions_and_no_leakage():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(5,128)), dtype=torch.float32)
    masked = mask_spectra(x, keep=16)
    assert tuple(masked.shape) == (5, 2, 128)
    assert (masked[:,1,:].sum(axis=1)==16).all()
    assert torch.all(masked[:,0,:][masked[:,1,:]==0] == 0)
    net = MaskAwareSpectralCNN()
    assert tuple(net(masked).shape) == (5,)


def test_film_bootstrap_is_deterministic():
    y = np.array([1.,2.,3.,4.,5.,6.]); pred=y+np.array([1,2,0,1,0,3]);g=np.array(['a','a','b','b','c','c'])
    assert film_bootstrap_mae_ci(y,pred,g,n_boot=50,seed=3) == film_bootstrap_mae_ci(y,pred,g,n_boot=50,seed=3)
    s=score(y,pred,g)
    assert s['independent_films']==3 and s['scans']==6


def test_conformal_uses_group_max_not_scan_rows():
    y = np.zeros(20);p = np.arange(20).astype(float)
    groups = np.repeat(np.arange(10), 2)
    q,meta=film_conformal_width(y,p,groups,alpha=.1)
    assert q == 19
    assert meta['calibration_films'] == 10
