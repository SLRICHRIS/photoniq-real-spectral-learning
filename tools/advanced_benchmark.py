"""PhotonIQ v2: leakage-aware evaluation of measured optical-density thickness regression.

Source: Zenodo 10.5281/zenodo.19237616 (original third-party dataset, not redistributed).
Research limitations: single dataset, same instrument(s), 20 fabrication dates, no
physical-law constraint, no independent external validation.

Usage:
    python tools/advanced_benchmark.py path/to/Dataset.zip results/advanced

No features calculated from target thickness or published ODint-derived columns.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

SEED = 42
BUDGETS = (16, 32, 64, 128, 256, 351)
N_FOLDS = 5
TARGET_NAME = 'Perovskite_thickness_nm'
GROUP_NAME = 'Date'
SOURCE_DOI = '10.5281/zenodo.19237616'


def reproducible(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_data(archive: Path):
    if not zipfile.is_zipfile(archive):
        raise ValueError('Invalid ZIP archive; no synthetic fallback permitted')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        dbname = next((x for x in names if x.endswith('blade_perovskite_database.csv')), None)
        spname = next((x for x in names if x.endswith('curvedata_opticaldensity.csv')), None)
        if not dbname or not spname:
            raise ValueError('Source archive lacks original experimental database or optical curves')
        database = pd.read_csv(io.BytesIO(z.read(dbname)))
        curve = pd.read_csv(io.BytesIO(z.read(spname)))
    wavelength = pd.to_numeric(curve['Wavelength_nm'], errors='coerce').to_numpy(float)
    if not np.isfinite(wavelength).all() or not np.all(np.diff(wavelength) > 0):
        raise ValueError('Wavelength grid must be strictly increasing and finite')
    if database.Sample.duplicated().any():
        raise ValueError('Duplicate specimen identifiers')
    Xdf = curve.set_index('Wavelength_nm').apply(pd.to_numeric, errors='coerce').T
    Xdf.columns = wavelength
    matched = database.loc[database.Sample.isin(Xdf.index)].copy()
    matched[TARGET_NAME] = pd.to_numeric(matched[TARGET_NAME], errors='coerce')
    matched = matched.loc[matched[TARGET_NAME].notna() & (matched[TARGET_NAME] > 0)].copy()
    sample_ids = matched.Sample.astype(str).tolist()
    raw = Xdf.loc[sample_ids].to_numpy(float)
    valid_frac = np.mean(np.isfinite(raw), axis=1)
    mask = valid_frac >= 0.8
    matched = matched.loc[mask].reset_index(drop=True)
    raw = raw[mask]
    # Interpolation is within the measured optical channels for each sample,
    # independent of other films and target labels.
    X = np.vstack([np.interp(wavelength, wavelength[np.isfinite(x)], x[np.isfinite(x)])
                   for x in raw]).astype(np.float32)
    y = matched[TARGET_NAME].to_numpy(float)
    groups = matched[GROUP_NAME].astype(str).to_numpy()
    if len(y) < 100 or len(set(groups)) < N_FOLDS:
        raise ValueError('Insufficient labelled experimental data or fabrication groups')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite values survived cleaning')
    audit = {
        'original_dataset_rows':len(database), 'original_spectral_columns':len(Xdf),
        'valid_joined_films':len(X), 'discarded_rows':len(database)-len(X),
        'distinct_fabrication_dates':len(set(groups)), 'n_wavelengths':len(wavelength),
        'wavelength_min_nm':float(wavelength.min()),
        'wavelength_max_nm':float(wavelength.max()),
        'source_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'source_doi':SOURCE_DOI,
        'raw_data_used_as_model_inputs':'curvedata_opticaldensity.csv only',
        'excluded_target_derived_features':'Perovskite_ODint_* and Perovskite_thickness_nm',
        'raw_optical_curves_interpolation':'per-film, no cross-film statistics',
        'missing_spectral_values_filled':int(np.size(raw)-np.isfinite(raw).sum())
    }
    return matched, X, y, groups, wavelength, audit


class SpectralCNN(nn.Module):
    """Small 1D CNN (retrained independently for each sparse wavelength budget)."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(1,16,7,padding=3),nn.GELU(),nn.AvgPool1d(2),
            nn.Conv1d(16,32,5,padding=2),nn.GELU(),nn.AvgPool1d(2),
            nn.AdaptiveAvgPool1d(8),nn.Flatten(),nn.Linear(256,64),
            nn.GELU(),nn.Linear(64,1))
    def forward(self,x):
        return self.net(x.unsqueeze(1)).flatten()


def train_cnn(x_train, y_train, x_val, y_val, seed, max_epochs=150, patience=20):
    reproducible(seed)
    feature_scaler = StandardScaler().fit(x_train)
    tr = torch.from_numpy(feature_scaler.transform(x_train).astype(np.float32))
    va = torch.from_numpy(feature_scaler.transform(x_val).astype(np.float32))
    mean, std = float(y_train.mean()), float(y_train.std())
    if std <= 0: raise ValueError('Constant training target')
    yt = torch.from_numpy(((y_train-mean)/std).astype(np.float32))
    yv = torch.from_numpy(((y_val-mean)/std).astype(np.float32))
    model=SpectralCNN()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed)
    best_loss=float('inf'); best_epoch=0; best_state=None
    hist=[]; stale=0
    for epoch in range(1,max_epochs+1):
        model.train()
        indices=rng.permutation(len(tr))
        for start in range(0,len(indices),256):
            batch=indices[start:start+256]
            optimizer.zero_grad()
            loss=(model(tr[batch])-yt[batch]).square().mean()
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss=float((model(va)-yv).square().mean().item())
        hist.append({'epoch':epoch,'validation_scaled_mse':val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss=val_loss;best_epoch=epoch;stale=0
            best_state={key: val.detach().clone() for key,val in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(best_state); model.eval()
    return model,feature_scaler,mean,std,best_epoch,hist


def infer(model,scaler,mean,std,X):
    with torch.no_grad():
        pred=model(torch.from_numpy(scaler.transform(X).astype(np.float32))).cpu().numpy()
    return pred*std+mean


def evaluate(y,pred):
    return {'MAE_nm':float(mean_absolute_error(y,pred)),
            'RMSE_nm':float(np.sqrt(mean_squared_error(y,pred))),
            'R2':float(r2_score(y,pred))}


def descriptive_bootstrap(predictions:pd.DataFrame, n_boot=10000, seed=42):
    # Resample fabrication dates, not individual specimens; repeated CV test
    # predictions are OOF, but only 20 independent fabrication groups exist.
    rng=np.random.default_rng(seed)
    dates=predictions['fabrication_date'].unique()
    group_ix={date:np.flatnonzero(predictions.fabrication_date.to_numpy()==date) for date in dates}
    index_groups=[group_ix[date] for date in dates]
    cols={c:predictions[c].to_numpy(float) for c in ['actual_thickness_nm','Mean baseline','Spectral-statistics Ridge','Full-spectrum Ridge','Full-spectrum CNN']}
    m=len(dates)
    values={k:np.empty(n_boot,float) for k in ['CNN_MAE_nm','Ridge_MAE_nm','CNN_vs_Ridge_MAE_reduction_pct']}
    for i in range(n_boot):
        picked=rng.integers(0,m,size=m)
        indices=np.concatenate([index_groups[k] for k in picked])
        actual=cols['actual_thickness_nm'][indices]
        cnn=np.mean(abs(actual-cols['Full-spectrum CNN'][indices]))
        ridge=np.mean(abs(actual-cols['Full-spectrum Ridge'][indices]))
        values['CNN_MAE_nm'][i]=cnn
        values['Ridge_MAE_nm'][i]=ridge
        values['CNN_vs_Ridge_MAE_reduction_pct'][i]=100*(ridge-cnn)/ridge if ridge>0 else float('nan')
    return {k:{'p2.5':float(np.nanpercentile(v,2.5)),'p97.5':float(np.nanpercentile(v,97.5))}
            for k,v in values.items()}


def figures(output,metrics,byfold,diag,pred):
    plt.rcParams.update({'font.size':10})
    # Plot saved as SVG: plain-text GitHub friendly, no binary dependencies.
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    full=metrics.loc[metrics.model.isin(['Mean baseline','Spectral-statistics Ridge','Full-spectrum Ridge','Full-spectrum CNN'])]
    ax.barh(full.model,full.MAE_nm)
    ax.set_xlabel('Out-of-fold MAE (nm) · lower is better')
    ax.invert_yaxis();fig.tight_layout();fig.savefig(output/'model_comparison.svg');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,4.4))
    m=metrics.loc[metrics.model.str.match(r'(Ridge|CNN) - [0-9]+ wavelengths')].copy()
    for typ in ['Ridge','CNN']:
        pts=m[m.model.str.startswith(typ+' - ')].sort_values('wavelength_budget')
        ax.plot(pts.wavelength_budget,pts.MAE_nm,marker='o',label=typ)
    ax.set_xscale('log',base=2);ax.set_xticks(list(BUDGETS),[str(v) for v in BUDGETS]);ax.set_xlabel('Retained wavelength channels')
    ax.set_ylabel('Out-of-fold MAE (nm)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'sparse_wavelength_tradeoff.svg');plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5.3))
    ax.scatter(pred.actual_thickness_nm,pred['Full-spectrum CNN'],alpha=.5,s=18)
    low=float(min(pred.actual_thickness_nm.min(),pred['Full-spectrum CNN'].min()))
    high=float(max(pred.actual_thickness_nm.max(),pred['Full-spectrum CNN'].max()))
    ax.plot([low,high],[low,high],ls='--',label='Ideal prediction')
    ax.set_xlabel('Experimentally measured thickness (nm)');ax.set_ylabel('CNN predicted thickness (nm)')
    ax.legend();fig.tight_layout();fig.savefig(output/'actual_vs_predicted.svg');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.6))
    sub=diag.loc[diag.dimension=='fabrication_date'].sort_values('CNN_MAE_nm',ascending=False)
    ax.bar(sub.group,sub.CNN_MAE_nm);ax.tick_params(axis='x',rotation=75)
    ax.set_ylabel('CNN MAE by fabrication date (nm)');fig.tight_layout();fig.savefig(output/'error_by_fabrication_date.svg');plt.close(fig)


def run(archive: Path,output: Path,folds=5,epochs=150,bootstrap=10000):
    output.mkdir(parents=True,exist_ok=True)
    reproducible(SEED)
    db,X,y,groups,wl,audit=load_data(archive)
    if folds > len(set(groups)): raise ValueError('More folds than independent fabrication dates')
    audit['validation_method']=f'{folds}-fold GroupKFold over fabrication dates, with group-disjoint inner validation'
    audit['budgets']=list(BUDGETS)
    (output/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    gkf=GroupKFold(n_splits=folds)
    predictions=[];fold_scores=[];learning=[];fold_manifest=[];split_manifest=[]
    (output/'checkpoints').mkdir(exist_ok=True)
    for fold,(development,test) in enumerate(gkf.split(X,y,groups),start=1):
        gss=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=SEED+fold)
        trrel,varel=next(gss.split(X[development],y[development],groups[development]))
        train=development[trrel];valid=development[varel]
        for a,b in [(train,test),(train,valid),(valid,test)]:
            assert not(set(groups[a])&set(groups[b])), 'Group leakage detected'
        base=DummyRegressor(strategy='mean').fit(X[train],y[train]).predict(X[test])
        # An explicit low-dimensional physics-oriented comparator; all features
        # come directly from measured OD and contain no target-derived columns.
        stat_features=np.stack([X.mean(axis=1),X.std(axis=1),
                        X[:,(wl>=400)&(wl<=700)].mean(axis=1)],axis=1)
        desc=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(stat_features[train],y[train])
        pred_desc=desc.predict(stat_features[test])
        fold_pred=pd.DataFrame({'sample':db.Sample.iloc[test].to_numpy(),
                                'fabrication_date':groups[test], 'fold':fold,
                                'actual_thickness_nm':y[test],
                                'thickness_bin':pd.cut(y[test],[0,300,600,np.inf],
                                    labels=['<300 nm','300–600 nm','>600 nm']).astype(str),
                                'composition':db.Perovskite_synthesis_composition_short.iloc[test].astype(str).to_numpy(),
                                'substrate':db.Substrate_stack_sequence.iloc[test].astype(str).to_numpy(),
                                'humidity_pct':pd.to_numeric(db.Perovskite_deposition_relative_humidity_pct.iloc[test],errors='coerce').to_numpy(),
                                'blade_temperature_C':pd.to_numeric(db.Perovskite_deposition_blade_temperature_C.iloc[test],errors='coerce').to_numpy(),
                                'Mean baseline':base,
                                'Spectral-statistics Ridge':pred_desc})
        for label,pred in [('Mean baseline',base),('Spectral-statistics Ridge',pred_desc)]:
            fold_scores.append(dict(fold=fold,model=label,wavelength_budget=0,test_films=len(test),**evaluate(y[test],pred)))
        for budget in BUDGETS:
            keep=np.unique(np.rint(np.linspace(0,len(wl)-1,budget)).astype(int))
            if len(keep) != budget: raise ValueError('Budget selection did not produce unique wavelengths')
            xb=X[:,keep]
            reg=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(xb[train],y[train])
            p_ridge=reg.predict(xb[test])
            model,scale,mu,sd,best,trace=train_cnn(xb[train],y[train],xb[valid],y[valid],
                                                    SEED+fold*100+budget,epochs)
            p_cnn=infer(model,scale,mu,sd,xb[test])
            if budget in (64,351):
                torch.save({'state_dict':model.state_dict(), 'architecture':'SpectralCNN',
                            'fold':fold,'channels':budget,'wavelengths_nm':wl[keep].tolist(),
                            'feature_mean':scale.mean_.tolist(),'feature_scale':scale.scale_.tolist(),
                            'target_mean_nm':mu,'target_scale_nm':sd,
                            'source_sha256':audit['source_sha256']},
                           output/'checkpoints'/f'cnn_{budget}wl_fold{fold}.pt')
            for typ,p in [('Ridge',p_ridge),('CNN',p_cnn)]:
                label=f'{typ} - {budget} wavelengths'
                fold_pred[label]=p
                fold_scores.append(dict(fold=fold,model=label,wavelength_budget=budget,
                                        test_films=len(test),**evaluate(y[test],p)))
            learning.append(dict(fold=fold,wavelength_budget=budget,best_epoch=best,epochs_run=len(trace),
                                 best_validation_scaled_mse=float(min(v['validation_scaled_mse'] for v in trace))))
        fold_pred['Full-spectrum CNN']=fold_pred['CNN - 351 wavelengths']
        fold_pred['Full-spectrum Ridge']=fold_pred['Ridge - 351 wavelengths']
        predictions.append(fold_pred)
        for i in test: fold_manifest.append(dict(sample=db.Sample.iloc[i],fabrication_date=groups[i],fold=fold))
        for split_name, split_indices in [('train',train),('validation',valid),('test',test)]:
            for i in split_indices:
                split_manifest.append(dict(sample=db.Sample.iloc[i],fabrication_date=groups[i],
                                           fold=fold,split=split_name))
        print(f'Fold {fold}/{folds} | test films={len(test)}, test dates={len(set(groups[test]))} '
              f'| CNN 351 MAE {mean_absolute_error(y[test],fold_pred["Full-spectrum CNN"]):.2f} nm',flush=True)
    pred=pd.concat(predictions,ignore_index=True).sort_values('sample')
    assert len(pred)==len(X) and pred['sample'].nunique()==len(X), 'OOF coverage integrity failure'
    scores=pd.DataFrame(fold_scores)
    all_models=['Mean baseline','Spectral-statistics Ridge']+[f'{t} - {b} wavelengths' for b in BUDGETS for t in ('Ridge','CNN')]
    summary=[]
    for model in all_models:
        pooled=evaluate(pred.actual_thickness_nm.to_numpy(),pred[model].to_numpy())
        sub=scores.loc[scores.model==model]
        summary.append(dict(model=model,wavelength_budget=0 if 'baseline' in model or model=='Spectral-statistics Ridge' else int(model.split(' - ')[1].split(' ')[0]),
                            test_films_total=len(pred),independent_fabrication_dates=len(set(groups)),
                            mean_fold_MAE_nm=float(sub.MAE_nm.mean()),sd_fold_MAE_nm=float(sub.MAE_nm.std(ddof=1)),
                            **pooled))
    results=pd.DataFrame(summary)
    # Pre-registered exploratory groups: thickness, composition, substrate, date.
    diag=[]
    pred['humidity_bin']=pd.cut(pred.humidity_pct,[-np.inf,40,60,np.inf],
                               labels=['RH ≤40%','RH 40–60%','RH >60%']).astype(str)
    pred['blade_temperature_bin']=pd.cut(pred.blade_temperature_C,[-np.inf,40,70,np.inf],
                               labels=['Blade ≤40°C','Blade 40–70°C','Blade >70°C']).astype(str)
    for dimension in ['thickness_bin','composition','substrate','fabrication_date','humidity_bin','blade_temperature_bin']:
        for value,sub in pred.groupby(dimension,dropna=False):
            if len(sub)<5:continue
            err=sub['Full-spectrum CNN']-sub.actual_thickness_nm
            diag.append(dict(dimension=dimension,group=str(value),n_films=len(sub),
                             CNN_MAE_nm=float(err.abs().mean()),CNN_bias_nm=float(err.mean()),
                             Ridge_MAE_nm=float((sub['Full-spectrum Ridge']-sub.actual_thickness_nm).abs().mean())))
    diagnostic=pd.DataFrame(diag)
    pred.to_csv(output/'oof_predictions.csv',index=False)
    scores.to_csv(output/'per_fold_metrics.csv',index=False)
    results.to_csv(output/'cross_validation_summary.csv',index=False)
    pd.DataFrame(learning).to_csv(output/'training_summary.csv',index=False)
    diagnostic.to_csv(output/'error_diagnostics.csv',index=False)
    pd.DataFrame(fold_manifest).to_csv(output/'fold_manifest.csv',index=False)
    pd.DataFrame(split_manifest).to_csv(output/'full_split_manifest.csv',index=False)
    boot=descriptive_bootstrap(pred,n_boot=bootstrap,seed=SEED)
    (output/'cluster_bootstrap.json').write_text(json.dumps({'source':'Out-of-fold sample-level predictions; one prediction per film',
                         'cluster_variable':'fabrication_date','n_clusters':len(set(groups)),
                         'n_resamples':bootstrap,'percentile_intervals':boot,
                         'interpretation':'Descriptive 95% bootstrap percentile intervals, not confidence guarantees for new labs.'},indent=2)+'\n')
    figures(output,results,scores,diagnostic,pred)
    (output/'run_metadata.json').write_text(json.dumps({**audit,'seed':SEED,'folds':folds,
              'outer_split':'GroupKFold grouped on fabrication date',
              'inner_validation':'GroupShuffleSplit on development fabrication dates, random_state=42+fold',
              'spectral_budgets':'uniform wavelengths selected independently for every training and inference run',
              'training':'Independent CNN trained per fold and per budget; AdamW lr=0.001, 150 max epochs, patience 20',
              'test_protocol':'Every film predicted exactly once by an unseen-date fold',
              'software_version':{'python': __import__('platform').python_version(),
                                  'torch':torch.__version__,'numpy':np.__version__,
                                  'pandas':pd.__version__,'sklearn':__import__('sklearn').__version__},
              'publication_caution':'Not prospective, no external laboratory, no calibrated production metrology.'},indent=2)+'\n')
    print('\nOUT-OF-FOLD SUMMARY:')
    print(results[['model','MAE_nm','RMSE_nm','R2']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('Artifacts in',output)
    return results,diagnostic

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--folds',type=int,default=N_FOLDS)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args()
    run(args.archive,args.output,args.folds,args.epochs,args.bootstrap)
