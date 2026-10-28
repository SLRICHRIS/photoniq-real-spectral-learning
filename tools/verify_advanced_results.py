"""Independent integrity checks on published out-of-fold experimental metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def verify(directory: Path) -> dict:
    pred=pd.read_csv(directory/'oof_predictions.csv')
    results=pd.read_csv(directory/'cross_validation_summary.csv')
    splits=pd.read_csv(directory/'full_split_manifest.csv')
    meta=json.loads((directory/'run_metadata.json').read_text())
    assert pred['sample'].nunique()==len(pred)==meta['valid_joined_films']==517
    assert pred['fabrication_date'].nunique()==meta['distinct_fabrication_dates']==20
    assert sorted(pred['fold'].unique())==list(range(1,meta['folds']+1))
    for fold,df in splits.groupby('fold'):
        assert df['sample'].nunique()==len(pred)
        by={k:set(g.fabrication_date) for k,g in df.groupby('split')}
        assert set(by)=={'train','validation','test'}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert by[a].isdisjoint(by[b]),f'Leakage detected: {fold}:{a}:{b}'
        actual_test=set(df[df.split=='test']['sample'])
        oof_test=set(pred[pred.fold==fold]['sample'])
        assert actual_test==oof_test
    error=[]
    for row in results.itertuples(index=False):
        actual=pred.actual_thickness_nm.to_numpy()
        forecast=pred[row.model].to_numpy()
        scores={'MAE_nm':float(mean_absolute_error(actual,forecast)),
                'RMSE_nm':float(np.sqrt(mean_squared_error(actual,forecast))),
                'R2':float(r2_score(actual,forecast))}
        for key,value in scores.items():
            if not np.isclose(value,getattr(row,key),rtol=1e-7,atol=1e-7):
                error.append(f'{row.model}: {key} differs')
    if error: raise AssertionError('; '.join(error))
    assert len(results)==14
    summary={'verified':True,'test_samples':len(pred),'fabrication_date_groups':20,
             'number_of_models':len(results),'folds':meta['folds'],
             'method':'all 517 specimens have exactly one held-out-date prediction; no date overlaps in train/val/test; all scores recomputed'}
    (directory/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Verified:',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
