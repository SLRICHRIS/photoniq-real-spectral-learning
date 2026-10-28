"""Independent REAL experimental benchmark: Zenodo 10.5281/zenodo.19237616.
No thickness-derived features, no ODint columns, no synthetic training.
Group by fabrication date to reduce same-batch leakage.
"""
import hashlib, io, json, zipfile, random, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn

SEED=42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
source=Path(sys.argv[1]); output=Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(source) as z:
    data=pd.read_csv(io.BytesIO(z.read("Dataset/blade_perovskite_database.csv")))
    spectral=pd.read_csv(io.BytesIO(z.read("Dataset/curvedata_opticaldensity.csv")))
wl=pd.to_numeric(spectral["Wavelength_nm"],errors="coerce").to_numpy()
columns=[c for c in spectral if c!="Wavelength_nm"]
target=pd.to_numeric(data["Perovskite_thickness_nm"],errors="coerce")
ids=data["Sample"].astype(str)
dates=data["Date"].astype(str)
features=spectral[columns].apply(pd.to_numeric,errors="coerce").T
features.index=columns
rows=[]
for idx, sample in enumerate(ids):
    if sample not in features.index or not np.isfinite(target.iloc[idx]) or target.iloc[idx]<=0:
        continue
    x=features.loc[sample].to_numpy(dtype=float)
    valid=np.isfinite(wl) & np.isfinite(x)
    if valid.sum()<int(len(wl)*0.8):
        continue
    if not valid.all():
        x=np.interp(wl,wl[valid],x[valid])
    rows.append((sample, dates.iloc[idx],float(target.iloc[idx]),x))
assert len(rows)>100,f"Insufficient experimental records: {len(rows)}"
sample=np.array([r[0] for r in rows]); groups=np.array([r[1] for r in rows])
y=np.array([r[2] for r in rows],dtype=np.float64)
X=np.array([r[3] for r in rows],dtype=np.float32)
assert len(set(sample))==len(sample), "Repeated specimen IDs"
assert np.isfinite(X).all() and np.isfinite(y).all()
ix=np.arange(len(y))
gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=SEED)
dev,test=next(gss.split(X,y,groups))
sub=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=SEED+1)
dtrain,dval=next(sub.split(X[dev],y[dev],groups[dev]))
train,val=dev[dtrain],dev[dval]
assert not(set(groups[train]) & set(groups[val]))
assert not(set(groups[train]) & set(groups[test]))
assert not(set(groups[val]) & set(groups[test]))
pd.DataFrame({"sample":sample,"fabrication_date":groups,"thickness_nm":y,
              "split":np.where(np.isin(ix,train),"train",np.where(np.isin(ix,val),"val","test"))}).to_csv(output/"split_manifest.csv",index=False)

baseline=DummyRegressor(strategy="mean").fit(X[train],y[train])
ridge=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(X[train],y[train])
preds={"Train mean baseline":baseline.predict(X[test]),"Ridge full spectrum":ridge.predict(X[test])}

scaler=StandardScaler().fit(X[train])
Xt=scaler.transform(X).astype(np.float32)
mu=float(y[train].mean()); sd=float(y[train].std()); assert sd>0
Y=((y-mu)/sd).astype(np.float32)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(1,16,7,padding=3),nn.GELU(),nn.AvgPool1d(2),
             nn.Conv1d(16,32,5,padding=2),nn.GELU(),nn.AvgPool1d(2),
             nn.AdaptiveAvgPool1d(8),nn.Flatten(),nn.Linear(32*8,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x[:,None,:]).flatten()
model=CNN(); opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
tr_x=torch.from_numpy(Xt[train]); tr_y=torch.from_numpy(Y[train])
va_x=torch.from_numpy(Xt[val]); va_y=torch.from_numpy(Y[val])
history=[]; best=float("inf"); patience=0; checkpoint=None
torch.set_num_threads(2)
for epoch in range(1,151):
    model.train()
    perm=torch.randperm(len(tr_y))
    for batch in perm.split(32):
        opt.zero_grad()
        loss=(model(tr_x[batch])-tr_y[batch]).square().mean()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        val_loss=float((model(va_x)-va_y).square().mean())
    history.append({"epoch":epoch,"validation_mse_scaled":val_loss})
    if val_loss<best-1e-5:
        best=val_loss;patience=0;checkpoint={k:v.detach().clone() for k,v in model.state_dict().items()}
    else:
        patience+=1
        if patience>=20:break
model.load_state_dict(checkpoint);model.eval()
with torch.no_grad():preds["CNN full spectrum"]=model(torch.from_numpy(Xt[test])).numpy()*sd+mu
pd.DataFrame(history).to_csv(output/"learning_curve.csv",index=False)
torch.save({"state_dict":model.state_dict(),"train_feature_mean":scaler.mean_,"train_feature_scale":scaler.scale_,
            "train_target_mean_nm":mu,"train_target_std_nm":sd,"wavelengths_nm":wl},output/"zenodo_cnn.pt")
result=[]
predframe=pd.DataFrame({"sample":sample[test],"fabrication_date":groups[test],"actual_thickness_nm":y[test]})
for name,p in preds.items():
    predframe[name]=p
    result.append({"model":name,"MAE_nm":mean_absolute_error(y[test],p),
                   "RMSE_nm":float(np.sqrt(mean_squared_error(y[test],p))),
                   "R2":r2_score(y[test],p)})
pd.DataFrame(result).to_csv(output/"test_metrics.csv",index=False)
predframe.to_csv(output/"test_predictions.csv",index=False)
metadata={"source_doi":"10.5281/zenodo.19237616","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
          "n_valid_samples":len(rows),"n_wavelengths":len(wl),"wavelength_min_nm":float(wl.min()),
          "wavelength_max_nm":float(wl.max()),"train_count":len(train),"validation_count":len(val),"test_count":len(test),
          "train_fabrication_dates":len(set(groups[train])),"val_fabrication_dates":len(set(groups[val])),
          "test_fabrication_dates":len(set(groups[test])),"seed":SEED,
          "software_version":{"python":platform.python_version(),"torch":torch.__version__},
          "protocol":"Date-grouped holdout. CNN checkpoint chosen on validation. Test evaluated once.",
          "interpretation":"Measured optical density to thickness on this dataset, not Mendeley validation. OD may correlate with thickness."}
(output/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
print("VALID RECORDS",len(rows),"SPLIT",len(train),len(val),len(test))
print(pd.DataFrame(result).to_string(index=False))
