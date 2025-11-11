# train_gmm_sweep.py
import itertools, json
import numpy as np, pandas as pd
from joblib import dump
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
df = pd.read_csv("data/daic_prosodic_summary_60s.csv")
X = df[["pitch_sd","energy_sd"]].dropna().to_numpy()
scaler = StandardScaler().fit(X); Xz = scaler.transform(X)

rows = []
for k, cov in itertools.product([2,3,4], ["full","diag"]):
    gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, n_init=5, reg_covar=1e-6).fit(Xz)
    hard = gmm.predict(Xz)
    bic, aic = gmm.bic(Xz), gmm.aic(Xz)
    sil = silhouette_score(Xz, hard) if k>1 else np.nan
    rows.append({"k":k,"cov":cov,"bic":bic,"aic":aic,"silhouette":sil})
pd.DataFrame(rows).sort_values("bic").to_csv("experiments/ser/sweep_summary.csv", index=False)
print("Wrote experiments/ser/sweep_summary.csv")
