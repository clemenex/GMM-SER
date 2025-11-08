import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from joblib import dump

df = pd.read_csv("data/daic_prosodic_summary_60s.csv")

FEATURES = ["pitch_sd", "energy_sd"]

X = df.dropna(subset=FEATURES)[FEATURES].to_numpy()

scaler = StandardScaler()
Xz = scaler.fit_transform(X)

gmm = GaussianMixture(
    n_components=4,
    covariance_type="diag",
    random_state=42,
    n_init=10,
    reg_covar=1e-6
)
gmm.fit(Xz)

# ---- "low expressiveness" vs "high expressiveness" ----
means_z = gmm.means_
means_orig = scaler.inverse_transform(means_z)

K = gmm.n_components
pitch_means = means_orig[:, 0]
order = np.argsort(pitch_means)
mid = K // 2
low_expr_comps = order[:mid].tolist()
high_expr_comps = order[mid:].tolist()

pitch_means = means_orig[:, 0]
low_expr_comp = int(np.argmin(pitch_means))
high_expr_comp = 1 - low_expr_comp

bic = gmm.bic(Xz)
aic = gmm.aic(Xz)
sil = silhouette_score(Xz, gmm.predict(Xz))
print({
    "BIC": bic,
    "AIC": aic,
    "silhouette": sil,
    "means_orig": means_orig,
    "weights": gmm.weights_.tolist()
})

scaler_path = "models/ser60_scaler.joblib"
gmm_path    = f"models/ser60_gmm{gmm.n_components}_{gmm.covariance_type}.joblib"
meta_path   = "models/ser60_meta.joblib"

dump(scaler, "models/ser60_scaler.joblib")
dump(gmm, "models/ser60_gmm2.joblib")

meta = {
    "features": FEATURES,
    "low_expr_comp": low_expr_comp,
    "high_expr_comp": high_expr_comp,
    "low_expr_comps": [int(i) for i in low_expr_comps],
    "high_expr_comps": [int(i) for i in high_expr_comps],
    "component_means_z": means_z.tolist(),
    "component_means_orig": means_orig.tolist(),
    "weights": gmm.weights_.tolist(),
    "thresholds": {"symmetric": 0.90, "low_expr": 0.90, "high_expr": 0.95},
    "model_config": {                      # <-- make this dynamic, not hard-coded
        "n_components": gmm.n_components,
        "covariance_type": gmm.covariance_type,
        "random_state": 42,
        "n_init": 10,
        "reg_covar": 1e-6
    },
    "artifacts": {                         # helpful provenance
        "scaler": scaler_path,
        "gmm": gmm_path,
        "meta": meta_path
    },
    "naming_rule": "group by pitch_sd mean in original space (lower half = low_expr)",
    "created_at": datetime.now().isoformat(timespec="seconds")
}
dump(meta, meta_path)

print(f"Saved: {scaler_path}, {gmm_path}, {meta_path}")
