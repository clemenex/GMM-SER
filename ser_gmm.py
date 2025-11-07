# ser_gmm_train.py
import pandas as pd
import numpy as np
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
    n_components=2,
    covariance_type="full",
    random_state=42,
    n_init=5,
    reg_covar=1e-6
)
gmm.fit(Xz)

# ---- Identify which component is "low expressiveness" vs "high expressiveness" ----
means_z = gmm.means_                        # in standardized space
means_orig = scaler.inverse_transform(means_z)

pitch_means = means_orig[:, 0]
low_expr_comp = int(np.argmin(pitch_means))
high_expr_comp = 1 - low_expr_comp

# ---- Quick diagnostics ----
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

# ---- Persist model artifacts ----
dump(scaler, "models/ser60_scaler.joblib")
dump(gmm, "models/ser60_gmm2.joblib")
dump({"low_expr_comp": low_expr_comp, "high_expr_comp": high_expr_comp, "features": FEATURES},
     "models/ser60_meta.joblib")
print("Saved: models/ser60_scaler.joblib, ser60_gmm2.joblib, ser60_meta.joblib")
