import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ==== Load data & artifacts (uses your existing files) ====
df = pd.read_csv("data/daic_prosodic_summary_60s.csv")   # same as training
scaler = load("models/ser60_scaler.joblib")
gmm = load("models/ser60_gmm2.joblib")
meta = load("models/ser60_meta.joblib")
FEATURES = meta["features"]        # ["pitch_sd", "energy_sd"]
low_expr_comp = meta["low_expr_comp"]
high_expr_comp = meta["high_expr_comp"]

X = scaler.transform(df[FEATURES].to_numpy())
post = gmm.predict_proba(X)
pred_comp = post.argmax(axis=1)
margin = np.abs(post[:, high_expr_comp] - post[:, low_expr_comp])
entropy = -(post * np.log(post + 1e-12)).sum(axis=1)

print("=== Unsupervised diagnostics ===")
print(f"BIC: {gmm.bic(X):.2f}  AIC: {gmm.aic(X):.2f}")
print(f"Silhouette: {silhouette_score(X, pred_comp):.3f}")
print(f"Posterior margin: mean={margin.mean():.3f}, p25={np.percentile(margin,25):.3f}, p75={np.percentile(margin,75):.3f}")
print(f"Posterior entropy: mean={entropy.mean():.3f}")

# ==== Threshold sweep for confidence policy ====
def label_with_threshold(p_low, p_high, t):
    if p_low >= t:   return "flat_prosody"
    if p_high >= t:  return "expressive_prosody"
    return "prosody_ambiguous"

print("\n=== Threshold sweep (coverage vs. rates) ===")
for t in np.linspace(0.50, 0.95, 10):
    labels = [label_with_threshold(p[low_expr_comp], p[high_expr_comp], t) for p in post]
    coverage = 1.0 - (np.array(labels) == "prosody_ambiguous").mean()
    print(f"t={t:.2f}  coverage={coverage:.2%}")

# ==== Supervised metrics (optional, only if you have labels) ====
if "target" in df.columns:
    # target must be 'low_expr' or 'high_expr' per window
    y = df["target"].map({"low_expr":0, "high_expr":1}).to_numpy()
    p_high = post[:, high_expr_comp]
    y_pred = (p_high >= 0.50).astype(int)  # default decision boundary; threshold will be tuned below

    print("\n=== Supervised metrics @0.50 ===")
    print(classification_report(y, y_pred, target_names=["low_expr","high_expr"], digits=3))
    try:
        print("ROC-AUC:", roc_auc_score(y, p_high))
    except ValueError:
        pass

    # threshold tuning for high-confidence-only subset
    print("\n=== High-confidence F1 vs threshold ===")
    for t in np.linspace(0.50, 0.95, 10):
        mask = (p_high >= t) | (1 - p_high >= t)
        if mask.sum() == 0:
            continue
        y_sub = y[mask]
        y_pred_sub = (p_high[mask] >= 0.50).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(y_sub, y_pred_sub, average="macro")
        prec = precision_score(y_sub, y_pred_sub, average="macro", zero_division=0)
        rec = recall_score(y_sub, y_pred_sub, average="macro")
        coverage = mask.mean()
        print(f"t={t:.2f}  cov={coverage:.2%}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")
