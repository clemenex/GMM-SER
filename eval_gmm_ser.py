
#!/usr/bin/env python3
"""
eval_gmm_ser.py  (integrated with auto-logging)

Features:
- Loads scaler/GMM/meta and dataset
- Computes unsupervised diagnostics (BIC, AIC, silhouette, posterior margin/entropy)
- Threshold sweep -> coverage table
- Optional supervised metrics if labels are available
- Auto-writes experiment logs to experiments/ser/<run_id>/:
    * summary.json
    * coverage_vs_threshold.csv
    * (optional) confusion_matrix_0.50.csv
    * (optional) supervised_by_threshold.csv
    * README.md
- Prints a concise console summary (same stats you saw earlier)

Usage (no labels):
python eval_gmm_ser.py \
  --data data/daic_prosodic_summary_60s.csv \
  --scaler models/ser60_scaler.joblib \
  --gmm models/ser60_gmm2.joblib \
  --meta models/ser60_meta.joblib \
  --outdir experiments/ser \
  --t-start 0.50 --t-end 0.95 --t-step 0.05

Usage (with labels column):
python eval_gmm_ser.py \
  --data data/daic_prosodic_summary_60s.csv \
  --scaler models/ser60_scaler.joblib \
  --gmm models/ser60_gmm2.joblib \
  --meta models/ser60_meta.joblib \
  --outdir experiments/ser \
  --labels-column target --pos-label high_expr --neg-label low_expr
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import platform

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    silhouette_score, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, classification_report
)
from sklearn.mixture import GaussianMixture

# ---------- helpers ----------
def safe_version(pkg):
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"

def make_run_id(prefix="SER-GMM"):
    return f"{prefix}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def posterior_stats(post):
    """Return (margin, entropy). margin is top1 - top2; entropy in nats."""
    sorted_p = -np.sort(-post, axis=1)
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    entropy = -(post * np.log(post + 1e-12)).sum(axis=1)
    return margin, entropy

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ---------- core ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV path with features (and optional labels).")
    ap.add_argument("--scaler", required=True, help="Path to scaler.joblib")
    ap.add_argument("--gmm", required=True, help="Path to gmm.joblib")
    ap.add_argument("--meta", required=True, help="Path to meta.joblib (includes 'features', 'low_expr_comp', 'high_expr_comp')")
    ap.add_argument("--outdir", default="experiments/ser", help="Base directory to write experiment logs.")
    ap.add_argument("--run-id", default=None, help="Optional run id override.")
    ap.add_argument("--labels-column", default=None, help="Optional label column for supervised metrics.")
    ap.add_argument("--pos-label", default="high_expr", help="Positive class value.")
    ap.add_argument("--neg-label", default="low_expr", help="Negative class value.")
    ap.add_argument("--t-start", type=float, default=0.50)
    ap.add_argument("--t-end", type=float, default=0.95)
    ap.add_argument("--t-step", type=float, default=0.05)
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.data)
    scaler = load(args.scaler)
    gmm: GaussianMixture = load(args.gmm)
    meta = load(args.meta)

    features = meta.get("features", ["pitch_sd", "energy_sd"])
    low_list  = meta.get("low_expr_comps")  or [int(meta.get("low_expr_comp", 0))]
    high_list = meta.get("high_expr_comps") or [int(meta.get("high_expr_comp", 1))]
    thr = meta.get("thresholds", {})
    T_LOW  = float(thr.get("low_expr", thr.get("symmetric", 0.90)))
    T_HIGH = float(thr.get("high_expr", thr.get("symmetric", 0.90)))

    X = scaler.transform(df[features].to_numpy())
    post = gmm.predict_proba(X)
    p_low  = post[:, low_list].sum(axis=1)
    p_high = post[:, high_list].sum(axis=1)
    hard = np.argmax(post, axis=1)

    try:
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
    except Exception:
        bic = aic = None
    try:
        sil = float(silhouette_score(X, hard))
    except Exception:
        sil = None

    margin, entropy = posterior_stats(post)
    comp_means = {}
    try:
        for k in range(gmm.n_components):
            comp_means[int(k)] = {features[i]: float(gmm.means_[k, i]) for i in range(len(features))}
    except Exception:
        pass

    deploy_mask = (p_low >= T_LOW) | (p_high >= T_HIGH)
    deploy_coverage = float(deploy_mask.mean())

    # Threshold sweep
    ts = np.arange(args.t_start, args.t_end + 1e-9, args.t_step)
    coverage_rows = []
    for t in ts:
        conf_mask = (p_low >= t) | (p_high >= t)
        coverage = float(conf_mask.mean())
        coverage_rows.append({"t": round(float(t), 4), "coverage": round(coverage, 6)})
    coverage_df = pd.DataFrame(coverage_rows)

    supervised_summary = None
    cm_df = None
    sup_by_t = None
    if args.labels_column and args.labels_column in df.columns:
        y_raw = df[args.labels_column].tolist()
        y = np.array([1 if v == args.pos_label else 0 if v == args.neg_label else np.nan for v in y_raw])
        mask_valid = ~np.isnan(y)
        if mask_valid.any():
            y = y[mask_valid].astype(int)
            post_l = post[mask_valid]
            p_high = post_l[:, high_comp]

            y_pred050 = (p_high >= 0.50).astype(int)
            supervised_summary = {
                "macroF1_at_0.50": float(f1_score(y, y_pred050, average="macro")),
                "precisionMacro_at_0.50": float(precision_score(y, y_pred050, average="macro", zero_division=0)),
                "recallMacro_at_0.50": float(recall_score(y, y_pred050, average="macro")),
                "accuracy_at_0.50": float((y == y_pred050).mean())
            }
            try:
                supervised_summary["roc_auc"] = float(roc_auc_score(y, p_high))
            except Exception:
                pass

            cm = confusion_matrix(y, y_pred050, labels=[0,1])
            cm_df = pd.DataFrame(cm, columns=["pred_low","pred_high"], index=["true_low","true_high"])

            rows = []
            for t in ts:
                conf_mask = (p_high >= t) | (1 - p_high >= t)
                if conf_mask.sum() == 0:
                    continue
                y_sub = y[conf_mask]
                y_pred_sub = (p_high[conf_mask] >= 0.50).astype(int)
                rows.append({
                    "t": round(float(t), 4),
                    "coverage": round(float(conf_mask.mean()), 6),
                    "f1_macro": round(float(f1_score(y_sub, y_pred_sub, average="macro")), 6),
                    "precision_macro": round(float(precision_score(y_sub, y_pred_sub, average="macro", zero_division=0)), 6),
                    "recall_macro": round(float(recall_score(y_sub, y_pred_sub, average="macro")), 6)
                })
            sup_by_t = pd.DataFrame(rows)

    run_id = args.run_id or make_run_id()
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "python": platform.python_version(),
        "numpy": safe_version("numpy"),
        "pandas": safe_version("pandas"),
        "sklearn": safe_version("sklearn"),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    model_conf = {
        
        "n_components": int(getattr(gmm, "n_components", -1)),
        "covariance_type": getattr(gmm, "covariance_type", None),
        "max_iter": int(getattr(gmm, "max_iter", -1)),
        "tol": float(getattr(gmm, "tol", -1.0)),
        "reg_covar": float(getattr(gmm, "reg_covar", -1.0)),
        "init_params": getattr(gmm, "init_params", None),
        "component_means": comp_means,
        "mixing_weights": getattr(gmm, "weights_", np.array([])).tolist(),
    }

    summary = {
        "experimentId": run_id,
        "runDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": {"path": args.data, "features": features, "n_rows": int(len(df))},
        "artifacts": {"scaler": args.scaler, "gmm": args.gmm, "meta": args.meta},
        "environment": env,
        "modelConfig": model_conf,
        "metrics": {
            "unsupervised": {
                "bic": None if bic is None else float(bic),
                "aic": None if aic is None else float(aic),
                "silhouette": None if sil is None else float(sil),
                "posteriorMargin": {
                    "mean": float(np.mean(margin)),
                    "p25": float(np.percentile(margin, 25)),
                    "p75": float(np.percentile(margin, 75)),
                },
                "posteriorEntropy": {"mean": float(np.mean(entropy))},
            },
            "supervised_at_0.50": supervised_summary,
        },
    }

    summary["deploymentPolicy"] = {
        "thresholds": {
            "symmetric": thr.get("symmetric", None),
            "low_expr": T_LOW,
            "high_expr": T_HIGH,
        },
        "coverage": deploy_coverage
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    coverage_df.to_csv(run_dir / "coverage_vs_threshold.csv", index=False)
    if cm_df is not None:
        cm_df.to_csv(run_dir / "confusion_matrix_0.50.csv", index=False)
    if sup_by_t is not None:
        sup_by_t.to_csv(run_dir / "supervised_by_threshold.csv", index=False)

    readme = f"""# {run_id}

**Data:** {args.data}  
**Artifacts:** scaler={args.scaler}, gmm={args.gmm}, meta={args.meta}  
**Features:** {features}

## Unsupervised Diagnostics
- BIC: {bic}
- AIC: {aic}
- Silhouette: {sil}
- Posterior margin (mean / p25 / p75): {np.mean(margin):.3f} / {np.percentile(margin,25):.3f} / {np.percentile(margin,75):.3f}
- Posterior entropy (mean): {np.mean(entropy):.3f}

## Deployment Policy
low={T_LOW:.2f}, high={T_HIGH:.2f} — coverage={deploy_coverage*100:.2f}%

See `coverage_vs_threshold.csv` for coverage across thresholds.
{ "Supervised metrics at 0.50 are in summary.json; confusion matrix saved to confusion_matrix_0.50.csv and per-threshold metrics in supervised_by_threshold.csv." if supervised_summary is not None else ""}
"""
    with open(run_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print("=== Unsupervised diagnostics ===")
    if bic is not None and aic is not None:
        print(f"BIC: {bic:.2f}  AIC: {aic:.2f}")
    if sil is not None:
        print(f"Silhouette: {sil:.3f}")
    print(f"Posterior margin: mean={np.mean(margin):.3f}, p25={np.percentile(margin,25):.3f}, p75={np.percentile(margin,75):.3f}")
    print(f"Posterior entropy: mean={np.mean(entropy):.3f}\n")

    print("=== Threshold sweep (coverage vs. rates) ===")
    for _, row in coverage_df.iterrows():
        print(f"t={row['t']:.2f}  coverage={row['coverage']*100:.2f}%")
    print(f"\n[Deployment policy] low={T_LOW:.2f}, high={T_HIGH:.2f}  coverage={deploy_coverage*100:.2f}%")
    print(f"\n[OK] Wrote logs to {run_dir}")
    print(" - summary.json")
    print(" - coverage_vs_threshold.csv")
    if cm_df is not None:
        print(" - confusion_matrix_0.50.csv")
    if sup_by_t is not None:
        print(" - supervised_by_threshold.csv")
    print(" - README.md")

if __name__ == "__main__":
    main()
