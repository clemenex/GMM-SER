
#!/usr/bin/env python3
"""
eval_gmm_ser_logging.py
Drop-in evaluator for your GMM-SER pipeline with experiment auto-logging.

What it does:
- Loads scaler, GMM, and meta artifacts
- Computes unsupervised diagnostics (BIC, AIC, silhouette, posterior margin + entropy)
- Sweeps decision threshold(s) and saves coverage table
- If labels exist, computes macro-F1/precision/recall + confusion matrix
- Writes everything to experiments/ser/<run_id>/ as JSON/CSV (and a short README.md)

Usage example:
python eval_gmm_ser_logging.py \
  --data data/daic_prosodic_summary_60s.csv \
  --scaler models/ser60_scaler.joblib \
  --gmm models/ser60_gmm2.joblib \
  --meta models/ser60_meta.joblib \
  --outdir experiments/ser \
  --t-start 0.50 --t-end 0.95 --t-step 0.05 \
  --labels-column target --pos-label high_expr --neg-label low_expr
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import platform

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

def safe_version(pkg_name):
    try:
        mod = __import__(pkg_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"

def make_run_id(prefix="SER-GMM"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}-{ts}"

def posterior_stats(post):
    # margin defined for 2-component model; generalize by top2 gap
    sorted_p = -np.sort(-post, axis=1)  # descending
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    entropy = -(post * np.log(post + 1e-12)).sum(axis=1)
    return margin, entropy

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV with features (and optional labels).")
    p.add_argument("--scaler", required=True, help="Path to scaler.joblib")
    p.add_argument("--gmm", required=True, help="Path to gmm.joblib")
    p.add_argument("--meta", required=True, help="Path to meta.joblib (must include features and component mapping)")
    p.add_argument("--outdir", default="experiments/ser", help="Base output directory")
    p.add_argument("--run-id", default=None, help="Override run id")
    p.add_argument("--labels-column", default=None, help="Optional column containing labels")
    p.add_argument("--pos-label", default="high_expr", help="Positive class value in labels column")
    p.add_argument("--neg-label", default="low_expr", help="Negative class value in labels column")
    p.add_argument("--t-start", type=float, default=0.50)
    p.add_argument("--t-end", type=float, default=0.95)
    p.add_argument("--t-step", type=float, default=0.05)
    args = p.parse_args()

    # Load data & artifacts
    df = pd.read_csv(args.data)
    scaler = load(args.scaler)
    gmm: GaussianMixture = load(args.gmm)
    meta = load(args.meta)

    features = meta.get("features", ["pitch_sd", "energy_sd"])
    low_comp = meta.get("low_expr_comp", 0)
    high_comp = meta.get("high_expr_comp", 1)

    X = scaler.transform(df[features].to_numpy())
    post = gmm.predict_proba(X)
    hard = post.argmax(axis=1)

    # Unsupervised diagnostics
    try:
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
    except Exception:
        bic, aic = None, None
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

    # Threshold sweep
    ts = np.arange(args.t_start, args.t_end + 1e-9, args.t_step)
    coverage_rows = []
    for t in ts:
        # confident if either class posterior >= t
        conf_mask = (post[:, low_comp] >= t) | (post[:, high_comp] >= t)
        coverage = float(conf_mask.mean())
        coverage_rows.append({"t": round(float(t), 4), "coverage": round(coverage, 6)})
    coverage_df = pd.DataFrame(coverage_rows)

    # Supervised (optional)
    supervised = None
    cm_df = None
    sup_by_t = None
    if args.labels_column and args.labels_column in df.columns:
        y_raw = df[args.labels_column].tolist()
        y = np.array([1 if v == args.pos_label else 0 if v == args.neg_label else np.nan for v in y_raw])
        mask_valid = ~np.isnan(y)
        if mask_valid.sum() > 0:
            y = y[mask_valid].astype(int)
            post = post[mask_valid]
            p_high = post[:, high_comp]
            y_pred_050 = (p_high >= 0.50).astype(int)

            supervised = {
                "macroF1_at_0.50": float(f1_score(y, y_pred_050, average="macro")),
                "precisionMacro_at_0.50": float(precision_score(y, y_pred_050, average="macro", zero_division=0)),
                "recallMacro_at_0.50": float(recall_score(y, y_pred_050, average="macro")),
                "accuracy_at_0.50": float((y == y_pred_050).mean())
            }
            try:
                supervised["roc_auc"] = float(roc_auc_score(y, p_high))
            except Exception:
                pass

            # confusion matrix at 0.50
            cm = confusion_matrix(y, y_pred_050, labels=[0,1])
            cm_df = pd.DataFrame(cm, columns=["pred_low","pred_high"], index=["true_low","true_high"])

            # high-confidence subset metrics per t
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

    # Build run directory
    run_id = args.run_id or make_run_id()
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Environment & model config
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
        "component_means": comp_means
    }

    # Write files
    summary = {
        "experimentId": run_id,
        "runDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "data": {
            "path": args.data,
            "features": features,
            "n_rows": int(len(df))
        },
        "artifacts": {
            "scaler": args.scaler,
            "gmm": args.gmm,
            "meta": args.meta
        },
        "environment": env,
        "modelConfig": model_conf,
        "metrics": {
            "unsupervised": {
                "bic": bic,
                "aic": aic,
                "silhouette": sil,
                "posteriorMargin": {
                    "mean": float(np.mean(margin)),
                    "p25": float(np.percentile(margin, 25)),
                    "p75": float(np.percentile(margin, 75))
                },
                "posteriorEntropy": {
                    "mean": float(np.mean(entropy))
                }
            },
            "supervised_at_0.50": supervised
        }
    }
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "coverage_vs_threshold.csv", coverage_df)
    if cm_df is not None:
        write_csv(run_dir / "confusion_matrix_0.50.csv", cm_df)
    if sup_by_t is not None:
        write_csv(run_dir / "supervised_by_threshold.csv", sup_by_t)

    # README
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

See `coverage_vs_threshold.csv` for coverage across thresholds.
{ "Supervised metrics at 0.50 are in summary.json; confusion matrix saved to confusion_matrix_0.50.csv and per-threshold metrics in supervised_by_threshold.csv." if supervised is not None else ""}
"""
    with open(run_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    # Console echo
    print(f"[OK] Wrote logs to {run_dir}")
    print(" - summary.json")
    print(" - coverage_vs_threshold.csv")
    if supervised is not None:
        print(" - confusion_matrix_0.50.csv")
        print(" - supervised_by_threshold.csv")
    print(" - README.md")

if __name__ == "__main__":
    main()
