# Speech Emotion Recognition via Gaussian Mixture Model (GMM)

Unsupervised Speech Emotion Recognition (SER) that clusters short speech windows using a 2-component Gaussian Mixture Model over prosodic variability features (`pitch_sd`, `energy_sd`). Posterior probabilities are then mapped to clinically legible descriptors (e.g., “flat prosody” vs “expressive prosody”) and can be exported as compact text for downstream RAG/clinical-note workflows.  

---

## Highlights

* **Tiny feature set**: just two robust summary stats per window (`pitch_sd`, `energy_sd`).
* **Unsupervised by default**: fits a 2-component GMM and names components via interpretable rules. 
* **Human-friendly outputs**: maps posteriors to DSM-style wording with configurable confidence thresholds; emits compact RAG context strings. 
* **Built-in evaluation**: unsupervised diagnostics (BIC/AIC/silhouette/posterior margin & entropy) and optional supervised metrics (macro-F1/precision/recall/ROC-AUC) if labels are available.  

---

## Project structure

```
.
├── data/
│   └── daic_prosodic_summary_60s.csv     # Example CSV with pitch_sd, energy_sd (+ optional labels)
├── models/
│   ├── ser60_scaler.joblib               # StandardScaler
│   ├── ser60_gmm2.joblib                 # 2-component GMM
│   └── ser60_meta.joblib                 # {features, low/high component ids}
├── outputs/
│   └── ser60_inference_with_descriptors.csv
├── experiments/
│   └── ser/<run_id>/ ... (metrics & logs)
├── ser_gmm_train.py                      # Train + save artifacts
├── ser_gmm_infer.py                      # Inference + descriptor mapping + RAG context
├── eval_gmm_ser.py                       # Quick diagnostics + optional supervised metrics
└── eval_gmm_ser_logging.py               # Full evaluator with auto-logging to experiments/
```

---

## Setup

> Requires Python ≥3.9 (tested on 3.11) and a platform where `pip` can install scientific Python.

```bash
# 1) Create and activate a virtual environment (Windows PowerShell example)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data format

The repository expects a window-level CSV like:

| column      | type   | notes                                                      |
| ----------- | ------ | ---------------------------------------------------------- |
| `pitch_sd`  | float  | standard deviation of F0 in the window                     |
| `energy_sd` | float  | standard deviation of energy in the window                 |
| `target`    | string | *(optional)* `low_expr` or `high_expr` for supervised eval |

If you don’t have labels, all unsupervised diagnostics still work.

---

## Quickstart

### 1) Train the GMM (and save artifacts)

```bash
python ser_gmm.py
```

* Trains a **2-component** GMM on `pitch_sd` & `energy_sd`, logs quick diagnostics (BIC, AIC, silhouette), and writes:

  * `models/ser60_scaler.joblib`, `models/ser60_gmm2.joblib`, `models/ser60_meta.joblib` (includes which component is low vs high expressiveness). 

### 2) Run inference + descriptors + RAG context

```bash
python ser_gmm_infer.py
```

* Produces `outputs/ser60_inference_with_descriptors.csv` with, per window:

  * `gmm_posteriors.low_expr`, `gmm_posteriors.high_expr`
  * `ser_label` ∈ {`flat_prosody`, `expressive_prosody`, `prosody_ambiguous`}
  * `confidence` ∈ {`high`, `low`}
  * human-readable `descriptor`
  * a ready-to-use `rag_context` string. 

**Confidence policy:** the threshold defaults to `THRESH_CONFIDENT = 0.90`. Tune in `ser_gmm_infer.py`. 

### 3) Evaluate

**Lightweight (console):**

```bash
python eval_gmm_ser.py
```

* Prints unsupervised diagnostics; if `target` exists, prints supervised metrics and optional ROC-AUC. 

**Full run with auto-logging:**

```bash
python eval_gmm_ser_logging.py \
  --data data/daic_prosodic_summary_60s.csv \
  --scaler models/ser60_scaler.joblib \
  --gmm models/ser60_gmm2.joblib \
  --meta models/ser60_meta.joblib \
  --outdir experiments/ser \
  --t-start 0.50 --t-end 0.95 --t-step 0.05 \
  --labels-column target --pos-label high_expr --neg-label low_expr
```

* Writes a timestamped folder with `summary.json`, `coverage_vs_threshold.csv`, optional `confusion_matrix_0.50.csv`, `supervised_by_threshold.csv`, and a mini `README.md`. 

---

## How the model decides

1. **Fit:** Standardize features, fit 2-comp GMM (`full` covariance, multiple inits). 
2. **Name components:** The component with **lower** (de-standardized) `pitch_sd` is tagged **low expressiveness**, the other **high**. 
3. **Posterior → label:**

   * If `P(low_expr) ≥ t`: `flat_prosody` (high confidence)
   * Else if `P(high_expr) ≥ t`: `expressive_prosody` (high confidence)
   * Else: `prosody_ambiguous` (low confidence)
     with default `t = 0.90`. Optional wording nudges use energy percentiles. 

---

## Metrics & what they mean

* **Unsupervised** (always available):

  * **BIC/AIC**: model penalized fit. Lower is better.
  * **Silhouette** (on hard labels): cluster separation/cohesion (−1…1).
  * **Posterior margin**: `top1 − top2` probability; higher ⇒ more confident.
  * **Posterior entropy**: distribution uncertainty; lower ⇒ more confident.
    Computed in `eval_gmm_ser.py` & `eval_gmm_ser_logging.py`.  
* **Supervised** (if `target` present):

  * **Macro-F1/Precision/Recall** at a fixed decision boundary (0.50).
  * **ROC-AUC** using `P(high_expr)` scores.
  * **High-confidence subset curves** across thresholds: coverage vs macro-F1/Precision/Recall (filtering to windows where either class posterior ≥ *t*).  

---

## Example output (one row)

```json
{
  "pitch_sd": 22.1,
  "energy_sd": 0.15,
  "gmm_posteriors": {"low_expr": 0.94, "high_expr": 0.06},
  "ser_label": "flat_prosody",
  "confidence": "high",
  "descriptor": "Speech shows monotone/flat prosody with diminished emotional expressiveness. Low energy variation is also observed.",
  "rag_context": "Observed prosody: Speech shows monotone/flat prosody ... (posterior low_expr=0.94, high_expr=0.06; pitch_sd=22.1, energy_sd=0.15). Retrieve DSM-5-TR sections ..."
}
```

Descriptor & RAG context generation logic: 

---

## Configuration

Key places to tweak:

* **Model size & covariance** (train): `n_components`, `covariance_type`, `n_init`, `reg_covar`. 
* **Confidence threshold** (infer): `THRESH_CONFIDENT` (`0.90` default). 
* **Threshold sweeps** (eval): `--t-start`, `--t-end`, `--t-step` for coverage and high-confidence curves. 

---

## Reproducible experiment logs

Run `eval_gmm_ser_logging.py` (see **Evaluate** section). It stamps a run id (e.g., `SER-GMM-YYYYMMDD_HHMMSS`) and saves:

* `summary.json`: data & environment snapshot, model config, diagnostics.
* `coverage_vs_threshold.csv`: how much of your data is “high-confidence” at each *t*.
* If labels exist: confusion matrix at 0.50 and per-threshold macro metrics.
* A mini `README.md` summarizing the run. 

---

## Integrations

* **RAG / clinical note drafting**: use `rag_context` strings per window to condition retrieval toward DSM sections where prosody is diagnostically relevant. 
* **Downstream rules**: aggregate window-level outputs to session summaries (e.g., % of windows with `flat_prosody` at *t* ≥ 0.9).

---

## Troubleshooting

* `ModuleNotFoundError: No module named 'joblib'`
  Activate your venv (`.venv\Scripts\Activate.ps1`) then `pip install joblib`. On Windows, use `py -m pip ...` to avoid mixing global/venv installations.
* `FileNotFoundError` for `models/*.joblib`
  Run `ser_gmm_train.py` first or adjust paths.
* `KeyError: 'pitch_sd'`
  Ensure your CSV has `pitch_sd` and `energy_sd` columns named exactly like that.

---

## Ethics & usage

This repository is **for research/education**. The descriptors are **observational** and **not** diagnostic. Always pair SER with broader clinical context and professional judgment.

---

## License

Code: Apache-2.0 (see `LICENSE`)

---

## Acknowledgements

* Built with NumPy, pandas, scikit-learn, and joblib.
* Example data layout inspired by standard SER/DAIC-style window summaries.

---

## Appendix: Commands cheat sheet

```bash
# Train
python ser_gmm.py

# Infer
python ser_gmm_infer.py

# Evaluate (quick)
python eval_gmm_ser.py

# Evaluate (logged)
python eval_gmm_ser_logging.py --data data/daic_prosodic_summary_60s.csv \
  --scaler models/ser60_scaler.joblib \
  --gmm models/ser60_gmm2.joblib \
  --meta models/ser60_meta.joblib \
  --outdir experiments/ser \
  --t-start 0.50 --t-end 0.95 --t-step 0.05
```

---

### Repo internals referenced

* Training script & artifact saving: 
* Inference, descriptor mapping, and RAG context: 
* Quick evaluation script: 
* Full evaluator with experiment logging: 

---

# Peter Angelo C. Dantes
- Program: BS Computer Science
- Specialization: Data Science
