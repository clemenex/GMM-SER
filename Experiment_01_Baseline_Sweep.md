---
experimentId: SER-GMM-EXP01
title: "Baseline threshold sweep on 60s windows (DAIC prosody)"
runDate: "2025-11-06 23:52 (UTC+8)"
codeRefs:
  trainScript: "ser_gmm.py"
  inferScript: "ser_gmm_infer.py"
  evalScript: "eval_gmm_ser.py"
  configFile: null
artifacts:
  scaler: "models/ser60_scaler.joblib"
  gmm: "models/ser60_gmm2.joblib"
  meta: "models/ser60_meta.joblib"
data:
  dataset: "daic_prosodic_summary_60s.csv"
  windowSize: "60s"
  features: ["pitch_sd", "energy_sd"]
  labelsAvailable: false
  labelDefinition: null
  splits: "single set; unsupervised diagnostics only"
environment:
  python: "<version>"
  numpy: "<version>"
  sklearn: "<version>"
  cpu_or_gpu: "CPU"
  seed: "<random_state>"
modelConfig:
  n_components: 2
  covariance_type: "diag"  # update if different
  n_init: 1
  max_iter: "<as in code>"
  tol: "<as in code>"
  reg_covar: "<as in code>"
  init_params: "kmeans"
thresholdPolicy:
  type: "posterior"
  symmetricThreshold: 0.90
  asymmetricThresholds:
    low_expr: null
    high_expr: null
objective:
  purpose: "Establish baseline unsupervised diagnostics and choose a default posterior threshold via coverage trade-off."
  hypothesis: "Given strong cluster separation, high thresholds will retain high coverage (>90%)."
procedure:
  steps:
    - "Fit GMM (K=2) on standardized [pitch_sd, energy_sd]."
    - "Compute BIC, AIC, silhouette, posterior margin and entropy."
    - "Sweep t in [0.50, 0.95] step 0.05 and measure coverage."
metrics:
  unsupervised:
    bic: 12569.07
    aic: 12503.01
    silhouette: 0.596
    posteriorMargin:
      mean: 0.938
      p25: 0.960
      p75: 0.984
    posteriorEntropy:
      mean: 0.105
  coverageVsThreshold:
    - { t: 0.50, coverage: 1.0000 }
    - { t: 0.55, coverage: 0.9940 }
    - { t: 0.60, coverage: 0.9893 }
    - { t: 0.65, coverage: 0.9837 }
    - { t: 0.70, coverage: 0.9790 }
    - { t: 0.75, coverage: 0.9730 }
    - { t: 0.80, coverage: 0.9660 }
    - { t: 0.85, coverage: 0.9526 }
    - { t: 0.90, coverage: 0.9336 }
    - { t: 0.95, coverage: 0.8852 }
resultsSummary:
  keyFindings:
    - "Clusters are well-separated (silhouette≈0.596), posteriors are confident (entropy≈0.105)."
    - "Even at t=0.90, coverage remains ≈93.36%; at t=0.95, ≈88.52%."
  decision:
    thresholdDefault: 0.90
    thresholdConservative: 0.95
  anomaliesOrNotes:
    - "Posterior margin mean (0.938) < p25 (0.960); recompute percentiles to confirm."
strengths:
  - "High posterior confidence with minimal abstention."
  - "Simple 2D feature space still yields clean separation."
limitations:
  - "No supervised labels; cannot compute F1/accuracy yet."
  - "Only K=2 examined; need BIC sweep over covariance types."
nextActions:
  - "Run stability test across seeds (n_init>1)."
  - "Sweep K and covariance types (select via BIC; tie-break with silhouette and margin)."
  - "Consider asymmetric thresholds if precision on 'expressive' must be maximized."
  - "Collect a small labeled subset (N≈200 windows) to estimate macro-F1."
reproducibility:
  runCommand: "python eval_gmm_ser.py"
  gitCommit: "<hash if applicable>"
  randomState: "<int>"
---

# Narrative

## Purpose
Establish a defensible default posterior threshold for GMM-SER without supervised labels and characterize cluster quality.

## Method
We trained a 2-component Gaussian Mixture Model on standardized 60-second window features (pitch_sd, energy_sd). We computed BIC, AIC, silhouette, and posterior diagnostics (margin, entropy). We then swept the posterior decision threshold t ∈ [0.50, 0.95] at 0.05 increments to measure the coverage (non-ambiguous rate).

## Results
Silhouette was 0.596, with low posterior entropy (0.105) and very high margins, indicating confident separation. Coverage stayed above 93% at t=0.90 and above 88% at t=0.95, implying a conservative threshold can be used without excessively abstaining.

## Decision
Adopt t=0.90 as the default threshold and t=0.95 for conservative settings. Schedule a follow-up sweep across K and covariance types using BIC as the primary selector, with silhouette and margin as tie-breakers. Plan for a small labeled subset to compute macro-F1 and confirm threshold placement.
