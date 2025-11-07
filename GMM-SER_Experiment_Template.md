---
experimentId: <exp-id>            # e.g., SER-GMM-EXP01
title: "<short, informative title>"
runDate: "<YYYY-MM-DD HH:MM (UTC+8)>"
codeRefs:
  trainScript: "ser_gmm.py"
  inferScript: "ser_gmm_infer.py"
  evalScript: "eval_gmm_ser.py"
  configFile: "<path/to/config.yaml if any>"
artifacts:
  scaler: "<models/...joblib>"
  gmm: "<models/...joblib>"
  meta: "<models/...joblib>"
data:
  dataset: "<name / path>"
  windowSize: "60s"
  features: ["pitch_sd", "energy_sd"]
  labelsAvailable: <true|false>
  labelDefinition: "<if supervised: low_expr vs high_expr per window>"
  splits: "<train/val/test details or CV folds>"
environment:
  python: "<version>"
  numpy: "<version>"
  sklearn: "<version>"
  cpu_or_gpu: "<machine>"
  seed: "<random_state>"
modelConfig:
  n_components: <int>
  covariance_type: "<diag|full|tied|spherical>"
  n_init: <int>
  max_iter: <int>
  tol: <float>
  reg_covar: <float>
  init_params: "<kmeans|random>"
thresholdPolicy:
  type: "posterior"
  symmetricThreshold: <float>   # e.g., 0.90
  asymmetricThresholds:
    low_expr: <null|float>
    high_expr: <null|float>
objective:
  purpose: "<what this experiment tests (e.g., threshold sweep, K sweep, covariance type)>"
  hypothesis: "<expected outcome>"
procedure:
  steps:
    - "<what was done step-by-step>"
    - "<e.g., sweep t in [0.50, 0.95] by 0.05>"
metrics:
  unsupervised:
    bic: <float>
    aic: <float>
    silhouette: <float>
    posteriorMargin:
      mean: <float>
      p25: <float>
      p75: <float>
    posteriorEntropy:
      mean: <float>
  supervised:  # remove if labels are not available
    accuracy: <float>
    macroF1: <float>
    precisionMacro: <float>
    recallMacro: <float>
    confusionMatrixPath: "<fig or csv>"
  coverageVsThreshold:
    - { t: 0.50, coverage: <float> }
    - { t: 0.55, coverage: <float> }
    # ...
resultsSummary:
  keyFindings:
    - "<one-line finding>"
    - "<another>"
  decision:
    thresholdDefault: <float>
    thresholdConservative: <float>
  anomaliesOrNotes:
    - "<e.g., margin mean < p25; recheck>"
strengths:
  - "<what worked well>"
limitations:
  - "<what didn't; data caveats>"
nextActions:
  - "<e.g., stability test across seeds>"
  - "<e.g., K & covariance sweep by BIC>"
reproducibility:
  runCommand: "python eval_gmm_ser.py"
  gitCommit: "<hash if applicable>"
  randomState: "<int>"
---

# Narrative

## Purpose
<Explain motivation and expected impact.>

## Method
<Describe how the experiment was executed in prose (data, model settings, threshold policy).>

## Results
<Discuss unsupervised diagnostics and (if any) supervised metrics. Include brief interpretation.>

## Decision
<State what will be carried forward (e.g., threshold=0.90).>

## Appendix
- Logs: <path>
- Figures: <path>
