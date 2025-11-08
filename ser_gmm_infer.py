# ser_gmm_infer.py
import pandas as pd
import numpy as np
from joblib import load, dump
import os

# ---- Load artifacts ----
scaler = load("models/ser60_scaler.joblib")
gmm = load("models/ser60_gmm2.joblib")
meta = load("models/ser60_meta.joblib")
FEATURES = meta["features"]
low_list  = [int(i) for i in meta.get("low_expr_comps",  [meta.get("low_expr_comp", 0)])]
high_list = [int(i) for i in meta.get("high_expr_comps", [meta.get("high_expr_comp", 1)])]

# ---- Tunable thresholds ----
THRESH = meta.get("thresholds", {})
THRESH_CONFIDENT = float(os.getenv("SER_THRESH", THRESH.get("symmetric", 0.90)))
T_LOW  = float(os.getenv("SER_TLOW",  THRESH.get("low_expr", THRESH_CONFIDENT)))
T_HIGH = float(os.getenv("SER_THIGH", THRESH.get("high_expr", THRESH_CONFIDENT)))

print(f"[SER] thresholds -> low={T_LOW:.2f}, high={T_HIGH:.2f} (source: meta/env)")
LOW_PITCH_SD_ABS = None
HIGH_PITCH_SD_ABS = None

def map_to_descriptor(row, probs):
    """
    Map posterior probabilities + raw stats to DSM-style clinical wording.
    row: pandas Series with pitch_sd, energy_sd (if available)
    probs: array([P(comp0), P(comp1)]) aligned with gmm component order
    """
    p_low  = float(probs[low_list].sum())
    p_high = float(probs[high_list].sum())
    pitch_sd = row.get("pitch_sd", np.nan)
    energy_sd = row.get("energy_sd", np.nan)

    # Default
    label = "prosody_ambiguous"
    descriptor = "Prosodic features are inconclusive; consider additional signs/symptoms."
    confidence = "low"

    # Confidence logic
    if p_low  >= T_LOW:
        label, confidence, descriptor = "flat_prosody", "high", "Speech shows monotone/flat prosody with diminished emotional expressiveness."
    elif p_high >= T_HIGH:
        label, confidence, descriptor = "expressive_prosody", "high", "Speech shows varying/expressive intonation."
    else:
        label, confidence, descriptor = "prosody_ambiguous", "low", "Prosodic features are inconclusive; consider additional signs/symptoms."

    if label == "flat_prosody":

        if not np.isnan(energy_sd) and energy_sd < np.nanpercentile(df_ref["energy_sd"], 25):
            descriptor += " Low energy variation is also observed."
    elif label == "expressive_prosody":
        if not np.isnan(energy_sd) and energy_sd > np.nanpercentile(df_ref["energy_sd"], 75):
            descriptor += " Elevated energy variation is also observed."

    return {
        "gmm_posteriors": {"low_expr": float(p_low), "high_expr": float(p_high)},
        "ser_label": label,
        "confidence": confidence,
        "pitch_sd": None if np.isnan(pitch_sd) else float(pitch_sd),
        "energy_sd": None if np.isnan(energy_sd) else float(energy_sd),
        "descriptor": descriptor
    }

def infer_ser_descriptors(df_windows):
    """
    df_windows: DataFrame with at least columns in FEATURES
    Returns: DataFrame with SER outputs + DSM-style descriptor text
    """
    X = df_windows[FEATURES].to_numpy()
    Xz = scaler.transform(X)
    post = gmm.predict_proba(Xz)

    global df_ref
    df_ref = df_windows.copy()

    out_rows = []
    for i, row in df_windows.iterrows():
        out = map_to_descriptor(row, post[i, :])
        out_rows.append(out)

    ser = pd.DataFrame(out_rows)
    return pd.concat([df_windows.reset_index(drop=True), ser], axis=1)

# ---- Example usage: per-window SER → descriptors → RAG context
if __name__ == "__main__":
    # Load your new case/session windows (already computed 60s features)
    new_df = pd.read_csv("data/daic_prosodic_summary_60s.csv")  # same columns as training features
    results = infer_ser_descriptors(new_df)

    # Build compact RAG context strings
    def make_rag_context(row):
        # Short “bridge” sentence to prepend to the RAG query:
        return (
            f"Observed prosody: {row['descriptor']} "
            f"(posterior low_expr={row['gmm_posteriors']['low_expr']:.2f}, "
            f"high_expr={row['gmm_posteriors']['high_expr']:.2f}; "
            f"pitch_sd={row['pitch_sd']}, energy_sd={row['energy_sd']}). "
            "Retrieve DSM-5-TR sections where diminished emotional expression, flat affect, "
            "or prosody-related observations are clinically relevant for diagnosis/differential diagnosis."
        )

    results["rag_context"] = results.apply(make_rag_context, axis=1)
    results.to_csv("outputs/ser60_inference_with_descriptors.csv", index=False)
    print("Wrote outputs/ser60_inference_with_descriptors.csv")
