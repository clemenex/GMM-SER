# ser_gmm_infer.py
import pandas as pd
import numpy as np
from joblib import load

# ---- Load artifacts ----
scaler = load("models/ser60_scaler.joblib")
gmm = load("models/ser60_gmm2.joblib")
meta = load("models/ser60_meta.joblib")
FEATURES = meta["features"]
low_expr_comp = meta["low_expr_comp"]
high_expr_comp = meta["high_expr_comp"]

# ---- Tunable thresholds ----
THRESH_CONFIDENT = 0.90
LOW_PITCH_SD_ABS = None
HIGH_PITCH_SD_ABS = None

def map_to_descriptor(row, probs):
    """
    Map posterior probabilities + raw stats to DSM-style clinical wording.
    row: pandas Series with pitch_sd, energy_sd (if available)
    probs: array([P(comp0), P(comp1)]) aligned with gmm component order
    """
    p_low = probs[low_expr_comp]
    p_high = probs[high_expr_comp]
    pitch_sd = row.get("pitch_sd", np.nan)
    energy_sd = row.get("energy_sd", np.nan)

    # Default
    label = "prosody_ambiguous"
    descriptor = "Prosodic features are inconclusive; consider additional signs/symptoms."
    confidence = "low"

    # Confidence logic
    if p_low >= THRESH_CONFIDENT:
        label = "flat_prosody"
        confidence = "high"
        descriptor = "Speech shows monotone/flat prosody with diminished emotional expressiveness."
    elif p_high >= THRESH_CONFIDENT:
        label = "expressive_prosody"
        confidence = "high"
        descriptor = "Speech shows varying/expressive intonation."

    # Optional face-validity nudges from raw stats
    if label == "flat_prosody":
        # If energy_sd is available and low, strengthen wording
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
    # Posterior probs over components
    post = gmm.predict_proba(Xz)

    # Make a small reference for percentiles (only for optional nudges)
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
