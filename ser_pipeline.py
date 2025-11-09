# ser_pipeline.py
"""
GMM-SER pipeline for RAG (importable module, no CLI).

Main entry-points you’ll call from your web app:
- SERModel.load_from_paths(scaler_path, gmm_path, meta_path)
- SERModel.process_audio_file(audio_path, win_sec=60.0, hop_sec=60.0, sr_target=16000)
- SERModel.process_array(y, sr, win_sec=60.0, hop_sec=60.0)

Returns:
- A pandas DataFrame (one row per window) with features, posteriors, labels, descriptor, and rag_context
- A list[dict] of RAG documents: {"text": rag_context, "metadata": {...}}

Dependencies:
  pip install numpy pandas joblib librosa==0.10.2.post1 soundfile
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import json
import numpy as np
import pandas as pd
import librosa
from joblib import load

from dataclasses import dataclass, field


# ---------- small helpers ----------

def _percentile_safe(a: np.ndarray, q: float) -> float:
    """np.nanpercentile that tolerates all-NaN / empty by returning np.nan."""
    if a is None or len(a) == 0:
        return float("nan")
    a = np.asarray(a, dtype=float)
    if not np.any(np.isfinite(a)):
        return float("nan")
    return float(np.nanpercentile(a, q))


def _to_semitones(f0_hz: np.ndarray) -> np.ndarray:
    """Convert Hz to semitone offsets relative to the median f0 (voiced only)."""
    f0 = np.array(f0_hz, dtype=float)
    med = np.nanmedian(f0[np.isfinite(f0)])
    if not np.isfinite(med) or med <= 0:
        return np.full_like(f0, np.nan, dtype=float)
    return 12.0 * np.log2(f0 / med)

#---------- LEGACY PIPELINE --------------------
def _pitch_energy_legacy(window_y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Replicates the original thesis code:
    - pitch from librosa.piptrack
    - keep bins where magnitude > global median magnitude
    - pitch_sd in Hz (std), energy_sd from RMS over all frames
    """
    # piptrack returns [freq_bins x frames]
    pitches, mags = librosa.piptrack(y=window_y, sr=sr)
    mask = mags > np.median(mags)                 # global median threshold, as in the old code
    pitch_vals = pitches[mask]                    # 1D array of Hz values

    if pitch_vals.size > 0:
        pitch_sd_hz = float(np.std(pitch_vals))
    else:
        pitch_sd_hz = float("nan")

    rms = librosa.feature.rms(y=window_y)[0]      # no silence filtering in the old code
    energy_sd = float(np.std(rms)) if rms.size > 0 else float("nan")
    return pitch_sd_hz, energy_sd


def _features_from_array_legacy(y: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> pd.DataFrame:
    """
    Segment audio into non-overlapping windows (like the old pipeline) and
    compute pitch_sd (Hz) and energy_sd per window with the legacy method.
    """
    # old code used non-overlapping contiguous windows, skipping very short tails
    win_len = int(sr * win_sec)
    starts = list(range(0, len(y), win_len))
    rows = []
    for i, s in enumerate(starts):
        e = min(s + win_len, len(y))
        if (e - s) < 0.5 * win_len:               # same 50% minimum as your old code
            continue
        w = y[s:e]
        pitch_sd, energy_sd = _pitch_energy_legacy(w, sr)
        rows.append({
            "window_id": i,
            "start_s": round(s / sr, 3),
            "end_s": round(e / sr, 3),
            "pitch_sd": pitch_sd,                 # Hz
            "energy_sd": energy_sd               # RMS SD
        })
    return pd.DataFrame(rows)


@dataclass
class SERArtifacts:
    scaler: object
    gmm: object
    meta: dict
    features: List[str]
    low_list: List[int]
    high_list: List[int]
    T_LOW: float
    T_HIGH: float
    feature_schema: str = "legacy_piptrack_hz_sd__rms_sd"
    feature_mode: str = field(default="legacy")


# ---------- the main model ----------

class SERModel:
    """Encapsulates artifacts + feature extraction + inference for GMM-SER."""

    def __init__(self, artifacts: SERArtifacts):
        self.art = artifacts

    # ----- loading -----

    @classmethod
    def load_from_paths(cls, scaler_path: str, gmm_path: str, meta_path: str) -> "SERModel":
        scaler = load(scaler_path)
        gmm = load(gmm_path)
        meta = load(meta_path)

        features = meta.get("features", ["pitch_sd", "energy_sd"])
        raw_low  = meta.get("low_expr_comps")
        raw_high = meta.get("high_expr_comps")

        if not raw_low:
            raw_low = [meta.get("low_expr_comp", 0)]
        if not raw_high:
            raw_high = [meta.get("high_expr_comp", 1)]

        low_list  = [int(i) for i in raw_low]
        high_list = [int(i) for i in raw_high]

        if (len(low_list) + len(high_list)) == 0:
            raise ValueError("No components assigned to low/high groups in meta.")

        thr = meta.get("thresholds", {})
        t_default = float(os.getenv("SER_THRESH", thr.get("symmetric", 0.90)))
        T_LOW  = float(os.getenv("SER_TLOW",  thr.get("low_expr", t_default)))
        T_HIGH = float(os.getenv("SER_THIGH", thr.get("high_expr", t_default)))

        schema = meta.get("feature_schema", "legacy_piptrack_hz_sd__rms_sd")
        # Allow override via env: SER_FEATURE_MODE = legacy|improved
        mode_env = os.getenv("SER_FEATURE_MODE", "").strip().lower()
        feature_mode = (
            mode_env if mode_env in {"legacy","improved"} else
            ("legacy" if schema.startswith("legacy_piptrack") else "improved")
        )

        art = SERArtifacts(
            scaler=scaler, gmm=gmm, meta=meta, features=features,
            low_list=low_list, high_list=high_list, T_LOW=T_LOW, T_HIGH=T_HIGH,
            feature_schema=schema, feature_mode=feature_mode
        )
        model = cls(art)
        model._assert_schema_compat()
        return model
    
    def _assert_schema_compat(self):
        schema = self.art.feature_schema or ""
        if self.art.feature_mode == "legacy" and not schema.startswith("legacy_piptrack"):
            raise ValueError(f"Extractor=legacy but meta.feature_schema={schema}. Retrain or set SER_FEATURE_MODE=improved.")
        if self.art.feature_mode == "improved" and schema.startswith("legacy_piptrack"):
            raise ValueError(f"Extractor=improved but meta.feature_schema={schema}. Retrain or set SER_FEATURE_MODE=legacy.")

    # ----- audio IO & windowing -----

    @staticmethod
    def load_audio_mono(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(path, sr=sr, mono=True, res_type="soxr_vhq")
        # DC + gentle peak normalize
        y = y - np.mean(y)
        peak = np.max(np.abs(y)) + 1e-9
        if peak > 1.0:
            y = y / peak
        return y, sr

    @staticmethod
    def _make_windows(y: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        n = len(y)
        win = int(round(win_sec * sr))
        hop = int(round(hop_sec * sr))
        starts = np.arange(0, max(1, n - win + 1), hop, dtype=int)
        ends = np.minimum(starts + win, n).astype(int)
        return starts, ends

    # ----- low-level feature extraction -----

    @staticmethod
    def _frame_features(y: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = 256,
                        fmin: float = 65.0, fmax: float = 400.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (frame_times, f0_hz, rms). f0 has NaNs for unvoiced."""
        # frame times
        n_frames = max(1, 1 + (len(y) - frame_length) // hop_length)
        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True).flatten()

        # Pitch with pYIN (robust voicing)
        f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=frame_length, hop_length=hop_length, center=True)
        return frame_times.astype(float), f0.astype(float), rms.astype(float)

    @staticmethod
    def _window_summary(frame_times: np.ndarray, f0_hz: np.ndarray, rms: np.ndarray,
                        start_s: float, end_s: float, energy_pct_exclude: float = 20.0) -> Dict[str, Optional[float]]:
        m = (frame_times >= start_s) & (frame_times < end_s)
        if not np.any(m):
            return {"pitch_sd": np.nan, "energy_sd": np.nan}

        f0_w = f0_hz[m]
        rms_w = rms[m]

        # remove the quietest frames inside the window
        thr = _percentile_safe(rms_w, energy_pct_exclude)
        keep = np.ones_like(rms_w, dtype=bool) if not np.isfinite(thr) else (rms_w >= thr)

        # pitch_sd in semitone offsets (speaker-normalized)
        st = _to_semitones(f0_w) if np.any(np.isfinite(f0_w)) else np.full_like(f0_w, np.nan)
        pitch_sd = float(np.nanstd(st[keep])) if np.any(keep) else np.nan

        # energy_sd on non-silent frames
        energy_sd = float(np.nanstd(rms_w[keep])) if np.any(keep) else np.nan

        return {"pitch_sd": pitch_sd, "energy_sd": energy_sd}

    # ----- high-level: features per window -----

    def _features_from_array(self, y: np.ndarray, sr: int, win_sec: float, hop_sec: float,
                             fmin: float, fmax: float) -> pd.DataFrame:
        starts, ends = self._make_windows(y, sr, win_sec, hop_sec)
        ft, f0, rms = self._frame_features(y, sr, fmin=fmin, fmax=fmax)

        rows = []
        for i, (smp_s, smp_e) in enumerate(zip(starts, ends)):
            s, e = smp_s / sr, smp_e / sr
            feats = self._window_summary(ft, f0, rms, s, e, energy_pct_exclude=20.0)
            rows.append({"window_id": i, "start_s": round(s, 3), "end_s": round(e, 3), **feats})
        return pd.DataFrame(rows)

    # ----- inference & mapping -----

    @staticmethod
    def _map_descriptor(row: pd.Series, p_low: float, p_high: float,
                        q25_energy: float, q75_energy: float, T_LOW: float, T_HIGH: float) -> Dict:
        label = "prosody_ambiguous"
        confidence = "low"
        descriptor = "Prosodic features are inconclusive; consider additional signs/symptoms."

        if p_low >= T_LOW:
            label = "flat_prosody"
            confidence = "high"
            descriptor = "Speech shows monotone/flat prosody with diminished emotional expressiveness."
            if pd.notna(row.get("energy_sd")) and pd.notna(q25_energy) and row["energy_sd"] < q25_energy:
                descriptor += " Low energy variation is also observed."
        elif p_high >= T_HIGH:
            label = "expressive_prosody"
            confidence = "high"
            descriptor = "Speech shows varying/expressive intonation."
            if pd.notna(row.get("energy_sd")) and pd.notna(q75_energy) and row["energy_sd"] > q75_energy:
                descriptor += " Elevated energy variation is also observed."

        return {"ser_label": label, "confidence": confidence, "descriptor": descriptor,
                "gmm_posteriors": {"low_expr": float(p_low), "high_expr": float(p_high)}}

    @staticmethod
    def _build_rag_text(row: pd.Series) -> str:
        return (
            f"Observed prosody: {row['descriptor']} "
            f"(posterior low_expr={row['gmm_posteriors']['low_expr']:.2f}, "
            f"high_expr={row['gmm_posteriors']['high_expr']:.2f}; "
            f"pitch_sd={row['pitch_sd']}, energy_sd={row['energy_sd']}). "
            "Retrieve DSM-5-TR sections where diminished emotional expression, flat affect, or "
            "prosody-related observations are clinically relevant for diagnosis or differential diagnosis."
        )

    # ----- public API you’ll call -----

    def process_audio_file(self, audio_path: str, win_sec: float = 60.0, hop_sec: float = 60.0,
                           sr_target: int = 16000, fmin: float = 65.0, fmax: float = 400.0
                           ) -> Tuple[pd.DataFrame, List[Dict]]:
        y, sr = self.load_audio_mono(audio_path, sr=sr_target)
        return self.process_array(y, sr, win_sec=win_sec, hop_sec=hop_sec, fmin=fmin, fmax=fmax)

    def process_array(self, y: np.ndarray, sr: int,
                      win_sec: float = 60.0, hop_sec: float = 60.0,
                      fmin: float = 65.0, fmax: float = 400.0,
                      feature_mode: Optional[str] = None
                      ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Full pipeline on in-memory audio: features → GMM → descriptors → RAG docs.
        Set feature_mode to override ('legacy'/'improved'); defaults to artifacts' mode.
        """
        mode = (feature_mode or self.art.feature_mode).lower()
        if mode not in {"legacy","improved"}:
            raise ValueError(f"Unknown feature_mode={mode}")

        # 1) features per window
        if mode == "legacy":
            # Legacy uses non-overlapping windows; we intentionally ignore hop_sec here.
            df = _features_from_array_legacy(y, sr, win_sec, hop_sec)
            df["feature_schema"] = "legacy_piptrack_hz_sd__rms_sd"
        else:
            df = self._features_from_array(y, sr, win_sec, hop_sec, fmin, fmax)
            df["feature_schema"] = "pyin_semitone_sd__rms_sd_excl20p"  # or yin_* if that's your extractor

        # 2) align features & get posteriors
        X = df[self.art.features].to_numpy()
        Xz = self.art.scaler.transform(X)
        post = self.art.gmm.predict_proba(Xz)

        print("features:", self.art.features)
        print("low_list:", self.art.low_list, "high_list:", self.art.high_list)
        print("n_components:", self.art.gmm.n_components)
        print("post[0][:5]:", post[0][:min(5, post.shape[1])], "sum:", post[0].sum())
        
        p_low  = post[:, self.art.low_list].sum(axis=1)
        p_high = post[:, self.art.high_list].sum(axis=1)

        # 3) dataset-level energy quantiles for text mapping
        q25_energy = _percentile_safe(df["energy_sd"].to_numpy(), 25.0)
        q75_energy = _percentile_safe(df["energy_sd"].to_numpy(), 75.0)

        mapped = []
        for i, row in df.iterrows():
            info = self._map_descriptor(
                row=row, p_low=p_low[i], p_high=p_high[i],
                q25_energy=q25_energy, q75_energy=q75_energy,
                T_LOW=self.art.T_LOW, T_HIGH=self.art.T_HIGH
            )
            mapped.append(info)

        df["gmm_posteriors"] = [m["gmm_posteriors"] for m in mapped]
        df["ser_label"]      = [m["ser_label"]      for m in mapped]
        df["confidence"]     = [m["confidence"]     for m in mapped]
        df["descriptor"]     = [m["descriptor"]     for m in mapped]
        df["rag_context"]    = df.apply(self._build_rag_text, axis=1)

        # 4) RAG docs
        docs = [{
            "text": r["rag_context"],
            "metadata": {
                "window_id": int(r["window_id"]),
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "ser_label": r["ser_label"],
                "confidence": r["confidence"],
                "pitch_sd": None if pd.isna(r["pitch_sd"]) else float(r["pitch_sd"]),
                "energy_sd": None if pd.isna(r["energy_sd"]) else float(r["energy_sd"]),
                "post_low": float(r["gmm_posteriors"]["low_expr"]),
                "post_high": float(r["gmm_posteriors"]["high_expr"]),
                "feature_schema": r["feature_schema"],
            }
        } for _, r in df.iterrows()]

        base_cols = ["window_id","start_s","end_s","pitch_sd","energy_sd","ser_label","confidence","feature_schema"]
        df = df[base_cols + ["gmm_posteriors","descriptor","rag_context"]]
        return df, docs

