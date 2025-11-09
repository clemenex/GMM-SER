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
        low_list  = [int(i) for i in meta.get("low_expr_comps",  [meta.get("low_expr_comp", 0)])]
        high_list = [int(i) for i in meta.get("high_expr_comps", [meta.get("high_expr_comp", 1)])]

        thr = meta.get("thresholds", {})
        t_default = float(os.getenv("SER_THRESH", thr.get("symmetric", 0.90)))
        T_LOW  = float(os.getenv("SER_TLOW",  thr.get("low_expr", t_default)))
        T_HIGH = float(os.getenv("SER_THIGH", thr.get("high_expr", t_default)))

        art = SERArtifacts(
            scaler=scaler, gmm=gmm, meta=meta, features=features,
            low_list=low_list, high_list=high_list, T_LOW=T_LOW, T_HIGH=T_HIGH
        )
        return cls(art)

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

    def process_array(self, y: np.ndarray, sr: int, win_sec: float = 60.0, hop_sec: float = 60.0,
                      fmin: float = 65.0, fmax: float = 400.0
                      ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Full pipeline on in-memory audio: features → GMM → descriptors → RAG docs.
        Returns (df, docs) where docs = [{"text": rag_context, "metadata": {...}}, ...]
        """
        # 1) features per window
        df = self._features_from_array(y, sr, win_sec, hop_sec, fmin, fmax)

        # 2) align features & get posteriors
        X = df[self.art.features].to_numpy()
        Xz = self.art.scaler.transform(X)
        post = self.art.gmm.predict_proba(Xz)
        p_low  = post[:, self.art.low_list].sum(axis=1)
        p_high = post[:, self.art.high_list].sum(axis=1)

        # 3) mapping with dataset-level energy cues
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
        df["ser_label"] = [m["ser_label"] for m in mapped]
        df["confidence"] = [m["confidence"] for m in mapped]
        df["descriptor"] = [m["descriptor"] for m in mapped]

        # 4) RAG docs
        df["rag_context"] = df.apply(self._build_rag_text, axis=1)
        docs = []
        for _, r in df.iterrows():
            docs.append({
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
                }
            })

        # tidy column order
        base_cols = ["window_id", "start_s", "end_s", "pitch_sd", "energy_sd", "ser_label", "confidence"]
        df = df[base_cols + ["gmm_posteriors", "descriptor", "rag_context"]]
        return df, docs
