# tests/test_ser_mapping.py
import unittest
import types
import pandas as pd
import numpy as np

#
# We import your inference module. Adjust the path if needed.
#
import ser_gmm_infer as S

class TestSERMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure confidence threshold (you can change if you tuned it)
        S.THRESH_CONFIDENT = 0.70

        # Ensure component indices are set (low vs high expressiveness)
        #  - We'll assume component 0 = low_expr, component 1 = high_expr for tests.
        #  - If your saved meta says otherwise, set accordingly in the test.
        S.low_expr_comp = 0
        S.high_expr_comp = 1

        # Provide a tiny df_ref so percentile-based nudges are deterministic.
        # energy_sd distribution: 10, 20, 30, 40, 50  -> 25th pct = 20, 75th pct = 40
        S.df_ref = pd.DataFrame({"energy_sd": [10, 20, 30, 40, 50]})

    def test_flat_label_high_confidence(self):
        # Probabilities: strong low-expressiveness
        probs = np.array([0.85, 0.15])  # [P(low), P(high)]
        row = pd.Series({"pitch_sd": 540.0, "energy_sd": 18.0})
        out = S.map_to_descriptor(row, probs)

        self.assertEqual(out["ser_label"], "flat_prosody")
        self.assertEqual(out["confidence"], "high")
        self.assertIn("monotone/flat prosody", out["descriptor"])
        # Since energy_sd (18) < 25th percentile (20), low-energy nudge should appear
        self.assertIn("Low energy variation is also observed.", out["descriptor"])

        # Basic structure checks
        self.assertIn("gmm_posteriors", out)
        self.assertIn("low_expr", out["gmm_posteriors"])
        self.assertIn("high_expr", out["gmm_posteriors"])

    def test_expressive_label_high_confidence(self):
        # Probabilities: strong high-expressiveness
        probs = np.array([0.10, 0.90])  # [P(low), P(high)]
        row = pd.Series({"pitch_sd": 900.0, "energy_sd": 45.0})
        out = S.map_to_descriptor(row, probs)

        self.assertEqual(out["ser_label"], "expressive_prosody")
        self.assertEqual(out["confidence"], "high")
        self.assertIn("expressive intonation", out["descriptor"])
        # energy_sd (45) > 75th percentile (40), so elevated-energy nudge expected
        self.assertIn("Elevated energy variation is also observed.", out["descriptor"])

    def test_ambiguous_label_low_confidence(self):
        # Neither posterior crosses threshold
        probs = np.array([0.55, 0.45])
        row = pd.Series({"pitch_sd": 700.0, "energy_sd": 30.0})
        out = S.map_to_descriptor(row, probs)

        self.assertEqual(out["ser_label"], "prosody_ambiguous")
        self.assertEqual(out["confidence"], "low")
        self.assertIn("inconclusive", out["descriptor"])

    def test_missing_energy_does_not_crash(self):
        # energy_sd missing -> no nudge; still maps label by posterior
        probs = np.array([0.80, 0.20])
        row = pd.Series({"pitch_sd": 520.0})  # no energy_sd
        out = S.map_to_descriptor(row, probs)

        self.assertEqual(out["ser_label"], "flat_prosody")
        self.assertEqual(out["confidence"], "high")
        # No KeyError, and descriptor remains valid text
        self.assertIn("monotone/flat prosody", out["descriptor"])

    def test_rag_context_build_string(self):
        # Simulate a row after infer_ser_descriptors
        fake_row = {
            "descriptor": "Speech shows monotone/flat prosody with diminished emotional expressiveness.",
            "gmm_posteriors": {"low_expr": 0.82, "high_expr": 0.18},
            "pitch_sd": 535.0,
            "energy_sd": 19.0,
        }

        # Recreate the make_rag_context callable from your script
        def make_rag_context(row):
            return (
                f"Observed prosody: {row['descriptor']} "
                f"(posterior low_expr={row['gmm_posteriors']['low_expr']:.2f}, "
                f"high_expr={row['gmm_posteriors']['high_expr']:.2f}; "
                f"pitch_sd={row['pitch_sd']}, energy_sd={row['energy_sd']}). "
                "Retrieve DSM-5-TR sections where diminished emotional expression, flat affect, "
                "or prosody-related observations are clinically relevant for diagnosis/differential diagnosis."
            )

        ctx = make_rag_context(fake_row)
        self.assertIn("Observed prosody:", ctx)
        self.assertIn("low_expr=0.82", ctx)
        self.assertIn("DSM-5-TR", ctx)

if __name__ == "__main__":
    unittest.main()
