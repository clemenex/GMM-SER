# ser_tester.py
import inspect, ser_pipeline
from ser_pipeline import SERModel

print("Loaded from:", ser_pipeline.__file__)
print("Signature   :", inspect.signature(SERModel.load_from_paths))

ser = SERModel.load_from_paths(
    scaler_path="models/ser60_scaler.joblib",
    gmm_path="models/ser60_gmm2.joblib",
    meta_path="models/ser60_meta.joblib",
)

df, docs = ser.process_audio_file("inputs/301_AUDIO.wav", win_sec=60.0, hop_sec=60.0)
print(df.head(20))
print("Docs:", len(docs))
