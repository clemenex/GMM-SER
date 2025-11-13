import inspect, ser_pipeline
from ser_pipeline import SERModel

print("Loaded from:", ser_pipeline.__file__)
print("Signature   :", inspect.signature(SERModel.load_from_paths))

ser = SERModel.load_from_paths(
    scaler_path="models/ser60_scaler.joblib",
    gmm_path="models/ser60_gmm2.joblib",
    meta_path="models/ser60_meta.joblib",
)

result = ser.process_audio_file("inputs/300_AUDIO.wav", rag_txt_path="outputs/rag_txt/rag_txt.txt")
print(result.keys())

docs = result["docs"]

print(result["rag_prompt_path"])