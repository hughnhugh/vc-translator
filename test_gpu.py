from faster_whisper import WhisperModel

print("Loading tiny model on CUDA to test GPU compatibility...")
model = WhisperModel("tiny", device="cuda", compute_type="float16")
print("Model loaded successfully on GPU.")
