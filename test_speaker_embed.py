import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

print("Loading ECAPA-TDNN speaker embedding model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"},
    local_strategy=LocalStrategy.COPY,
)

# 2 seconds of random noise at 16kHz just to confirm the forward pass works
audio = np.random.uniform(-0.1, 0.1, size=16000 * 2).astype(np.float32)
wav = torch.from_numpy(audio).unsqueeze(0)
with torch.no_grad():
    embedding = classifier.encode_batch(wav).squeeze().cpu().numpy()

print("Embedding shape:", embedding.shape)
print("First 5 values:", embedding[:5])
