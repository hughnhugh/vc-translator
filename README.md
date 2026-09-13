# vc-translator

Live translator for Discord (or any Windows system audio). Captures your PC's
audio output via WASAPI loopback, detects speech with Silero VAD, and
translates it to English using `faster-whisper` — all running locally on
your GPU, no cloud APIs.

Shows a small always-on-top overlay with the original text (e.g. Chinese or
Vietnamese) and its English translation.

## Requirements

- Windows with an NVIDIA GPU (CUDA)
- Python 3.12

## Setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install faster-whisper PyAudioWPatch numpy opencc-python-reimplemented
```

## Run

```powershell
.\venv\Scripts\python.exe translate_vc.py
```

An overlay window appears near the bottom of your screen. Drag it to
reposition, click the ✕ or press Escape to close. Have Discord (or whatever
app) playing audio through your default output device.

## Notes

- First run downloads the Silero VAD model and the `large-v3` Whisper model
  (~3GB). Switch `MODEL_SIZE` in `translate_vc.py` to `"medium"` for lower
  latency on less powerful GPUs.
- Translates *into* English from whatever language is detected per segment.
- Chinese output is converted from Traditional to Simplified automatically.
