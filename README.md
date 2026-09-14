# vc-translator

Live translator for Discord (or any Windows system audio). Captures your PC's
audio output via WASAPI loopback plus your microphone, detects speech with
Silero VAD, and translates it to a target language of your choosing using
`faster-whisper` (and NLLB-200 for non-English targets) — all running
locally on your GPU, no cloud APIs.

Shows a small always-on-top overlay with the original text and its
translation. Run the script multiple times with different `--target` values
to get multiple simultaneous overlays (e.g. one for English, one for
Chinese).

## Requirements

- Windows with an NVIDIA GPU (CUDA)
- Python 3.12

## Setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install faster-whisper PyAudioWPatch numpy opencc-python-reimplemented transformers sentencepiece
```

## Run

```powershell
.\venv\Scripts\python.exe translate_vc.py --target en
.\venv\Scripts\python.exe translate_vc.py --target zh
```

An overlay window appears on your screen (bottom for `en`, top for other
targets by default). Drag it to reposition, click the ✕ or press Escape to
close, drag the bottom-right corner to resize, and `_` to minimize it to a
small tab. Have Discord (or whatever app) playing audio through your default
output device — your microphone is captured too, tagged `[You]`.

### Options

- `--target LANG` - target language code (default `en`). 93 languages
  supported out of the box (see `lang_codes.py`), including `zh` (Simplified
  Chinese) and `zh-hant` (Traditional Chinese) as separate targets. Codes
  are validated against the loaded NLLB tokenizer at startup - add more to
  `WHISPER_TO_FLORES` in `lang_codes.py` (needs a valid NLLB flores200 code).
- `--model SIZE` - Whisper model size (default: `large-v3` for `en`,
  `medium` for everything else, to leave GPU memory for the second overlay's
  translation model).
- `--position top|bottom` - overlay screen anchor (default: `bottom` for
  `en`, `top` otherwise).
- `--no-mic` - don't capture your microphone, system audio only.

## Notes

- First run downloads the Silero VAD model, the Whisper model (~3GB for
  `large-v3`), and for non-English targets, NLLB-200 (~1.2GB).
- Whisper natively translates any spoken language straight into English in
  one pass. For every other target, it transcribes the native speech first,
  then a second model (NLLB-200) translates that text into the target
  language.
- Chinese speech is normalized to Simplified for display; pick `--target
  zh-hant` if you want Traditional output instead.
- Every session's captions are logged to `logs/` (gitignored) with
  timestamps, one file per run.
