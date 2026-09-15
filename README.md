# vc-translator

Live translator for Discord (or any Windows system audio). Captures your PC's
audio output via WASAPI loopback plus your microphone, detects speech with
Silero VAD, and translates it to a target language of your choosing using
`faster-whisper` (and NLLB-200 for non-English targets) — all running
locally on your GPU, no cloud APIs.

Shows a small always-on-top overlay with the original text and its
translation. Pass a comma-separated `--target` list to get multiple
simultaneous overlays (e.g. one for English, one for Chinese) from a single
process - audio capture and Whisper transcription happen once per spoken
segment and are shared across every target; only the (cheap) translation
step runs once per target.

## Requirements

- Windows with an NVIDIA GPU (CUDA)
- Python 3.12

## Setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install faster-whisper PyAudioWPatch numpy opencc-python-reimplemented transformers sentencepiece websockets
```

## Run

```powershell
.\venv\Scripts\python.exe translate_vc.py --target en
.\venv\Scripts\python.exe translate_vc.py --target zh
.\venv\Scripts\python.exe translate_vc.py --target en,vi
```

An overlay window appears per target (bottom for `en`, top for other
targets by default; windows sharing an anchor stack instead of overlapping).
Drag one to reposition, click its ✕ or press Escape to close just that
window, drag the bottom-right corner to resize, and `_` to minimize it to a
small tab. Closing the last remaining window ends the process. Have Discord
(or whatever app) playing audio through your default output device — your
microphone is captured too, tagged `[You]`.

### Options

- `--target LANG[,LANG...]` - one or more comma-separated target language
  codes (default `en`). 93 languages supported out of the box (see
  `lang_codes.py`), including `zh` (Simplified Chinese) and `zh-hant`
  (Traditional Chinese) as separate targets. Codes are validated against the
  loaded NLLB tokenizer at startup - add more to `WHISPER_TO_FLORES` in
  `lang_codes.py` (needs a valid NLLB flores200 code).
- `--model SIZE` - Whisper model size, shared by every target in this
  process (default: `large-v3`).
- `--position top|bottom` - overlay screen anchor for every window (default:
  `bottom` for `en`, `top` otherwise, chosen per target).
- `--no-mic` - don't capture your microphone, system audio only.

## Speaker identification (optional, Discord only)

Speaker labels come only from ground truth, never a guess: a companion
Vencord plugin (`vencord-plugin/discordSpeakingBridge.ts`) reads Discord's
own "who is currently speaking" state (the same signal behind the green
speaking ring in the UI) and the real username, and pushes that to a small
local WebSocket server `translate_vc.py` runs (`--discord-bridge-port`,
default `8765`). When connected, loopback-audio segment boundaries and
labels come straight from Discord's own speaking start/stop events. See
`vencord-plugin/README.md` for build/install steps.

Without the plugin connected, `vc-translator` works exactly the same, just
with unlabeled captions for system audio (your own `[You]` mic captions are
always labeled). Note the bridge can't isolate simultaneous speakers' audio
from each other (that would need a bot with real per-user voice-receive) -
it only improves *identity and segment timing* on top of the existing
mixed loopback audio.

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
  timestamps, one file per overlay window (target) per run.
