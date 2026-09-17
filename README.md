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

- **Windows 10/11.** This uses WASAPI loopback audio capture and Win32-specific
  packages - it will not run on macOS/Linux.
- **An NVIDIA GPU with CUDA support, and its driver installed.** Whisper
  (large-v3 especially) and NLLB are heavy models - this isn't optional, there's
  no CPU fallback mode. 8GB+ VRAM recommended for `large-v3`; smaller Whisper
  models (see Options below) work fine on less.
- **Python 3.12.** Get it from [python.org/downloads](https://www.python.org/downloads/)
  - during install, check **"Add python.exe to PATH"**, and check
  **"py launcher"** (on by default) - both are just checkboxes on the first
  install screen.
- **~10GB free disk space** - dependencies (~4-5GB, mostly PyTorch+CUDA) plus
  models downloaded on first run (~5GB - see Notes below).
- **Internet connection** for setup and for the one-time model download on
  first launch. Not needed after that.

## Setup

**Option A - automated (recommended):** double-click **`setup.bat`** in this
folder. It finds your Python install, creates the project's virtual
environment (`venv\`), and installs everything, printing clear errors if
something's missing (e.g. no Python found, or no NVIDIA GPU detected). Safe
to re-run any time - it reuses an existing `venv\` rather than recreating it,
so re-running after a `git pull` just catches up on any new dependencies.

If Windows won't run `setup.ps1` directly (only relevant if you run the
`.ps1` yourself instead of double-clicking `setup.bat`), that's PowerShell's
script-execution policy blocking it by default - `setup.bat` already works
around this for you by invoking it with the restriction bypassed for just
that one run, without changing any system-wide setting.

**Option B - manual**, if you'd rather see/control each step:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install faster-whisper PyAudioWPatch numpy opencc-python-reimplemented transformers sentencepiece websockets
.\venv\Scripts\python.exe -m pip install nvidia-ml-py
```

`nvidia-ml-py` (the last line) is optional (Settings' GPU memory readout
only) - queries the driver directly via NVML so it reflects everything using
VRAM, including Whisper (which runs through ctranslate2, invisible to
PyTorch's own memory stats). Without it, the readout falls back to a
PyTorch-only number that misses Whisper's usage entirely.

## Run

```powershell
.\venv\Scripts\python.exe translate_vc.py
.\venv\Scripts\python.exe translate_vc.py --target en
.\venv\Scripts\python.exe translate_vc.py --target zh
.\venv\Scripts\python.exe translate_vc.py --target en,vi
```

This is a persistent app, not a one-shot script: a **Settings window** is
always available - a normal, taskbar-visible window (alt-tabbable like any
other app) where you pick target languages (by name, e.g. "Vietnamese", not
code), Whisper model size, overlay position, mic capture, and the Discord
bridge port, all editable live while it keeps running - checking/unchecking
a language opens/closes its overlay immediately, no restart needed.

The Settings window always starts visible right now (whether or not you
pass `--target`) - it's meant to stay hidden for scripted `--target`
launches with the tray as the way back in, but the tray is currently
disabled (see below), so hiding it would strand it unreachable.

**Window behavior:** closing Settings (✕) quits the whole app - streams,
models, the Discord bridge, everything. Minimizing behaves like a normal
window for now - a system tray/minimize-to-tray integration was tried
(`tray.py`) but pystray's native Win32 message-loop thread turned out to
segfault against Tkinter's own message loop on Windows, so it's disabled
until a safer approach is found.

An overlay window appears per active target (bottom for `en`, top for other
targets by default; windows sharing an anchor stack instead of overlapping).
Drag one to reposition, click its ✕ or press Escape to close just that
window (unchecks it in Settings too), drag the bottom-right corner to
resize, `_` to minimize it to a small tab, `T` to toggle that window's
translation off/on (dims when off) - handy if you just want the native
transcription without a second line underneath - and `Me` to toggle your
own mic captioning off/on (shared across every window, since it's one mic
feed) - handy for muting yourself out of the captions without disabling
your mic in Discord itself. Have Discord (or whatever app) playing audio
through your default output device — your microphone is captured too,
tagged `[You]`, unless toggled off.

### Options

- `--target LANG[,LANG...]` - one or more comma-separated target language
  codes to start captioning immediately (default: none - starts with
  Settings open instead). 93 languages supported out of the box (see
  `lang_codes.py`), including `zh` (Simplified Chinese) and `zh-hant`
  (Traditional Chinese) as separate targets. Codes are validated against the
  loaded NLLB tokenizer at startup - add more to `WHISPER_TO_FLORES` (and a
  display name to `LANGUAGE_NAMES`) in `lang_codes.py` (needs a valid NLLB
  flores200 code). More targets can always be added live from Settings too.
- `--model SIZE` - initial Whisper model size (default: `large-v3`) -
  changeable live from Settings afterward (briefly pauses captioning while
  it swaps).
- `--position top|bottom` - initial overlay screen anchor for every window
  (default: `bottom` for `en`, `top` otherwise, chosen per target) -
  changeable live from Settings afterward.
- `--no-mic` - don't capture your microphone at startup - changeable live
  from Settings afterward.

## Speaker identification (optional, Discord only, advanced setup)

This part is meaningfully harder to set up than everything above - it needs
a full Vencord *source* build (there's no marketplace install, since this
plugin isn't an official Vencord one), which means Git, Node.js, and `pnpm`,
not just Python. `vc-translator` is fully usable without it - you only need
this if you specifically want real Discord usernames attached to captions
instead of unlabeled ones. See `vencord-plugin/README.md` for the full
build/install steps if you want it.

Speaker labels come only from ground truth, never a guess: a companion
Vencord plugin (`vencord-plugin/discordSpeakingBridge.ts`) reads Discord's
own "who is currently speaking" state (the same signal behind the green
speaking ring in the UI) and the real username, and pushes that to a small
local WebSocket server `translate_vc.py` runs (`--discord-bridge-port`,
default `8765`, also changeable live from Settings). When connected,
loopback-audio segment boundaries and labels come straight from Discord's
own speaking start/stop events. See `vencord-plugin/README.md` for
build/install steps. The overlay shows a one-line status whenever the
bridge connects or drops, so it's always clear which segmentation path is
active. Note changing the port from Settings only affects the Python side -
the Vencord plugin has its own separate `port` setting to match.

Without the plugin connected, `vc-translator` works exactly the same, just
with unlabeled captions for system audio (your own `[You]` mic captions are
always labeled). Note the bridge can't isolate simultaneous speakers' audio
from each other (that would need a bot with real per-user voice-receive) -
it only improves *identity and segment timing* on top of the existing
mixed loopback audio.

## Notes

- First run downloads the Silero VAD model, the Whisper model (~3GB for
  `large-v3`), and NLLB-200 (~1.2GB) - NLLB now always loads at startup
  (not just when a non-English target is requested), since targets are
  addable live from Settings at any point during a run.
- Whisper natively translates any spoken language straight into English in
  one pass. For every other target, it transcribes the native speech first,
  then a second model (NLLB-200) translates that text into the target
  language.
- Chinese speech is normalized to Simplified for display; pick `--target
  zh-hant` if you want Traditional output instead.
- Every session's captions are logged to `logs/` (gitignored) with
  timestamps, one file per overlay window (target) per run.

## Troubleshooting

- **"No Python install found" from `setup.bat`/`setup.ps1`.** Install Python
  3.12 from [python.org/downloads](https://www.python.org/downloads/),
  checking "Add python.exe to PATH" during install, then re-run.
- **"'nvidia-smi' not found" warning during setup.** Means either there's no
  NVIDIA GPU, or its driver isn't installed - install the latest driver from
  [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx). The app
  will not run without a working NVIDIA GPU + driver.
- **PowerShell won't run `setup.ps1` directly** ("running scripts is disabled
  on this system"). Use `setup.bat` instead - it invokes the `.ps1` with the
  restriction bypassed for just that one run. (Or manually:
  `powershell -ExecutionPolicy Bypass -File setup.ps1`.)
- **First launch seems stuck / slow.** It's downloading the Whisper, NLLB,
  and VAD models (~5GB total, one-time) - check the Settings window's status
  panel, which shows "Loading..." per component. A slow/metered connection
  can make this take several minutes.
- **Symlink warning from `huggingface_hub` on first run.** Harmless - Windows
  needs Developer Mode or admin rights for symlinks; without them, model
  files are just copied instead, using a bit more disk space.
- **No captions ever appear for other people talking.** Check the overlay's
  status line for `[Capturing system audio: ...]` - it should name your
  actual output device (headphones/speakers). If Discord's output device
  differs from your Windows default output device, loopback capture won't
  hear it - either match them, or see PyAudioWPatch's docs for capturing a
  specific device.
- **Model reload / GPU memory reading not moving.** The Settings window's
  "Whisper model" status row is the source of truth for whether a model
  switch actually completed (`Ready (size)`); the GPU row needs
  `nvidia-ml-py` installed to see Whisper's memory at all (see Setup above).
