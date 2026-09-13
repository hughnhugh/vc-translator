"""
Live translator for Discord (or any system audio).
Captures whatever your PC is currently playing (WASAPI loopback),
detects speech with Silero VAD, and translates it to English text
using faster-whisper (task="translate"), which works directly from
Vietnamese, Chinese (or most other languages) to English.

Shows captions in a small always-on-top overlay window.
"""

import queue
import sys
import threading
import time
import tkinter as tk

import numpy as np
import pyaudiowpatch as pyaudio
import torch
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from opencc import OpenCC

_t2s = OpenCC("t2s")

MODEL_SIZE = "large-v3"   # try "medium" if this feels laggy on your GPU
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

VAD_SAMPLE_RATE = 16000
FRAME_SAMPLES = 512              # 32ms @ 16kHz, required chunk size for Silero VAD
SPEECH_THRESHOLD = 0.5
SILENCE_HANGOVER_SEC = 0.8       # how much trailing silence ends a segment
MIN_SPEECH_SEC = 0.4             # ignore blips shorter than this
MAX_SEGMENT_SEC = 6.0            # force a cut even mid-sentence so long monologues don't stall output

MAX_LINES_SHOWN = 4

audio_q: "queue.Queue[bytes]" = queue.Queue()
caption_q: "queue.Queue[str]" = queue.Queue()


def pyaudio_callback(in_data, frame_count, time_info, status):
    audio_q.put(in_data)
    return (None, pyaudio.paContinue)


def open_loopback_stream(p):
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if not default_speakers.get("isLoopbackDevice", False):
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                break
        else:
            raise RuntimeError(
                "Could not find a loopback device matching your default speakers. "
                "Try setting your Discord/Windows output device explicitly."
            )

    caption_q.put(f"[Capturing: {default_speakers['name']}]")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=default_speakers["maxInputChannels"],
        rate=int(default_speakers["defaultSampleRate"]),
        frames_per_buffer=1024,
        input=True,
        input_device_index=default_speakers["index"],
        stream_callback=pyaudio_callback,
    )
    return stream, default_speakers


def resample_linear(audio, orig_sr, target_sr):
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    duration = len(audio) / orig_sr
    target_len = max(1, int(duration * target_sr))
    x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def process_segment(model, segment):
    t0 = time.time()
    native_segments, info = model.transcribe(
        segment,
        task="transcribe",
        vad_filter=False,
        beam_size=5,
    )
    native_text = "".join(s.text for s in native_segments).strip()

    if info.language == "zh":
        native_text = _t2s.convert(native_text)

    if info.language == "en":
        elapsed = time.time() - t0
        if native_text:
            caption_q.put(f"[en, {elapsed:.1f}s] {native_text}")
        return

    en_segments, _ = model.transcribe(
        segment,
        task="translate",
        vad_filter=False,
        beam_size=5,
        language=info.language,
    )
    en_text = "".join(s.text for s in en_segments).strip()

    elapsed = time.time() - t0
    if native_text or en_text:
        caption_q.put(f"[{info.language}, {elapsed:.1f}s] {native_text}\n    -> {en_text}")


def audio_worker(stop_event: threading.Event):
    try:
        caption_q.put("Loading VAD model...")
        vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

        caption_q.put(f"Loading Whisper model ({MODEL_SIZE}) on {DEVICE}...")
        whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

        p = pyaudio.PyAudio()
        stream, device_info = open_loopback_stream(p)
        native_rate = int(device_info["defaultSampleRate"])
        channels = device_info["maxInputChannels"]

        stream.start_stream()
        caption_q.put("Listening...")

        executor = ThreadPoolExecutor(max_workers=1)

        leftover = np.zeros(0, dtype=np.float32)
        speech_buffer = []
        in_speech = False
        silence_frames = 0
        silence_frames_needed = int(SILENCE_HANGOVER_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)
        min_speech_frames = int(MIN_SPEECH_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)

        while not stop_event.is_set():
            try:
                raw = audio_q.get(timeout=1)
            except queue.Empty:
                continue

            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1)

            pcm_16k = resample_linear(pcm, native_rate, VAD_SAMPLE_RATE)
            buf = np.concatenate([leftover, pcm_16k])

            n_frames = len(buf) // FRAME_SAMPLES
            usable = n_frames * FRAME_SAMPLES
            leftover = buf[usable:]

            for i in range(n_frames):
                frame = buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
                prob = vad_model(torch.from_numpy(frame), VAD_SAMPLE_RATE).item()

                if prob >= SPEECH_THRESHOLD:
                    speech_buffer.append(frame)
                    silence_frames = 0
                    in_speech = True
                elif in_speech:
                    speech_buffer.append(frame)
                    silence_frames += 1

                if in_speech:
                    total_frames = len(speech_buffer)
                    duration_sec = total_frames * FRAME_SAMPLES / VAD_SAMPLE_RATE
                    hit_silence_end = silence_frames >= silence_frames_needed
                    hit_max_duration = duration_sec >= MAX_SEGMENT_SEC
                    if hit_silence_end or hit_max_duration:
                        if total_frames - silence_frames >= min_speech_frames:
                            segment = np.concatenate(speech_buffer)
                            executor.submit(process_segment, whisper_model, segment)
                        speech_buffer = []
                        silence_frames = 0
                        # a forced max-duration cut doesn't mean silence started -
                        # keep accumulating the next chunk right away
                        in_speech = not hit_silence_end

        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        caption_q.put(f"[ERROR] {type(e).__name__}: {e}")


class OverlayApp:
    def __init__(self, root: tk.Tk, stop_event: threading.Event):
        self.root = root
        self.stop_event = stop_event
        self.lines = []

        root.overrideredirect(True)          # borderless
        root.attributes("-topmost", True)    # always on top
        root.attributes("-alpha", 0.85)      # slight transparency
        root.configure(bg="black")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w, win_h = int(screen_w * 0.6), 260
        x = (screen_w - win_w) // 2
        y = screen_h - win_h - 80
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        root.pack_propagate(False)

        self.label = tk.Label(
            root,
            text="Starting...",
            fg="white",
            bg="black",
            font=("Microsoft YaHei UI", 13),
            justify="left",
            anchor="sw",
            wraplength=win_w - 40,
        )
        self.label.pack(fill="both", expand=True, padx=15, pady=10)

        close_btn = tk.Label(root, text="✕", fg="white", bg="black", font=("Segoe UI", 10))
        close_btn.place(relx=1.0, x=-20, y=5)
        close_btn.bind("<Button-1>", lambda e: self.close())

        # drag to move
        root.bind("<ButtonPress-1>", self._start_drag)
        root.bind("<B1-Motion>", self._do_drag)
        self.label.bind("<ButtonPress-1>", self._start_drag)
        self.label.bind("<B1-Motion>", self._do_drag)

        root.bind("<Escape>", lambda e: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.root.after(100, self.poll_queue)

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_drag(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def poll_queue(self):
        updated = False
        while True:
            try:
                text = caption_q.get_nowait()
            except queue.Empty:
                break
            self.lines.append(text)
            self.lines = self.lines[-MAX_LINES_SHOWN:]
            updated = True

        if updated:
            self.label.config(text="\n\n".join(self.lines))

        self.root.after(100, self.poll_queue)

    def close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    stop_event = threading.Event()
    worker = threading.Thread(target=audio_worker, args=(stop_event,), daemon=True)
    worker.start()

    root = tk.Tk()
    OverlayApp(root, stop_event)
    root.mainloop()


if __name__ == "__main__":
    main()
