"""
Shared audio-capture, VAD segmentation, and overlay-window plumbing used by
translate_vc.py. A single process_segment_fn and audio_worker are shared
across every requested target language; run_translator owns the persistent
app: a taskbar-visible Settings window (live-editable options), a system
tray icon, and one overlay window per active target - windows can be
opened/closed live as targets are added/removed from Settings.
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pyaudiowpatch as pyaudio
import torch
from faster_whisper import WhisperModel

import discord_bridge
import tray

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

VAD_SAMPLE_RATE = 16000
FRAME_SAMPLES = 512              # 32ms @ 16kHz, required chunk size for Silero VAD
SPEECH_THRESHOLD = 0.5
SILENCE_HANGOVER_SEC = 0.8       # how much trailing silence ends a segment
MIN_SPEECH_SEC = 0.4             # ignore blips shorter than this
MAX_SEGMENT_SEC = 6.0            # force a cut even mid-sentence so long monologues don't stall output

# How far back the loopback thread keeps timestamped audio, so a Discord
# speaking_stop event can slice out exactly the span it covered. Only used
# when discord_bridge.is_active() - see _drain_discord_segments.
BRIDGE_BUFFER_WINDOW_SEC = 20.0
BRIDGE_SLICE_PAD_SEC = 0.2        # clock-skew cushion on each side of a slice

MAX_HISTORY = 300                # cap on stored caption entries so the log doesn't grow unbounded

# Anti-hallucination thresholds: Whisper doesn't say "nothing here" for silence
# or noise it's unsure about - it fabricates plausible-sounding text instead.
# These mirror the heuristics OpenAI's own reference decoder uses to flag that.
NO_SPEECH_PROB_MAX = 0.6
AVG_LOGPROB_MIN = -1.0
LANGUAGE_PROB_MIN = 0.5

WINDOW_W_FRAC = 0.36  # 60% of the previous 0.6
WINDOW_H = 260
STACK_GAP = 20
ANCHOR_MARGIN = 80


class SelfCaptionState:
    """Shared on/off flag for mic ("[You]") captioning, toggled from any
    overlay window's "Me" button - every window shares one flag, so toggling
    it in one place updates all of them. Mic segments simply aren't
    submitted for transcription while off, so this also saves the GPU work,
    not just hides the caption."""

    def __init__(self):
        self.enabled = True
        self._buttons: "list[tk.Label]" = []

    def register(self, button):
        self._buttons.append(button)
        self._refresh(button)

    def toggle(self):
        self.enabled = not self.enabled
        for button in self._buttons:
            self._refresh(button)

    def _refresh(self, button):
        button.config(fg="white" if self.enabled else "#555555")


class ModelHolder:
    """Thread-safe holder for the current WhisperModel, swappable at runtime.
    The capture loop fetches .get() at *submission* time (not once at
    worker-start), so a reload never yanks the model out from under an
    already-submitted transcription job. request_reload() queues the swap on
    the shared single-worker executor, so it's naturally serialized with
    in-flight/queued segments - no extra locking needed for that part."""

    def __init__(self, size, device="cuda", compute_type="float16"):
        self._lock = threading.Lock()
        self._model = None
        self.size = size
        self.device = device
        self.compute_type = compute_type
        self._executor = None

    def get(self):
        with self._lock:
            return self._model

    def _load_initial(self):
        status.set("whisper_model", f"Loading ({self.size})...")
        broadcast(f"Loading Whisper model ({self.size}) on {self.device}...")
        model = WhisperModel(self.size, device=self.device, compute_type=self.compute_type)
        with self._lock:
            self._model = model
        status.set("whisper_model", f"Ready ({self.size})")

    def _bind_executor(self, executor):
        self._executor = executor

    def request_reload(self, new_size):
        """Called from the Settings window (Tk thread) when the user picks a
        different model size. No-op if the pipeline isn't ready yet or the
        size didn't actually change."""
        if self._executor is None or new_size == self.size or not new_size:
            return

        def _do_reload():
            old_size = self.size
            status.set("whisper_model", f"Reloading ({old_size} -> {new_size})...")
            broadcast(f"Reloading Whisper model ({old_size} -> {new_size})...")
            try:
                new_model = WhisperModel(new_size, device=self.device, compute_type=self.compute_type)
                old_model = self.get()
                with self._lock:
                    self._model = new_model
                    self.size = new_size
                del old_model
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                status.set("whisper_model", f"Ready ({new_size})")
                broadcast(f"Model reloaded ({new_size}).")
            except Exception as e:
                status.set("whisper_model", f"Error: {type(e).__name__}")
                broadcast(f"[ERROR] model reload failed: {type(e).__name__}: {e}")

        self._executor.submit(_do_reload)


class MicCapture:
    """Owns the mic PyAudio stream + capture thread, startable/stoppable
    independently of loopback capture and the rest of the pipeline. Usable
    (checkbox-driven) before the audio pipeline finishes loading - toggling
    early just changes what happens once bind() runs after load."""

    def __init__(self, initially_on):
        self._desired_on = initially_on
        self._p = None
        self._model_holder = None
        self._executor = None
        self._process_segment_fn = None
        self._self_caption_state = None
        self._stream = None
        self._thread = None
        self._stop_event = None

    @property
    def enabled(self):
        return self._desired_on

    @property
    def active(self):
        return self._thread is not None

    def bind(self, p, model_holder, executor, process_segment_fn, self_caption_state):
        self._p, self._model_holder, self._executor = p, model_holder, executor
        self._process_segment_fn, self._self_caption_state = process_segment_fn, self_caption_state
        if self._desired_on:
            self._start_now()
        else:
            status.set("mic", "Off")

    def set_enabled(self, on):
        """Safe to call from the Tk thread (the Settings mic checkbox does) -
        the actual start work happens on a background thread since it can
        block for a while (torch.hub.load, opening the audio device), and
        blocking the Tk thread would freeze the whole UI."""
        self._desired_on = on
        if self._p is None:
            return  # not bound yet - bind() will apply _desired_on once loaded
        if on:
            threading.Thread(target=self._start_now, daemon=True).start()
        else:
            self._stop_now()

    def _start_now(self):
        if self.active:
            return
        status.set("mic", "Loading...")
        # Silero VAD is a stateful streaming model (it carries hidden state
        # between calls) - it needs its own instance, separate from
        # loopback's, same as at initial startup.
        vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        if not self._desired_on:
            status.set("mic", "Off")
            return  # toggled off again while VAD was loading - abandon

        mic_q: "queue.Queue" = queue.Queue()
        stream, info, channels = open_mic_stream(self._p, make_pyaudio_callback(mic_q))
        if not self._desired_on:
            stream.close()
            status.set("mic", "Off")
            return  # toggled off again while the device was opening - abandon

        stream.start_stream()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_capture_loop,
            args=(stop_event, mic_q, int(info["defaultSampleRate"]), channels, vad_model,
                  self._model_holder, self._executor, self._process_segment_fn, "mic", self._self_caption_state),
            daemon=True,
        )
        thread.start()
        self._stream, self._thread, self._stop_event = stream, thread, stop_event

    def _stop_now(self):
        if not self.active:
            return
        self._stop_event.set()
        self._stream.stop_stream()
        self._stream.close()
        self._stream = self._thread = self._stop_event = None
        status.set("mic", "Off")

    def shutdown(self):
        self._stop_now()


_broadcast_queues: "list[queue.Queue]" = []


def register_broadcast_queues(queues):
    _broadcast_queues[:] = queues


def broadcast(msg):
    for q in _broadcast_queues:
        q.put(msg)


STATUS_FIELDS = [
    ("vad", "VAD"),
    ("whisper_model", "Whisper model"),
    ("nllb", "Translation (NLLB)"),
    ("gpu", "GPU memory"),
    ("loopback", "System audio"),
    ("mic", "Microphone"),
    ("discord_bridge", "Discord bridge"),
]


class SystemStatus:
    """Structured per-subsystem status (VAD/Whisper/NLLB/audio devices/
    Discord bridge), mirrored into the Settings window as a small table
    instead of a single rolling status line. set() is called from whichever
    thread owns that subsystem (the audio worker thread, the model-reload
    executor thread, etc.) - safe from any thread, since listeners are
    plain queue.Queue.put() calls."""

    def __init__(self):
        self.values = {key: "..." for key, _ in STATUS_FIELDS}
        self._listeners: "list[queue.Queue]" = []

    def subscribe(self, q: "queue.Queue"):
        self._listeners.append(q)
        q.put(dict(self.values))  # immediate snapshot

    def set(self, key, text):
        self.values[key] = text
        snapshot = dict(self.values)
        for q in self._listeners:
            q.put(snapshot)


status = SystemStatus()


def is_reliable_transcription(segments, info):
    """False if this looks like a hallucination on silence/noise rather than real speech."""
    if not segments or info.language_probability < LANGUAGE_PROB_MIN:
        return False
    avg_no_speech = sum(s.no_speech_prob for s in segments) / len(segments)
    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
    return avg_no_speech <= NO_SPEECH_PROB_MAX and avg_logprob >= AVG_LOGPROB_MIN


def make_pyaudio_callback(audio_q: "queue.Queue[tuple[float, bytes]]"):
    def callback(in_data, frame_count, time_info, status):
        audio_q.put((time.time(), in_data))
        return (None, pyaudio.paContinue)
    return callback


def open_loopback_stream(p, callback):
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

    broadcast(f"[Capturing system audio: {default_speakers['name']}]")
    status.set("loopback", f"Capturing: {default_speakers['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=default_speakers["maxInputChannels"],
        rate=int(default_speakers["defaultSampleRate"]),
        frames_per_buffer=1024,
        input=True,
        input_device_index=default_speakers["index"],
        stream_callback=callback,
    )
    return stream, default_speakers


def open_mic_stream(p, callback):
    default_mic = p.get_default_input_device_info()
    channels = min(int(default_mic["maxInputChannels"]), 2) or 1

    broadcast(f"[Capturing mic: {default_mic['name']}]")
    status.set("mic", f"Capturing: {default_mic['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=int(default_mic["defaultSampleRate"]),
        frames_per_buffer=1024,
        input=True,
        input_device_index=default_mic["index"],
        stream_callback=callback,
    )
    return stream, default_mic, channels


def resample_linear(audio, orig_sr, target_sr):
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    duration = len(audio) / orig_sr
    target_len = max(1, int(duration * target_sr))
    x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _capture_loop(stop_event, audio_q, native_rate, channels, vad_model, model_holder, executor,
                   process_segment_fn, source_tag, self_caption_state=None):
    try:
        _capture_loop_inner(stop_event, audio_q, native_rate, channels, vad_model, model_holder, executor,
                             process_segment_fn, source_tag, self_caption_state)
    except Exception as e:
        label = "mic" if source_tag == "mic" else "system audio"
        broadcast(f"[ERROR] {label} capture stopped: {type(e).__name__}: {e}")


def _slice_timestamped_buffer(buffer, start_ts, end_ts):
    """buffer: deque of (chunk_start_ts, pcm_16k) - concatenate whichever
    chunks overlap [start_ts, end_ts] (padded for clock skew). Only
    chunk-level, not sample-exact, precision - Discord's timestamps and our
    local capture-time timestamps are close but not perfectly aligned."""
    lo = start_ts - BRIDGE_SLICE_PAD_SEC
    hi = end_ts + BRIDGE_SLICE_PAD_SEC
    parts = [arr for ts, arr in buffer if ts + len(arr) / VAD_SAMPLE_RATE >= lo and ts <= hi]
    return np.concatenate(parts) if parts else None


def _drain_discord_segments(timestamped_buffer, model_holder, executor, process_segment_fn):
    """Cuts a segment for every speaking interval the Discord bridge has
    just closed, using its (user_id, username) as a speaker_hint instead of
    the VAD + voice-embedding guess. Loopback-only - Discord audio only
    ever arrives via loopback, never the mic. model_holder: anything with a
    .get() returning the model to transcribe with (fetched fresh per
    segment, so a live model reload is picked up immediately)."""
    for closed in discord_bridge.drain_closed_intervals():
        duration = closed["end"] - closed["start"]
        if duration < MIN_SPEECH_SEC:
            continue
        start = max(closed["start"], closed["end"] - MAX_SEGMENT_SEC)  # cap runaway segments
        segment = _slice_timestamped_buffer(timestamped_buffer, start, closed["end"])
        if segment is None or len(segment) < int(MIN_SPEECH_SEC * VAD_SAMPLE_RATE):
            continue
        speaker_hint = (closed["user_id"], closed["username"])
        executor.submit(process_segment_fn, model_holder.get(), segment, "", speaker_hint)


def _capture_loop_inner(stop_event, audio_q, native_rate, channels, vad_model, model_holder, executor,
                         process_segment_fn, source_tag, self_caption_state=None):
    leftover = np.zeros(0, dtype=np.float32)
    speech_buffer = []
    in_speech = False
    silence_frames = 0
    silence_frames_needed = int(SILENCE_HANGOVER_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)
    min_speech_frames = int(MIN_SPEECH_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)

    is_loopback = source_tag == ""
    timestamped_buffer: "deque" = deque() if is_loopback else None
    bridge_was_active = False  # only meaningful when is_loopback

    while not stop_event.is_set():
        try:
            chunk_ts, raw = audio_q.get(timeout=1)
        except queue.Empty:
            continue

        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        pcm_16k = resample_linear(pcm, native_rate, VAD_SAMPLE_RATE)

        if is_loopback:
            timestamped_buffer.append((chunk_ts, pcm_16k))
            cutoff = chunk_ts - BRIDGE_BUFFER_WINDOW_SEC
            while timestamped_buffer and timestamped_buffer[0][0] < cutoff:
                timestamped_buffer.popleft()

            bridge_active = discord_bridge.is_active()
            if bridge_active != bridge_was_active:
                # Surface bridge connect/disconnect - otherwise there's no way to
                # tell, from the overlay alone, whether loopback segmentation is
                # coming from Discord's own speaking events or the VAD fallback
                # (they're mutually exclusive - see the gate a few lines below).
                broadcast(
                    "[Discord bridge: connected - using Discord speaking events for segmentation]"
                    if bridge_active else
                    "[Discord bridge: disconnected - falling back to VAD segmentation]"
                )
                status.set(
                    "discord_bridge",
                    "Connected (Discord speaking events)" if bridge_active else "Listening (no client - VAD fallback)"
                )
                bridge_was_active = bridge_active
            if bridge_active:
                _drain_discord_segments(timestamped_buffer, model_holder, executor, process_segment_fn)
        else:
            bridge_active = False

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
                    # the Discord bridge, when active, drives loopback
                    # segmentation itself (see _drain_discord_segments) -
                    # still run VAD to keep its state warm for an instant,
                    # glitch-free fallback the moment the bridge drops, but
                    # don't double-emit the same audio through both paths
                    self_captioning_off = source_tag == "mic" and self_caption_state is not None and not self_caption_state.enabled
                    if total_frames - silence_frames >= min_speech_frames and not bridge_active and not self_captioning_off:
                        segment = np.concatenate(speech_buffer)
                        executor.submit(process_segment_fn, model_holder.get(), segment, source_tag)
                    speech_buffer = []
                    silence_frames = 0
                    # a forced max-duration cut doesn't mean silence started -
                    # keep accumulating the next chunk right away
                    in_speech = not hit_silence_end


def audio_worker(stop_event: threading.Event, model_holder: ModelHolder, process_segment_fn,
                  mic_capture: MicCapture, self_caption_state: "SelfCaptionState"):
    try:
        status.set("vad", "Loading...")
        broadcast("Loading VAD model...")
        vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        status.set("vad", "Ready")

        p = pyaudio.PyAudio()
        executor = ThreadPoolExecutor(max_workers=1)
        # Bind the executor before the initial load, not after - otherwise
        # there's a narrow window where status already says the model is
        # "Ready" (Settings re-enables the model combo off that signal) but
        # request_reload() would still silently no-op since _executor is None.
        model_holder._bind_executor(executor)
        model_holder._load_initial()

        loopback_q: "queue.Queue[bytes]" = queue.Queue()
        loopback_stream, loopback_info = open_loopback_stream(p, make_pyaudio_callback(loopback_q))
        loopback_stream.start_stream()
        t = threading.Thread(
            target=_capture_loop,
            args=(stop_event, loopback_q, int(loopback_info["defaultSampleRate"]),
                  loopback_info["maxInputChannels"], vad_model, model_holder, executor, process_segment_fn, ""),
            daemon=True,
        )
        t.start()

        mic_capture.bind(p, model_holder, executor, process_segment_fn, self_caption_state)

        broadcast("Listening...")

        while not stop_event.is_set():
            time.sleep(0.2)

        mic_capture.shutdown()
        loopback_stream.stop_stream()
        loopback_stream.close()
        p.terminate()
    except Exception as e:
        broadcast(f"[ERROR] {type(e).__name__}: {e}")


class OverlayApp:
    def __init__(self, root, stop_event: threading.Event, title: str = "Live Captions", anchor: str = "bottom",
                 offset_index: int = 0, log_file=None, caption_queue: "queue.Queue" = None, on_close=None,
                 on_toggle_translation=None, self_caption_state: "SelfCaptionState" = None):
        self.root = root
        self.stop_event = stop_event
        self.entry_marks = deque()
        self._mark_seq = 0
        self.log_file = log_file
        self.caption_queue = caption_queue if caption_queue is not None else queue.Queue()
        self.on_close = on_close
        self.on_toggle_translation = on_toggle_translation
        self.translation_on = True
        self.self_caption_state = self_caption_state
        self.anchor = anchor

        root.overrideredirect(True)          # borderless
        root.attributes("-topmost", True)    # always on top
        root.attributes("-alpha", 0.85)      # slight transparency
        root.configure(bg="black")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        self.win_w, self.win_h = int(screen_w * WINDOW_W_FRAC), WINDOW_H
        x, y = self._geometry_for(anchor, offset_index)
        root.geometry(f"{self.win_w}x{self.win_h}+{x}+{y}")
        root.pack_propagate(False)

        self.minimized = False
        self._restore_geo = root.geometry()

        # Full caption view
        self.content_frame = tk.Frame(root, bg="black")
        self.content_frame.pack(fill="both", expand=True)

        # Title bar: drag handle + minimize/close (the text area below needs
        # click-drag free for text selection/scrolling, so dragging lives here)
        title_bar = tk.Frame(self.content_frame, bg="#1c1c1c", height=24)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        drag_label = tk.Label(title_bar, text=title, fg="gray", bg="#1c1c1c", font=("Segoe UI", 9))
        drag_label.pack(side="left", padx=8)

        close_btn = tk.Label(title_bar, text="✕", fg="white", bg="#1c1c1c", font=("Segoe UI", 10))
        close_btn.pack(side="right", padx=(0, 8))
        close_btn.bind("<Button-1>", lambda e: self.close())

        minimize_btn = tk.Label(title_bar, text="_", fg="white", bg="#1c1c1c", font=("Segoe UI", 11, "bold"))
        minimize_btn.pack(side="right", padx=(0, 4))
        minimize_btn.bind("<Button-1>", lambda e: self.minimize())

        self.translate_btn = tk.Label(title_bar, text="T", fg="white", bg="#1c1c1c", font=("Segoe UI", 10, "bold"))
        self.translate_btn.pack(side="right", padx=(0, 4))
        self.translate_btn.bind("<Button-1>", lambda e: self.toggle_translation())

        if self.self_caption_state is not None:
            self_caption_btn = tk.Label(title_bar, text="Me", fg="white", bg="#1c1c1c", font=("Segoe UI", 9, "bold"))
            self_caption_btn.pack(side="right", padx=(0, 4))
            self_caption_btn.bind("<Button-1>", lambda e: self.self_caption_state.toggle())
            self.self_caption_state.register(self_caption_btn)

        # Scrollable caption log
        text_container = tk.Frame(self.content_frame, bg="black")
        text_container.pack(fill="both", expand=True, padx=(15, 0), pady=(6, 10))

        scrollbar = tk.Scrollbar(text_container)
        scrollbar.pack(side="right", fill="y")

        self.text = tk.Text(
            text_container,
            fg="white",
            bg="black",
            font=("Microsoft YaHei UI", 13),
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            yscrollcommand=scrollbar.set,
            state="disabled",
            padx=5,
            pady=5,
        )
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text.yview)

        # speaker: who said it, bold accent. meta: "[lang, Xs]", dim/small.
        # native: the transcription itself, the main readable content.
        # translated: the "-> ..." line, a distinct warm color so it never
        # reads as a continuation of the original. status: system/loading
        # messages, dim/italic so they visually recede from real captions.
        self.text.tag_config("speaker", foreground="#7ec4ff", font=("Microsoft YaHei UI", 13, "bold"))
        self.text.tag_config("meta", foreground="#808080", font=("Microsoft YaHei UI", 11))
        self.text.tag_config("native", foreground="#f2f2f2", font=("Microsoft YaHei UI", 13))
        self.text.tag_config("translated", foreground="#ffcf6e", font=("Microsoft YaHei UI", 13, "italic"))
        self.text.tag_config("status", foreground="#808080", font=("Microsoft YaHei UI", 11, "italic"))

        resize_grip = tk.Label(self.content_frame, text="◲", fg="gray", bg="black", font=("Segoe UI", 10))
        resize_grip.place(relx=1.0, rely=1.0, x=-4, y=-4, anchor="se")
        resize_grip.config(cursor="size_nw_se")
        resize_grip.bind("<ButtonPress-1>", self._start_resize)
        resize_grip.bind("<B1-Motion>", self._do_resize)

        # Collapsed view - small draggable tab shown when minimized
        self.collapsed_frame = tk.Frame(root, bg="black")
        restore_btn = tk.Label(
            self.collapsed_frame, text=f"▲ {title}", fg="white", bg="black", font=("Segoe UI", 10)
        )
        restore_btn.pack(fill="both", expand=True)
        restore_btn.bind("<Button-1>", lambda e: self.restore())

        # drag to move (add="+" so this doesn't clobber the click handlers above)
        for widget in (title_bar, drag_label, restore_btn):
            widget.bind("<ButtonPress-1>", self._start_drag, add="+")
            widget.bind("<B1-Motion>", self._do_drag, add="+")

        root.bind("<Escape>", lambda e: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.root.after(100, self.poll_queue)

    def _geometry_for(self, anchor, offset_index):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - self.win_w) // 2
        stack_offset = offset_index * (self.win_h + STACK_GAP)
        y = ANCHOR_MARGIN + stack_offset if anchor == "top" else screen_h - self.win_h - ANCHOR_MARGIN - stack_offset
        return x, y

    def reposition(self, anchor, offset_index):
        """Recomputes this window's position for a new anchor/stack slot -
        called by _relayout_windows() whenever a window opens/closes or the
        shared position setting changes. No-op visually while minimized;
        the new geometry is remembered and applied on restore()."""
        self.anchor = anchor
        x, y = self._geometry_for(anchor, offset_index)
        geo = f"{self.win_w}x{self.win_h}+{x}+{y}"
        if self.minimized:
            self._restore_geo = geo
        else:
            self._restore_geo = geo
            self.root.geometry(geo)

    def minimize(self):
        if self.minimized:
            return
        self.minimized = True
        self._restore_geo = self.root.geometry()
        self.content_frame.pack_forget()
        self.collapsed_frame.pack(fill="both", expand=True)
        x, y = self.root.winfo_x(), self.root.winfo_y()
        self.root.geometry(f"140x30+{x}+{y}")

    def restore(self):
        if not self.minimized:
            return
        self.minimized = False
        self.collapsed_frame.pack_forget()
        self.content_frame.pack(fill="both", expand=True)
        self.root.geometry(self._restore_geo)

    def toggle_translation(self):
        self.translation_on = not self.translation_on
        self.translate_btn.config(fg="white" if self.translation_on else "#555555")
        if self.on_toggle_translation:
            self.on_toggle_translation(self.translation_on)

    def _start_resize(self, event):
        self._resize_start_x = event.x_root
        self._resize_start_y = event.y_root
        self._resize_start_w = self.root.winfo_width()
        self._resize_start_h = self.root.winfo_height()

    def _do_resize(self, event):
        dx = event.x_root - self._resize_start_x
        dy = event.y_root - self._resize_start_y
        new_w = max(250, self._resize_start_w + dx)
        new_h = max(100, self._resize_start_h + dy)
        x, y = self.root.winfo_x(), self.root.winfo_y()
        self.root.geometry(f"{new_w}x{new_h}+{x}+{y}")
        self.win_w = new_w

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_drag(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def poll_queue(self):
        new_items = []
        while True:
            try:
                new_items.append(self.caption_queue.get_nowait())
            except queue.Empty:
                break

        if new_items:
            if self.log_file:
                for item in new_items:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    text = self._plain_text(item) if isinstance(item, dict) else item
                    self.log_file.write(f"[{timestamp}] {text}\n")
                self.log_file.flush()

            at_bottom = self.text.yview()[1] >= 0.999

            self.text.config(state="normal")
            for item in new_items:
                if self.text.index("end-1c") != "1.0":
                    self.text.insert("end", "\n\n")
                mark = f"e{self._mark_seq}"
                self._mark_seq += 1
                self.text.mark_set(mark, "end-1c")
                self.entry_marks.append(mark)
                if isinstance(item, dict):
                    self._insert_caption(item)
                else:
                    self.text.insert("end", item, "status")

            # trim only the oldest entry's own text range - never rebuild the
            # whole widget, or the scrollbar snaps back to the top on every update
            while len(self.entry_marks) > MAX_HISTORY:
                old_mark = self.entry_marks.popleft()
                end_bound = self.entry_marks[0] if self.entry_marks else "end"
                self.text.delete(old_mark, end_bound)
                self.text.mark_unset(old_mark)

            self.text.config(state="disabled")

            if at_bottom:
                self.text.see("end")

        self.root.after(100, self.poll_queue)

    def _insert_caption(self, item):
        if item.get("speaker"):
            self.text.insert("end", item["speaker"] + " ", "speaker")
        self.text.insert("end", item["meta"] + " ", "meta")
        self.text.insert("end", item["native"], "native")
        if item.get("translated"):
            self.text.insert("end", "\n    -> ", "meta")
            self.text.insert("end", item["translated"], "translated")

    @staticmethod
    def _plain_text(item):
        parts = [p for p in (item.get("speaker"), item["meta"]) if p]
        text = " ".join(parts) + " " + item["native"]
        if item.get("translated"):
            text += f"\n    -> {item['translated']}"
        return text

    def close(self):
        if self.log_file:
            self.log_file.close()
        self.root.destroy()
        if self.on_close:
            self.on_close()


class PipelineController:
    """Bundles the live handles the Settings window needs to reconfigure the
    running pipeline - constructed once by run_translator."""

    def __init__(self, targets, add_window, remove_window, set_position, model_holder,
                 mic_capture, self_caption_state, set_bridge_port, initial_bridge_port,
                 initial_position, shutdown):
        self.targets = targets  # the live list - read for initial checkbox state
        self.add_window = add_window
        self.remove_window = remove_window
        self.set_position = set_position
        self.model_holder = model_holder
        self.mic_capture = mic_capture
        self.self_caption_state = self_caption_state
        self.set_bridge_port = set_bridge_port
        self.initial_bridge_port = initial_bridge_port
        self.initial_position = initial_position
        self.shutdown = shutdown


def run_translator(make_process_segment_fn, initial_targets, model_size="large-v3", device="cuda",
                    compute_type="float16", capture_mic=False, discord_bridge_port=discord_bridge.DEFAULT_PORT,
                    show_settings=True, initial_position=None):
    """make_process_segment_fn(targets, caption_queues, translation_enabled) -> process_segment:
    called once with the shared, live-mutable state objects this function
    owns, so add_window/remove_window (driven by the Settings window) just
    mutate those same objects and the pipeline picks the change up on its
    next segment - no restart needed.
    initial_targets: target language codes to open overlay windows for
    immediately (e.g. from --target); show_settings: whether the Settings
    window starts visible (no --target given) or hidden/tray-only (scripted
    launch)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.makedirs(LOG_DIR, exist_ok=True)

    targets: "list[str]" = []
    caption_queues: "dict[str, queue.Queue]" = {}
    translation_enabled: "dict[str, bool]" = {}
    process_segment_fn = make_process_segment_fn(targets, caption_queues, translation_enabled)

    stop_event = threading.Event()
    root = tk.Tk()
    root.withdraw()  # hidden - only real purpose is to own the shared mainloop

    self_caption_state = SelfCaptionState()
    model_holder = ModelHolder(model_size, device=device, compute_type=compute_type)
    mic_capture = MicCapture(initially_on=capture_mic)

    windows: "dict[str, OverlayApp]" = {}
    position_override = {"value": initial_position}  # None = auto (bottom for en, top otherwise)
    settings_window_ref = {"window": None}  # set once SettingsWindow is constructed, below

    def _default_anchor(target):
        return position_override["value"] or ("bottom" if target == "en" else "top")

    def _relayout_windows():
        anchor_counts: dict = {}
        for target in targets:
            win = windows.get(target)
            if win is None:
                continue
            anchor = _default_anchor(target)
            offset_index = anchor_counts.get(anchor, 0)
            anchor_counts[anchor] = offset_index + 1
            win.reposition(anchor, offset_index)

    def _refresh_broadcast_targets():
        queues = list(caption_queues.values())
        sw = settings_window_ref["window"]
        if sw is not None:
            queues.append(sw.status_queue)
        register_broadcast_queues(queues)

    def _on_window_closed(target):
        windows.pop(target, None)
        if target in targets:
            targets.remove(target)
        caption_queues.pop(target, None)
        translation_enabled.pop(target, None)
        _relayout_windows()
        _refresh_broadcast_targets()
        sw = settings_window_ref["window"]
        if sw is not None:
            sw.mark_target_unchecked(target)

    def add_window(target, title):
        if target in windows:
            return
        targets.append(target)
        caption_queues[target] = queue.Queue()
        translation_enabled[target] = True

        safe_title = "".join(c if c.isalnum() else "_" for c in title).strip("_")
        log_path = os.path.join(LOG_DIR, f"{safe_title}_{datetime.now():%Y%m%d_%H%M%S}.log")
        log_file = open(log_path, "a", encoding="utf-8")
        caption_queues[target].put(f"[Logging to: {log_path}]")

        win = tk.Toplevel(root)
        overlay_app = OverlayApp(
            win, stop_event, title=title, anchor=_default_anchor(target), offset_index=0,
            log_file=log_file, caption_queue=caption_queues[target],
            on_close=lambda target=target: _on_window_closed(target),
            on_toggle_translation=lambda on, target=target: translation_enabled.__setitem__(target, on),
            self_caption_state=self_caption_state,
        )
        windows[target] = overlay_app
        _relayout_windows()
        _refresh_broadcast_targets()

    def remove_window(target):
        win = windows.get(target)
        if win is not None:
            win.close()  # triggers _on_window_closed via on_close, above

    def set_position(pos):
        position_override["value"] = pos
        _relayout_windows()

    def set_bridge_port(new_port):
        # discord_bridge.stop() joins its server thread (up to a few seconds) -
        # run off the Tk thread so the Apply button doesn't freeze the UI.
        def _apply():
            discord_bridge.stop()
            discord_bridge.start(port=new_port)
            status.set("discord_bridge", f"Listening on port {new_port} (no client - VAD fallback)")
        threading.Thread(target=_apply, daemon=True).start()

    tray_icon = {"icon": None}

    def shutdown():
        if stop_event.is_set():
            return
        stop_event.set()
        discord_bridge.stop()
        if tray_icon["icon"] is not None:
            tray_icon["icon"].stop()
        root.destroy()

    for target in initial_targets:
        add_window(target, f"Live Captions ({target.upper()})")

    # --- Settings window: the app's persistent, taskbar-visible main window ---
    # Built before the worker thread starts, so its status area catches the
    # "Loading VAD model..." / "Loading Whisper model..." broadcasts too.
    settings_root = tk.Toplevel(root)
    controller = PipelineController(
        targets=targets, add_window=add_window, remove_window=remove_window, set_position=set_position,
        model_holder=model_holder, mic_capture=mic_capture, self_caption_state=self_caption_state,
        set_bridge_port=set_bridge_port, initial_bridge_port=discord_bridge_port,
        initial_position=initial_position, shutdown=lambda: root.after(0, shutdown),
    )
    from launcher import SettingsWindow  # local import - avoids a circular import at module load time
    settings_window_ref["window"] = SettingsWindow(settings_root, controller)
    _refresh_broadcast_targets()

    settings_root.title("vc-translator")
    settings_root.protocol("WM_DELETE_WINDOW", shutdown)

    # pystray's callbacks run on its own background thread, not the Tk
    # thread - calling Tkinter methods (even .after()) directly from there
    # is unsafe in Tcl/Tk and can hang the whole UI. Route through a plain
    # queue.Queue (genuinely thread-safe) that the Tk thread itself polls,
    # same pattern as every other cross-thread update in this file.
    tray_commands: "queue.Queue" = queue.Queue()

    def _poll_tray_commands():
        while True:
            try:
                cmd = tray_commands.get_nowait()
            except queue.Empty:
                break
            if cmd == "show":
                settings_root.deiconify()
                settings_root.lift()
                settings_root.focus_force()
            elif cmd == "quit":
                shutdown()
        root.after(100, _poll_tray_commands)

    _poll_tray_commands()

    # Tray is disabled for now: confirmed via isolation testing that
    # pystray's native Win32 message-loop thread (started by
    # icon.run_detached()) segfaults against Tkinter's own message loop in
    # this environment - random-location crashes within ~10s of startup,
    # gone entirely with tray.start() not called. A genuine native crash,
    # not something a try/except can catch. Falls back to normal
    # OS-taskbar minimize (the `else` branch below) until a safer tray
    # integration is found - see tray.py for the (currently unused) client.
    icon = None
    tray_icon["icon"] = icon

    if icon is not None:
        def _on_settings_unmap(event):
            if event.widget is settings_root and settings_root.state() == "iconic":
                settings_root.withdraw()
        settings_root.bind("<Unmap>", _on_settings_unmap)
    else:
        broadcast("[Tray icon disabled (unstable in this environment) - minimize behaves like a normal window]")

    if show_settings:
        settings_root.deiconify()
    else:
        settings_root.withdraw()

    worker = threading.Thread(
        target=audio_worker,
        args=(stop_event, model_holder, process_segment_fn, mic_capture, self_caption_state),
        daemon=True,
    )
    worker.start()

    discord_bridge.start(port=discord_bridge_port)
    status.set("discord_bridge", f"Listening on port {discord_bridge_port} (no client - VAD fallback)")

    root.mainloop()
