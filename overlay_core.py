"""
Shared audio-capture, VAD segmentation, and overlay-window plumbing used by
translate_vc.py. A single process_segment_fn and audio_worker are shared
across every requested target language; run_translator just opens one
overlay window (with its own caption queue and log file) per target.
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

_broadcast_queues: "list[queue.Queue]" = []


def register_broadcast_queues(queues):
    _broadcast_queues[:] = queues


def broadcast(msg):
    for q in _broadcast_queues:
        q.put(msg)


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


def _capture_loop(stop_event, audio_q, native_rate, channels, vad_model, whisper_model, executor,
                   process_segment_fn, source_tag):
    try:
        _capture_loop_inner(stop_event, audio_q, native_rate, channels, vad_model, whisper_model, executor,
                             process_segment_fn, source_tag)
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


def _drain_discord_segments(timestamped_buffer, whisper_model, executor, process_segment_fn):
    """Cuts a segment for every speaking interval the Discord bridge has
    just closed, using its (user_id, username) as a speaker_hint instead of
    the VAD + voice-embedding guess. Loopback-only - Discord audio only
    ever arrives via loopback, never the mic."""
    for closed in discord_bridge.drain_closed_intervals():
        duration = closed["end"] - closed["start"]
        if duration < MIN_SPEECH_SEC:
            continue
        start = max(closed["start"], closed["end"] - MAX_SEGMENT_SEC)  # cap runaway segments
        segment = _slice_timestamped_buffer(timestamped_buffer, start, closed["end"])
        if segment is None or len(segment) < int(MIN_SPEECH_SEC * VAD_SAMPLE_RATE):
            continue
        speaker_hint = (closed["user_id"], closed["username"])
        executor.submit(process_segment_fn, whisper_model, segment, "", speaker_hint)


def _capture_loop_inner(stop_event, audio_q, native_rate, channels, vad_model, whisper_model, executor,
                         process_segment_fn, source_tag):
    leftover = np.zeros(0, dtype=np.float32)
    speech_buffer = []
    in_speech = False
    silence_frames = 0
    silence_frames_needed = int(SILENCE_HANGOVER_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)
    min_speech_frames = int(MIN_SPEECH_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)

    is_loopback = source_tag == ""
    timestamped_buffer: "deque" = deque() if is_loopback else None

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
            if bridge_active:
                _drain_discord_segments(timestamped_buffer, whisper_model, executor, process_segment_fn)
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
                    if total_frames - silence_frames >= min_speech_frames and not bridge_active:
                        segment = np.concatenate(speech_buffer)
                        executor.submit(process_segment_fn, whisper_model, segment, source_tag)
                    speech_buffer = []
                    silence_frames = 0
                    # a forced max-duration cut doesn't mean silence started -
                    # keep accumulating the next chunk right away
                    in_speech = not hit_silence_end


def audio_worker(stop_event: threading.Event, model_size: str, device: str, compute_type: str, process_segment_fn,
                  capture_mic: bool = False):
    try:
        broadcast("Loading VAD model...")
        vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

        broadcast(f"Loading Whisper model ({model_size}) on {device}...")
        whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)

        p = pyaudio.PyAudio()
        executor = ThreadPoolExecutor(max_workers=1)
        streams = []
        capture_threads = []

        loopback_q: "queue.Queue[bytes]" = queue.Queue()
        loopback_stream, loopback_info = open_loopback_stream(p, make_pyaudio_callback(loopback_q))
        loopback_stream.start_stream()
        streams.append(loopback_stream)
        t = threading.Thread(
            target=_capture_loop,
            args=(stop_event, loopback_q, int(loopback_info["defaultSampleRate"]),
                  loopback_info["maxInputChannels"], vad_model, whisper_model, executor, process_segment_fn, ""),
            daemon=True,
        )
        t.start()
        capture_threads.append(t)

        if capture_mic:
            # Silero VAD is a stateful streaming model (it carries hidden state
            # between calls) - sharing one instance across two concurrently
            # running capture threads corrupts that state. Each source gets
            # its own instance instead.
            mic_vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

            mic_q: "queue.Queue[bytes]" = queue.Queue()
            mic_stream, mic_info, mic_channels = open_mic_stream(p, make_pyaudio_callback(mic_q))
            mic_stream.start_stream()
            streams.append(mic_stream)
            t2 = threading.Thread(
                target=_capture_loop,
                args=(stop_event, mic_q, int(mic_info["defaultSampleRate"]), mic_channels,
                      mic_vad_model, whisper_model, executor, process_segment_fn, "mic"),
                daemon=True,
            )
            t2.start()
            capture_threads.append(t2)

        broadcast("Listening...")

        while not stop_event.is_set():
            time.sleep(0.2)

        for stream in streams:
            stream.stop_stream()
            stream.close()
        p.terminate()
    except Exception as e:
        broadcast(f"[ERROR] {type(e).__name__}: {e}")


class OverlayApp:
    def __init__(self, root, stop_event: threading.Event, title: str = "Live Captions", anchor: str = "bottom",
                 offset_index: int = 0, log_file=None, caption_queue: "queue.Queue" = None, on_close=None):
        self.root = root
        self.stop_event = stop_event
        self.entry_marks = deque()
        self._mark_seq = 0
        self.log_file = log_file
        self.caption_queue = caption_queue if caption_queue is not None else queue.Queue()
        self.on_close = on_close

        root.overrideredirect(True)          # borderless
        root.attributes("-topmost", True)    # always on top
        root.attributes("-alpha", 0.85)      # slight transparency
        root.configure(bg="black")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        self.win_w, self.win_h = int(screen_w * 0.6), 260
        x = (screen_w - self.win_w) // 2
        stack_offset = offset_index * (self.win_h + 20)
        y = 80 + stack_offset if anchor == "top" else screen_h - self.win_h - 80 - stack_offset
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
                    self.log_file.write(f"[{timestamp}] {item}\n")
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
                self.text.insert("end", item)

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

    def close(self):
        if self.log_file:
            self.log_file.close()
        self.root.destroy()
        if self.on_close:
            self.on_close()


def run_translator(process_segment_fn, window_specs, model_size="large-v3", device="cuda", compute_type="float16",
                    capture_mic=False, discord_bridge_port=discord_bridge.DEFAULT_PORT):
    """window_specs: list of {"title", "anchor", "queue"} dicts, one per overlay
    window - all sharing a single audio_worker (capture + VAD + Whisper)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.makedirs(LOG_DIR, exist_ok=True)
    register_broadcast_queues([spec["queue"] for spec in window_specs])
    discord_bridge.start(port=discord_bridge_port)

    stop_event = threading.Event()
    root = tk.Tk()
    root.withdraw()  # hidden - only real purpose is to own the shared mainloop

    remaining = set(range(len(window_specs)))
    anchor_counts: dict = {}

    for i, spec in enumerate(window_specs):
        anchor = spec["anchor"]
        offset_index = anchor_counts.get(anchor, 0)
        anchor_counts[anchor] = offset_index + 1

        title = spec["title"]
        safe_title = "".join(c if c.isalnum() else "_" for c in title).strip("_")
        log_path = os.path.join(LOG_DIR, f"{safe_title}_{datetime.now():%Y%m%d_%H%M%S}.log")
        log_file = open(log_path, "a", encoding="utf-8")
        spec["queue"].put(f"[Logging to: {log_path}]")

        def on_close(i=i):
            remaining.discard(i)
            if not remaining:
                stop_event.set()
                root.quit()

        win = tk.Toplevel(root)
        OverlayApp(
            win, stop_event, title=title, anchor=anchor, offset_index=offset_index,
            log_file=log_file, caption_queue=spec["queue"], on_close=on_close,
        )

    worker = threading.Thread(
        target=audio_worker,
        args=(stop_event, model_size, device, compute_type, process_segment_fn, capture_mic),
        daemon=True,
    )
    worker.start()

    root.mainloop()
