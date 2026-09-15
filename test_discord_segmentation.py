"""Verifies overlay_core's Discord-bridge-driven segmentation
(_slice_timestamped_buffer / _drain_discord_segments) in isolation, driven
through discord_bridge's real message-handling path, with synthetic
timestamped audio - no GPU/Whisper/real Discord call needed."""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import discord_bridge
import overlay_core

SR = overlay_core.VAD_SAMPLE_RATE


def _make_buffer(seconds, chunk_sec=0.1, start_ts=None):
    start_ts = start_ts if start_ts is not None else time.time()
    chunk_len = int(chunk_sec * SR)
    n = int(seconds / chunk_sec)
    buf = deque()
    for i in range(n):
        buf.append((start_ts + i * chunk_sec, np.zeros(chunk_len, dtype=np.float32)))
    return buf, start_ts


def _run_drain(buffer):
    calls = []
    executor = ThreadPoolExecutor(max_workers=1)
    overlay_core._drain_discord_segments(buffer, "FAKE_MODEL", executor, lambda *a: calls.append(a))
    executor.shutdown(wait=True)
    return calls


def test_speaking_interval_cuts_expected_segment():
    buffer, t0 = _make_buffer(seconds=2.0)
    start_ts, end_ts = t0 + 0.3, t0 + 1.3  # 1s interval inside the buffered window

    discord_bridge._handle_message({"event": "channel_sync", "members": {"42": "Zaphod"}})
    discord_bridge._handle_message({"event": "speaking_start", "userId": "42", "ts": start_ts * 1000})
    discord_bridge._handle_message({"event": "speaking_stop", "userId": "42", "ts": end_ts * 1000})

    calls = _run_drain(buffer)
    assert len(calls) == 1, f"expected exactly one segment cut, got {calls}"
    model, segment, source_tag, speaker_hint = calls[0]
    assert model == "FAKE_MODEL"
    assert source_tag == "", f"expected loopback source_tag '', got {source_tag!r}"
    assert speaker_hint == ("42", "Zaphod"), f"unexpected speaker_hint: {speaker_hint}"

    expected_len = int((end_ts - start_ts) * SR)
    pad_len = int(2 * overlay_core.BRIDGE_SLICE_PAD_SEC * SR)
    assert expected_len <= len(segment) <= expected_len + pad_len + int(0.2 * SR), (
        f"segment length {len(segment)} outside expected range around {expected_len}"
    )
    print(f"OK: {end_ts - start_ts:.1f}s interval -> {len(segment)} samples (~{len(segment)/SR:.2f}s)")


def test_short_interval_is_dropped():
    buffer, t0 = _make_buffer(seconds=1.0)
    discord_bridge._handle_message({"event": "speaking_start", "userId": "1", "username": "Blip", "ts": t0 * 1000})
    discord_bridge._handle_message({"event": "speaking_stop", "userId": "1", "ts": (t0 + 0.1) * 1000})  # < MIN_SPEECH_SEC

    calls = _run_drain(buffer)
    assert calls == [], f"expected sub-MIN_SPEECH_SEC interval to be dropped, got {calls}"
    print("OK: short interval correctly dropped")


def test_runaway_interval_is_capped():
    buffer, t0 = _make_buffer(seconds=12.0)
    end_ts = t0 + 12.0
    discord_bridge._handle_message({"event": "speaking_start", "userId": "7", "username": "Marvin", "ts": t0 * 1000})
    discord_bridge._handle_message({"event": "speaking_stop", "userId": "7", "ts": end_ts * 1000})

    calls = _run_drain(buffer)
    assert len(calls) == 1
    segment = calls[0][1]
    cap_len = int(overlay_core.MAX_SEGMENT_SEC * SR)
    pad_len = int(2 * overlay_core.BRIDGE_SLICE_PAD_SEC * SR)
    assert len(segment) <= cap_len + pad_len + int(0.2 * SR), (
        f"segment of {len(segment)} samples wasn't capped near MAX_SEGMENT_SEC ({cap_len} samples)"
    )
    print(f"OK: 12s interval capped to {len(segment)/SR:.2f}s (MAX_SEGMENT_SEC={overlay_core.MAX_SEGMENT_SEC}s)")


if __name__ == "__main__":
    test_speaking_interval_cuts_expected_segment()
    test_short_interval_is_dropped()
    test_runaway_interval_is_capped()
    print("ALL OK")
