"""
Local WebSocket bridge that receives Discord speaking-activity events from
the companion Vencord plugin (vencord-plugin/discordSpeakingBridge.ts) and
exposes ground-truth "who was speaking when" to overlay_core.py's loopback
capture loop - used in place of the VAD + voice-embedding guess whenever
the plugin is connected and actively reporting.

Entirely optional: if nothing ever connects, is_active() just stays False
forever and callers fall back to their existing behavior.
"""

import asyncio
import json
import queue
import threading
import time

from websockets.asyncio.server import serve

DEFAULT_PORT = 8765
ACTIVE_TIMEOUT_SEC = 10   # no messages for this long -> treat the bridge as inactive
INTERVAL_WINDOW_SEC = 30  # how far back to retain speaking intervals

_lock = threading.Lock()
_names: dict = {}
_intervals: list = []     # [{"user_id", "start", "end"}], end=None while still open
_last_message_at = 0.0
_connected_clients = 0    # actual WebSocket connection count, not activity-based

# Segments to cut, one per closed (start, end) speaking interval - this is
# what actually drives event-driven segmentation in overlay_core.py, rather
# than making the capture loop reverse-map already-VAD-cut audio onto
# speaker_during() after the fact.
_closed_intervals_q: "queue.Queue" = queue.Queue()


def _prune_locked():
    global _intervals
    now = time.time()
    cutoff = now - INTERVAL_WINDOW_SEC
    _intervals = [iv for iv in _intervals if (iv["end"] if iv["end"] is not None else now) >= cutoff]


def _handle_message(msg):
    global _last_message_at
    if not isinstance(msg, dict):
        return
    event = msg.get("event")

    with _lock:
        _last_message_at = time.time()

        if event == "channel_sync":
            for user_id, username in (msg.get("members") or {}).items():
                _names[user_id] = username
            return

        user_id = msg.get("userId")
        if user_id is None:
            return
        username = msg.get("username")
        if username:
            _names[user_id] = username
        ts = (msg.get("ts") or time.time() * 1000) / 1000.0

        if event == "speaking_start":
            # close any stale still-open interval for this user first, in
            # case a stop event was ever dropped
            for iv in _intervals:
                if iv["user_id"] == user_id and iv["end"] is None:
                    iv["end"] = ts
            _intervals.append({"user_id": user_id, "start": ts, "end": None})
        elif event == "speaking_stop":
            for iv in reversed(_intervals):
                if iv["user_id"] == user_id and iv["end"] is None:
                    iv["end"] = ts
                    _closed_intervals_q.put({
                        "user_id": user_id,
                        "username": _names.get(user_id, f"User_{user_id}"),
                        "start": iv["start"],
                        "end": ts,
                    })
                    break

        _prune_locked()


async def _handler(websocket):
    global _connected_clients
    with _lock:
        _connected_clients += 1
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            _handle_message(msg)
    finally:
        with _lock:
            _connected_clients -= 1


async def _serve_forever(port, stop_event):
    async with serve(_handler, "127.0.0.1", port):
        await stop_event.wait()


_loop = None
_server_stop_event = None
_server_thread = None


def start(port: int = DEFAULT_PORT):
    """Starts the bridge server on a background daemon thread. Safe to call
    even if nothing ever connects - the rest of the app just keeps seeing
    is_active() == False. If a server from a previous start() is still
    running, call stop() first (e.g. to switch ports live)."""
    global _loop, _server_stop_event, _server_thread

    ready = threading.Event()

    def _run():
        global _loop, _server_stop_event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop
        _server_stop_event = asyncio.Event()
        ready.set()
        try:
            loop.run_until_complete(_serve_forever(port, _server_stop_event))
        except Exception as e:
            print(f"[discord-bridge] server stopped: {type(e).__name__}: {e}")
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait(timeout=5)
    _server_thread = t
    return t


def stop(timeout: float = 5.0):
    """Stops a server started by start(), if any - e.g. right before
    restarting on a different port. Safe to call even if nothing's running."""
    global _loop, _server_stop_event, _server_thread, _connected_clients
    if _loop is None or _server_stop_event is None:
        return
    loop, stop_event, thread = _loop, _server_stop_event, _server_thread
    loop.call_soon_threadsafe(stop_event.set)
    if thread is not None:
        thread.join(timeout=timeout)
    _loop = None
    _server_stop_event = None
    _server_thread = None
    with _lock:
        _connected_clients = 0


def drain_closed_intervals():
    """Every speaking interval that's closed (a speaking_stop arrived) since
    the last call - each one is a segment overlay_core.py should cut and
    transcribe, tagged with (user_id, username)."""
    items = []
    while True:
        try:
            items.append(_closed_intervals_q.get_nowait())
        except queue.Empty:
            break
    return items


def is_active() -> bool:
    """True if a message has arrived recently - used to gate segmentation
    (see overlay_core._capture_loop_inner): trust the bridge's own
    speaking-event cuts only while it's recently said something, falling
    back to VAD otherwise. This is an *activity* signal, not a connection
    one - it flips constantly during normal conversation (anyone can go
    quiet for 10+ seconds), so don't use it for a user-facing "connected"
    status - see is_connected() for that."""
    with _lock:
        return (time.time() - _last_message_at) < ACTIVE_TIMEOUT_SEC


def is_connected() -> bool:
    """True if the Vencord plugin's WebSocket is actually connected right
    now, regardless of whether anyone's spoken recently."""
    with _lock:
        return _connected_clients > 0


def speaker_during(start_ts: float, end_ts: float):
    """(user_id, username) for whoever's speaking interval overlaps
    [start_ts, end_ts] the most, or None if nothing overlaps at all."""
    with _lock:
        now = time.time()
        best_user, best_overlap = None, 0.0
        for iv in _intervals:
            iv_end = iv["end"] if iv["end"] is not None else now
            overlap = min(end_ts, iv_end) - max(start_ts, iv["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_user = iv["user_id"]
        if best_user is None:
            return None
        return best_user, _names.get(best_user, f"User_{best_user}")
