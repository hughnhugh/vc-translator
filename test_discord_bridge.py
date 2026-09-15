"""Standalone sanity check for discord_bridge.py - starts the server, sends
sample events over a real WebSocket connection, and checks the query API."""

import asyncio
import json
import time

import websockets

import discord_bridge


async def main():
    discord_bridge.start(port=8765)
    await asyncio.sleep(0.3)  # let the server thread come up

    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(json.dumps({
            "event": "channel_sync",
            "members": {"111": "Alice", "222": "Bob"},
        }))

        t0 = time.time()
        await ws.send(json.dumps({"event": "speaking_start", "userId": "111", "username": "Alice", "ts": t0 * 1000}))
        await asyncio.sleep(0.5)
        t1 = time.time()
        await ws.send(json.dumps({"event": "speaking_stop", "userId": "111", "username": "Alice", "ts": t1 * 1000}))

        await asyncio.sleep(0.3)  # let the server process before we query

    assert discord_bridge.is_active(), "expected bridge to be active after recent messages"

    result = discord_bridge.speaker_during(t0, t1)
    print("speaker_during(t0, t1) ->", result)
    assert result is not None, "expected an overlapping speaker"
    assert result[0] == "111" and result[1] == "Alice", f"unexpected result: {result}"

    no_overlap = discord_bridge.speaker_during(t0 - 100, t0 - 50)
    print("speaker_during(no overlap) ->", no_overlap)
    assert no_overlap is None

    closed = discord_bridge.drain_closed_intervals()
    print("drain_closed_intervals() ->", closed)
    assert len(closed) == 1 and closed[0]["user_id"] == "111" and closed[0]["username"] == "Alice"
    assert discord_bridge.drain_closed_intervals() == []  # drained already

    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
