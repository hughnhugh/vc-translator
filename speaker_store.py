"""
Persistent, self-organizing speaker identification. There's no enrollment
step - every finalized speech segment gets a voice embedding; if it's close
enough to an existing profile it's stored as another example under that
profile, otherwise a new anonymous profile is created (unless the segment
is too short to trust as a new profile's first example - see
MIN_DURATION_FOR_NEW_PROFILE). Rename a profile any time (e.g. by clicking
its label in the overlay) and the same voice keeps getting recognized under
the new name from then on - the ID is derived from the voice, the label is
just whatever you've called it.

Each profile keeps a handful of individual example embeddings rather than
one running average - matching against several real examples directly is
more robust than matching against one blurred average, since an early
short/atypical sample can otherwise drag the average away from how the
person usually sounds. Still tiny: MAX_EXEMPLARS embeddings/profile at a
few hundred bytes each, no raw audio ever stored.
"""

import json
import os
import threading
import time

import msvcrt
import numpy as np

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers.json")
LOCK_PATH = STORE_PATH + ".lock"
# Cosine similarity on L2-normalized ECAPA-TDNN embeddings. Observed
# same-speaker scores in practice ranged ~0.2-0.8 - noisy short segments
# score low even for genuine matches, which is why this also gates new
# profile creation by duration rather than relying on the threshold alone.
SIMILARITY_THRESHOLD = 0.25
MAX_EXEMPLARS = 10
MIN_DURATION_FOR_NEW_PROFILE = 0.5  # seconds - shorter clips can still match, just can't found a new profile

_lock = threading.Lock()


class _cross_process_lock:
    """Exclusive lock over speakers.json shared by every translate_vc.py
    process. threading.Lock alone only serializes calls within one process -
    with two instances running, each could _load() the same snapshot,
    independently decide to create the same new_id, and then whichever
    _save() ran second would silently overwrite the other's profile
    (a lost update, not just a corrupt file)."""

    def __enter__(self):
        self._fh = open(LOCK_PATH, "a+b")
        deadline = time.time() + 30
        while True:
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
                return self
            except OSError:
                if time.time() > deadline:
                    self._fh.close()
                    raise

    def __exit__(self, *exc_info):
        try:
            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._fh.close()


def _load():
    if not os.path.exists(STORE_PATH):
        return {"next_id": 1, "profiles": {}}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Multiple translate_vc.py processes share this one file with no
        # cross-process lock - a bad interleaving of two processes' writes
        # can corrupt it. Recover whatever valid JSON prefix we can rather
        # than crashing every future identify() call in every process.
        data, _ = json.JSONDecoder().raw_decode(raw)
        return data


def _save(data):
    # Unique per-process tmp path so two translate_vc.py instances writing
    # at the same time can't interleave their writes into one shared file.
    tmp_path = f"{STORE_PATH}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STORE_PATH)


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm else v.tolist()


def _cosine_sim(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def identify(embedding, duration_sec):
    """Match embedding against known profiles' individual examples, or
    create a new profile (only if duration_sec is long enough to trust).
    Returns (speaker_id, display_label). speaker_id is None if the segment
    didn't match anyone and was too short to found a new profile."""
    embedding = _normalize(embedding)
    with _lock, _cross_process_lock():
        data = _load()
        best_id, best_sim = None, -1.0
        for spk_id, profile in data["profiles"].items():
            for exemplar in profile["exemplars"]:
                sim = _cosine_sim(embedding, exemplar)
                if sim > best_sim:
                    best_id, best_sim = spk_id, sim

        print(f"[speaker-id] best_sim={best_sim:.3f} vs speaker_{best_id} "
              f"(threshold={SIMILARITY_THRESHOLD}, duration={duration_sec:.2f}s)")

        if best_id is not None and best_sim >= SIMILARITY_THRESHOLD:
            profile = data["profiles"][best_id]
            profile["exemplars"].append(embedding)
            profile["exemplars"] = profile["exemplars"][-MAX_EXEMPLARS:]
            _save(data)
            return best_id, profile["label"] or f"Speaker_{best_id}"

        if duration_sec < MIN_DURATION_FOR_NEW_PROFILE:
            return None, "Unknown"

        new_id = str(data["next_id"])
        data["next_id"] += 1
        data["profiles"][new_id] = {"exemplars": [embedding], "label": None}
        _save(data)
        return new_id, f"Speaker_{new_id}"


def rename(speaker_id, new_label):
    with _lock, _cross_process_lock():
        data = _load()
        if speaker_id in data["profiles"]:
            data["profiles"][speaker_id]["label"] = new_label
            _save(data)
