"""
Live translator for Discord (or any system audio).
Captures whatever your PC is currently playing (WASAPI loopback) plus your
mic, detects speech with Silero VAD, and translates it to a target language
of your choosing (--target). Whisper natively translates any language
straight into English; for every other target, this pivots through a
second text-translation model (NLLB-200) since Whisper itself can't
translate directly into anything but English.

Shows captions in a small always-on-top overlay window.

Usage:
    python translate_vc.py --target en
    python translate_vc.py --target zh
    python translate_vc.py --target az --model medium

Run it twice with different --target values to get two simultaneous
overlays (e.g. one for English, one for Chinese).
"""

import argparse
import time

import torch
from opencc import OpenCC
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lang_codes import WHISPER_TO_FLORES as LANG_TO_FLORES
from lang_codes import validate_flores_codes
from overlay_core import caption_q, is_reliable_transcription, run_translator

_t2s = OpenCC("t2s")
_s2t = OpenCC("s2t")

# Whisper only ever reports "zh" (it doesn't distinguish script) - these are
# the two script variants selectable as --target, both reachable from a "zh"
# source without any NLLB translation, just a script conversion.
CHINESE_TARGETS = {"zh": _t2s, "zh-hant": _s2t}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

_nllb_tokenizer = None
_nllb_model = None


def _load_nllb():
    global _nllb_tokenizer, _nllb_model
    if _nllb_model is None:
        caption_q.put(f"Loading translation model ({NLLB_MODEL})...")
        _nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to("cuda").half()
    return _nllb_tokenizer, _nllb_model


def translate_text(text, src_flores, target_flores):
    tokenizer, model = _load_nllb()
    tokenizer.src_lang = src_flores
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_flores)
    with torch.no_grad():
        output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=200)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


def make_process_segment(target_lang):
    target_flores = LANG_TO_FLORES[target_lang]

    def process_segment(model, segment, source_tag=""):
        prefix = "[You] " if source_tag == "mic" else ""

        t0 = time.time()
        segments_gen, info = model.transcribe(
            segment,
            task="transcribe",
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
        )
        native_segments = list(segments_gen)

        if not is_reliable_transcription(native_segments, info):
            return

        native_text = "".join(s.text for s in native_segments).strip()
        if not native_text:
            return

        if info.language == "zh":
            native_text = _t2s.convert(native_text)  # normalize display text to Simplified

            if target_lang in CHINESE_TARGETS:
                elapsed = time.time() - t0
                shown = CHINESE_TARGETS[target_lang].convert(native_text)
                caption_q.put(f"{prefix}[{info.language}, {elapsed:.1f}s] {shown}")
                return

        if info.language == target_lang:
            # already the target language - nothing to translate
            elapsed = time.time() - t0
            caption_q.put(f"{prefix}[{info.language}, {elapsed:.1f}s] {native_text}")
            return

        if target_lang == "en":
            # Whisper's own translate task is a strong, purpose-built X->English
            # decoder - no need to pivot through NLLB for this common case
            en_segments, _ = model.transcribe(
                segment,
                task="translate",
                vad_filter=False,
                beam_size=5,
                language=info.language,
                condition_on_previous_text=False,
            )
            translated = "".join(s.text for s in en_segments).strip()
        else:
            src_flores = LANG_TO_FLORES.get(info.language)
            if src_flores:
                translated = translate_text(native_text, src_flores, target_flores)
            else:
                en_segments, _ = model.transcribe(
                    segment,
                    task="translate",
                    vad_filter=False,
                    beam_size=5,
                    language=info.language,
                    condition_on_previous_text=False,
                )
                en_text = "".join(s.text for s in en_segments).strip()
                translated = translate_text(en_text, "eng_Latn", target_flores) if en_text else ""

        elapsed = time.time() - t0
        if translated:
            caption_q.put(f"{prefix}[{info.language}, {elapsed:.1f}s] {native_text}\n    -> {translated}")

    return process_segment


def main():
    parser = argparse.ArgumentParser(description="Live translator overlay for Discord/system audio.")
    parser.add_argument(
        "--target", "-t", default="en",
        help=f"Target language code. Supported: {', '.join(sorted(LANG_TO_FLORES))} (default: en)",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Whisper model size (default: large-v3 for --target en, medium otherwise)",
    )
    parser.add_argument(
        "--position", choices=["top", "bottom"], default=None,
        help="Overlay screen anchor (default: bottom for --target en, top otherwise)",
    )
    parser.add_argument("--no-mic", action="store_true", help="Don't also capture your microphone")
    args = parser.parse_args()

    target = args.target.lower()
    if target not in LANG_TO_FLORES:
        raise SystemExit(f"Unsupported target language '{target}'. Supported: {', '.join(sorted(LANG_TO_FLORES))}")

    if target != "en":
        # Load (and validate) NLLB upfront rather than lazily on first non-native
        # phrase, so a bad flores code or missing model surfaces immediately.
        tokenizer, _ = _load_nllb()
        bad_codes = validate_flores_codes(tokenizer, LANG_TO_FLORES)
        if bad_codes:
            print(f"Warning: dropping unrecognized NLLB codes: {sorted(bad_codes)}")
        if target not in LANG_TO_FLORES:
            raise SystemExit(f"Target '{target}' was dropped as an unrecognized NLLB code.")

    model_size = args.model or ("large-v3" if target == "en" else "medium")
    anchor = args.position or ("bottom" if target == "en" else "top")

    run_translator(
        make_process_segment(target),
        model_size=model_size,
        title=f"Live Captions ({target.upper()})",
        anchor=anchor,
        capture_mic=not args.no_mic,
    )


if __name__ == "__main__":
    main()
