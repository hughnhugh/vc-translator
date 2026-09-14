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
import re
import time

import torch
from opencc import OpenCC
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import speaker_store
from lang_codes import WHISPER_TO_FLORES as LANG_TO_FLORES
from lang_codes import validate_flores_codes
from overlay_core import VAD_SAMPLE_RATE, caption_q, is_reliable_transcription, run_translator

_t2s = OpenCC("t2s")
_s2t = OpenCC("s2t")

# Whisper only ever reports "zh" (it doesn't distinguish script) - these are
# the two script variants selectable as --target, both reachable from a "zh"
# source without any NLLB translation, just a script conversion.
CHINESE_TARGETS = {"zh": _t2s, "zh-hant": _s2t}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Whisper's task="translate" decoder occasionally fails to actually translate
# and just echoes the source text back instead (sometimes even through a
# script conversion, e.g. Simplified->Traditional Chinese) rather than
# producing English. A "translation" that's mostly non-Latin script is that
# failure, not a real translation - drop it rather than show it as one.
_NON_LATIN_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]")


def _looks_translated(text):
    if not text:
        return False
    return len(_NON_LATIN_RE.findall(text)) / len(text) < 0.3


def _is_degenerate(text):
    """NLLB sometimes decodes a short, context-less filler ("Uh-huh.",
    "Yeah, yeah, yeah.") into a long run of one repeated character instead
    of a real translation, e.g. ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,". Catch
    that so it doesn't get shown as if it were a translation."""
    stripped = text.replace(" ", "")
    if len(stripped) < 6:
        return False
    most_common_count = max(stripped.count(c) for c in set(stripped))
    return most_common_count / len(stripped) > 0.5


_nllb_tokenizer = None
_nllb_model = None
_speaker_classifier = None


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


def _load_speaker_model():
    global _speaker_classifier
    if _speaker_classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        caption_q.put(f"Loading speaker ID model ({SPEAKER_MODEL})...")
        _speaker_classifier = EncoderClassifier.from_hparams(
            source=SPEAKER_MODEL,
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda:0"},
            local_strategy=LocalStrategy.COPY,  # symlinks need admin/dev-mode privileges on Windows
        )
    return _speaker_classifier


def identify_speaker(segment_audio):
    classifier = _load_speaker_model()
    wav = torch.from_numpy(segment_audio).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode_batch(wav).squeeze().cpu().numpy()
    duration_sec = len(segment_audio) / VAD_SAMPLE_RATE
    return speaker_store.identify(embedding, duration_sec)


def _emit_caption(lead, speaker_id, lang, elapsed, native_text, translated_text=None):
    body = f"[{lang}, {elapsed:.1f}s] {native_text}"
    if translated_text is not None:
        body += f"\n    -> {translated_text}"
    if speaker_id:
        caption_q.put({"text": f"{lead}{body}", "speaker_id": speaker_id, "speaker_label": lead.strip()})
    else:
        caption_q.put(f"{lead}{body}")


def make_process_segment(target_lang):
    target_flores = LANG_TO_FLORES[target_lang]

    def process_segment(model, segment, source_tag=""):
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

        if source_tag == "mic":
            lead, speaker_id = "[You] ", None
        else:
            speaker_id, speaker_label = identify_speaker(segment)
            lead = f"{speaker_label} "

        if info.language == "zh":
            native_text = _t2s.convert(native_text)  # normalize display text to Simplified

            if target_lang in CHINESE_TARGETS:
                elapsed = time.time() - t0
                shown = CHINESE_TARGETS[target_lang].convert(native_text)
                _emit_caption(lead, speaker_id, info.language, elapsed, shown)
                return

        if info.language == target_lang:
            # already the target language - nothing to translate
            elapsed = time.time() - t0
            _emit_caption(lead, speaker_id, info.language, elapsed, native_text)
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
            if not _looks_translated(translated):
                translated = ""
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
                translated = translate_text(en_text, "eng_Latn", target_flores) if _looks_translated(en_text) else ""

        elapsed = time.time() - t0
        # The translate pass is a separate decode from the transcribe pass
        # above and isn't reliability-checked itself - it occasionally comes
        # back empty, or for a short/context-less filler NLLB can degenerate
        # into a run of one repeated character, even though the native
        # transcription was solid. Show the native text either way rather
        # than silently dropping the caption or showing that as if it were
        # a translation.
        if _is_degenerate(translated):
            translated = ""
        _emit_caption(lead, speaker_id, info.language, elapsed, native_text, translated or None)

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
