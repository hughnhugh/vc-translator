"""
Maps faster-whisper language codes to NLLB-200 flores200 codes, used both
for selecting --target languages and for identifying a detected source
language's code when pivoting text through NLLB.

Whisper recognizes ~99 languages; most are covered here. A handful of
rarer ones (Latin, Breton, Faroese, Sanskrit, Tatar, Hawaiian, Bashkir) are
left out because their exact NLLB-200 code isn't reliably known - they
still transcribe fine, translation just pivots through English instead
(the same fallback already used for any language not in this table).

NLLB's codes aren't always a predictable transform of the 2-letter code
(script suffixes and dialect choices vary), so call validate_flores_codes()
once the NLLB tokenizer is loaded - it drops anything the tokenizer
doesn't actually recognize instead of silently mistranslating with a bad
or nonexistent code.
"""

WHISPER_TO_FLORES = {
    "en": "eng_Latn", "zh": "zho_Hans", "zh-hant": "zho_Hant", "de": "deu_Latn", "es": "spa_Latn",
    "ru": "rus_Cyrl", "ko": "kor_Hang", "fr": "fra_Latn", "ja": "jpn_Jpan",
    "pt": "por_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "ca": "cat_Latn",
    "nl": "nld_Latn", "ar": "arb_Arab", "sv": "swe_Latn", "it": "ita_Latn",
    "id": "ind_Latn", "hi": "hin_Deva", "fi": "fin_Latn", "vi": "vie_Latn",
    "he": "heb_Hebr", "uk": "ukr_Cyrl", "el": "ell_Grek", "ms": "zsm_Latn",
    "cs": "ces_Latn", "ro": "ron_Latn", "da": "dan_Latn", "hu": "hun_Latn",
    "ta": "tam_Taml", "no": "nob_Latn", "th": "tha_Thai", "ur": "urd_Arab",
    "hr": "hrv_Latn", "bg": "bul_Cyrl", "lt": "lit_Latn", "mi": "mri_Latn",
    "ml": "mal_Mlym", "cy": "cym_Latn", "sk": "slk_Latn", "te": "tel_Telu",
    "fa": "pes_Arab", "lv": "lvs_Latn", "bn": "ben_Beng", "sr": "srp_Cyrl",
    "az": "azj_Latn", "sl": "slv_Latn", "kn": "kan_Knda", "et": "est_Latn",
    "mk": "mkd_Cyrl", "eu": "eus_Latn", "is": "isl_Latn", "hy": "hye_Armn",
    "ne": "npi_Deva", "mn": "khk_Cyrl", "bs": "bos_Latn", "kk": "kaz_Cyrl",
    "sq": "als_Latn", "sw": "swh_Latn", "gl": "glg_Latn", "mr": "mar_Deva",
    "pa": "pan_Guru", "si": "sin_Sinh", "km": "khm_Khmr", "sn": "sna_Latn",
    "yo": "yor_Latn", "so": "som_Latn", "af": "afr_Latn", "oc": "oci_Latn",
    "ka": "kat_Geor", "be": "bel_Cyrl", "tg": "tgk_Cyrl", "sd": "snd_Arab",
    "gu": "guj_Gujr", "am": "amh_Ethi", "yi": "ydd_Hebr", "lo": "lao_Laoo",
    "uz": "uzn_Latn", "ht": "hat_Latn", "ps": "pbt_Arab", "tk": "tuk_Latn",
    "nn": "nno_Latn", "mt": "mlt_Latn", "lb": "ltz_Latn", "my": "mya_Mymr",
    "bo": "bod_Tibt", "tl": "tgl_Latn", "mg": "plt_Latn", "as": "asm_Beng",
    "ln": "lin_Latn", "ha": "hau_Latn", "jw": "jav_Latn", "su": "sun_Latn",
    "yue": "yue_Hant",
}


LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese (Simplified)", "zh-hant": "Chinese (Traditional)", "de": "German",
    "es": "Spanish", "ru": "Russian", "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese",
    "tr": "Turkish", "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese", "he": "Hebrew",
    "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech", "ro": "Romanian", "da": "Danish",
    "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian", "th": "Thai", "ur": "Urdu", "hr": "Croatian",
    "bg": "Bulgarian", "lt": "Lithuanian", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani",
    "sl": "Slovenian", "kn": "Kannada", "et": "Estonian", "mk": "Macedonian", "eu": "Basque", "is": "Icelandic",
    "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian",
    "sw": "Swahili", "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer",
    "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian",
    "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish",
    "lo": "Lao", "uz": "Uzbek", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen",
    "nn": "Norwegian Nynorsk", "mt": "Maltese", "lb": "Luxembourgish", "my": "Burmese", "bo": "Tibetan",
    "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese", "ln": "Lingala", "ha": "Hausa", "jw": "Javanese",
    "su": "Sundanese", "yue": "Cantonese",
}


def validate_flores_codes(tokenizer, mapping):
    """Remove any code the loaded NLLB tokenizer doesn't actually recognize
    (mutates `mapping` in place). Returns the set of bad flores codes found."""
    unk_id = tokenizer.unk_token_id
    bad_codes = {v for v in set(mapping.values()) if tokenizer.convert_tokens_to_ids(v) == unk_id}
    if bad_codes:
        for key in [k for k, v in mapping.items() if v in bad_codes]:
            del mapping[key]
    return bad_codes
