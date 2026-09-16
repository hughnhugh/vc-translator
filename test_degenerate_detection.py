"""Regression test for _is_degenerate() - NLLB occasionally decodes a
short, context-less filler into a degenerate loop instead of a real
translation, either a repeated single character or (the bug this guards
against) the same short phrase repeated over and over. Pure text logic,
no GPU/Whisper/NLLB needed."""

from translate_vc import _is_degenerate

# Real NLLB output observed in production: "It's just fun for me" translated
# into "我只是很开心" repeated ~26 times instead of once.
REPEATED_PHRASE_BUG = (
    "只是,我只是很开心,我只是很开心,我只是很开心,我只是很开心,我只是很开心,"
    "我只是很开心,我只是很开心,我只是很开心,我只是很开心,我只是很开心,我只是很开心,"
    "我只是很开心,我只是很开心,我只是很开心,我只是很开心,我只是很开心,我只是很开心,"
    "我只是很开心,我只是很开心,我只是很开心,我只是很开心,我只是很开心,我只是很开心,"
    "我只是很开心,我很开心,我很开心"
)


def test_repeated_phrase_loop_is_degenerate():
    assert _is_degenerate(REPEATED_PHRASE_BUG) is True


def test_repeated_single_char_is_degenerate():
    assert _is_degenerate(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,") is True


def test_normal_sentence_is_not_degenerate():
    assert _is_degenerate("我只是觉得这很有趣,我喜欢学习新的语言。") is False


def test_short_legitimate_repetition_is_not_degenerate():
    # up to 3 genuine repeats (e.g. translating "yes, yes, yes") shouldn't
    # be suppressed - only runaway loops should be
    assert _is_degenerate("是,是,是") is False


def test_short_text_is_never_degenerate():
    assert _is_degenerate("Hi.") is False


if __name__ == "__main__":
    test_repeated_phrase_loop_is_degenerate()
    test_repeated_single_char_is_degenerate()
    test_normal_sentence_is_not_degenerate()
    test_short_legitimate_repetition_is_not_degenerate()
    test_short_text_is_never_degenerate()
    print("ALL OK")
