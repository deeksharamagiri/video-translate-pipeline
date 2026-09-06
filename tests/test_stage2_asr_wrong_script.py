"""
Test evidence for _is_wrong_script, added after a real job on "401.1.mp4"
(Marathi field audio) reported "finished ok" but had ~80% of its runtime
missing from the output with no warning explaining why.

Root cause: whisper.cpp confidently decoded large stretches into fluent,
non-repeating text in an unrelated Indic script (Bengali, Gurmukhi) instead
of Marathi/Devanagari. _is_degenerate_repetition only catches mechanical
token/character repetition loops, so this hallucination pattern passed
through untouched -- it wasn't repetitive, just wrong.
"""
from pipeline.stage2_asr import _is_degenerate_repetition, _is_wrong_script


def test_fluent_wrong_script_hallucination_is_caught():
    # Reproduced directly from a real whisper.cpp run on "401.1.mp4"
    # (language hint "mar"/Devanagari) -- fluent Gurmukhi text, no
    # token/phrase repeats itself enough to trip the repetition filter.
    text = (
        "਷ਿਲਿਪਾਲਨ ਷ਿਲਿਪਾਲਨਪਦਿ ਯਾਪਨ ਷ਿਲਿਪਾਲਨ ਷ਿਲਿਪਾਲਨ ਸਰਵਿਪਦਿ "
        "ਸ਼ਿਕਿਕਿਤਿ ਸ਼ਿਕਿਤਿ ਷ਿਲਿਪਾਲਨ ਗਾਟ ਆਨ ਫਿਲਿਲਿਮਾਰਗਦਾਰ਷ਕ"
    )
    assert _is_degenerate_repetition(text) is False
    assert _is_wrong_script(text, "mr") is True
    assert _is_wrong_script(text, "mar") is True


def test_pure_repetition_hallucination_still_caught_by_both():
    text = "ব " * 60
    assert _is_degenerate_repetition(text) is True
    assert _is_wrong_script(text, "mar") is True


def test_real_marathi_text_is_not_flagged():
    text = (
        "शेली वेवस्तापना वर सम्मदित प्रछिक्षण अने एक्स्पोचर देवुन "
        "विस्तार अनी सेवा प्रदाता मुनुन फिल्ड मार्ग दर्शक"
    )
    assert _is_degenerate_repetition(text) is False
    assert _is_wrong_script(text, "mar") is False


def test_short_text_is_not_judged():
    # Too little script-block content to trust either way.
    assert _is_wrong_script("ok", "mar") is False
    assert _is_wrong_script("", "mar") is False


def test_unmapped_language_is_never_flagged():
    assert _is_wrong_script("anything at all", "unknown") is False
    assert _is_wrong_script("ব ব ব ব ব ব ব", "eng") is False
