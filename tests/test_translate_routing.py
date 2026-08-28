"""
Test evidence for pipeline/translate.py's engine routing logic -- the
diagram's rule ("both Indian languages -> IndicTrans2, else -> NLLB") is
the kind of thing that's easy to silently break while touching nearby
code (it already regressed once for English<->Indic, per the comments in
config.py). These tests exercise routing/hashing/glossary logic only --
no model weights are loaded, so they run fast and fully offline.
"""
from pipeline.translate import (
    is_indic_pair,
    indictrans2_direction,
    model_version_tag,
    _protect_glossary_terms,
    _restore_glossary_terms,
)


def test_indic_indic_pair_routes_to_indictrans2():
    assert is_indic_pair("hin", "mar") is True
    assert indictrans2_direction("hin", "mar") == "indic_indic"


def test_english_to_indic_uses_dedicated_checkpoint():
    # Regression guard: English used to be miscategorized as "Indic" and
    # this pair fell through to the wrong (indic-indic) checkpoint.
    assert indictrans2_direction("eng", "hin") == "en_indic"


def test_indic_to_english_uses_dedicated_checkpoint():
    assert indictrans2_direction("hin", "eng") == "indic_en"


def test_non_indic_pair_falls_through_to_nllb():
    assert is_indic_pair("fra", "deu") is False
    assert indictrans2_direction("fra", "deu") is None


def test_english_pair_is_not_treated_as_indic():
    # eng/eng round-trip (or eng->non-indic) must not match is_indic_pair.
    assert is_indic_pair("eng", "fra") is False


def test_model_version_tag_changes_with_engine_override():
    auto_tag = model_version_tag("hin", "mar", "auto")
    forced_nllb_tag = model_version_tag("hin", "mar", "nllb")
    assert auto_tag != forced_nllb_tag
    assert "nllb" in forced_nllb_tag.lower()


def test_model_version_tag_stable_for_same_inputs():
    # The translation-memory cache keys on this value; it must be
    # deterministic for identical inputs.
    a = model_version_tag("hin", "eng", "auto")
    b = model_version_tag("hin", "eng", "auto")
    assert a == b


def test_glossary_protect_and_restore_round_trip():
    glossary = {"PDS": {"hin": "सार्वजनिक वितरण प्रणाली"}}
    protected, restore_map = _protect_glossary_terms(
        "Check your PDS card.", "hin", glossary
    )
    assert "PDS" not in protected
    assert restore_map

    # Simulate the NMT model passing the placeholder through untouched.
    restored = _restore_glossary_terms(protected, restore_map)
    assert "सार्वजनिक वितरण प्रणाली" in restored


def test_glossary_protect_no_terms_present_is_noop():
    protected, restore_map = _protect_glossary_terms(
        "Nothing to protect here.", "hin", {"PDS": {"hin": "x"}}
    )
    assert protected == "Nothing to protect here."
    assert restore_map == {}


def test_model_version_tag_encodes_beam_width(monkeypatch):
    # TRANSLATION_NUM_BEAMS is a CPU speed/quality knob (config.py) -- the
    # TM cache must never serve a translation made at one beam width as if
    # it were made at another, so the tag has to change when this does.
    import pipeline.translate as translate_mod

    monkeypatch.setattr(translate_mod, "TRANSLATION_NUM_BEAMS", 1)
    tag_beam1 = translate_mod.model_version_tag("hin", "mar", "auto")

    monkeypatch.setattr(translate_mod, "TRANSLATION_NUM_BEAMS", 5)
    tag_beam5 = translate_mod.model_version_tag("hin", "mar", "auto")

    assert tag_beam1 != tag_beam5
    assert "beam1" in tag_beam1
    assert "beam5" in tag_beam5
