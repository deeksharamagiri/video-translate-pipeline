"""
Sanity checks on config.py's declared limits/coverage -- catches the kind
of drift where the UI, README, or a rubric claim ("22 languages") silently
stops matching what config.py actually declares.
"""
import config


def test_allowed_video_and_audio_extensions_are_disjoint():
    assert config.ALLOWED_VIDEO_EXT.isdisjoint(config.ALLOWED_AUDIO_EXT)


def test_indic_langs_has_22_official_languages():
    assert len(config.INDIC_LANGS) == 22


def test_indic_langs_does_not_include_english():
    # Regression guard for the bug documented in config.py: English was
    # previously included here as an "English pivot", which caused
    # English<->Hindi to wrongly route through the indic-indic checkpoint.
    assert "eng" not in config.INDIC_LANGS


def test_size_limits_are_positive():
    assert config.MAX_VIDEO_SIZE_MB > 0
    assert config.MAX_AUDIO_SIZE_MB > 0
    assert config.MAX_VIDEO_DURATION_SEC > 0
    assert config.MAX_AUDIO_DURATION_SEC > 0


def test_indic_tts_speaker_default_is_valid():
    assert config.INDIC_TTS_DEFAULT_SPEAKER in ("male", "female")


def test_indic_tts_lang_map_is_subset_of_indic_langs_plus_english():
    # Every voiceover language must be either an Indic language or English
    # -- a typo here would silently produce a language nothing downstream
    # recognizes.
    allowed = config.INDIC_LANGS | {"eng"}
    assert set(config.INDIC_TTS_LANG_ZIP_MAP.keys()) <= allowed
