"""
Test evidence for surfacing ASR hallucination-drop gaps as a visible
quality warning, instead of only ever reaching a print() statement that
vanishes once the job finishes.

Reproduced from a real job: whisper.cpp dropped 4 stretches of a clip as
hallucinated repetition (including the entire first 60 seconds), leaving
correct-but-incomplete subtitles/translation with no indication anywhere
in the UI or Job Report of which parts were skipped or why -- reported as
"not translating the initial parts of the video."
"""
import pytest

from pipeline import delivery, orchestrator, stage1_preprocess, stage2_asr, translate


class _FakePreprocessResult:
    def __init__(self):
        self.audio_wav_path = "/tmp/fake.wav"
        self.subtitle_track_found = False
        self.subtitle_srt_path = None
        self.denoise_applied = False
        self.snr_db = None
        self.warnings = []
        self.input_kind = "video"

    def to_dict(self):
        return {
            "audio_wav_path": self.audio_wav_path,
            "subtitle_track_found": self.subtitle_track_found,
            "subtitle_srt_path": self.subtitle_srt_path,
            "denoise_applied": self.denoise_applied,
            "snr_db": self.snr_db,
            "warnings": self.warnings,
            "input_kind": self.input_kind,
        }


def test_run_job_surfaces_asr_dropped_ranges_as_a_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(
        stage1_preprocess, "run_stage1", lambda *a, **k: _FakePreprocessResult()
    )

    fake_segments = [
        stage2_asr.TranscriptSegment(
            start=60.0, end=71.0, text="असली मजकूर", confidence=0.8
        ),
    ]
    monkeypatch.setattr(
        stage2_asr,
        "run_stage2",
        lambda *a, **k: stage2_asr.ASRResult(
            detected_language="mr",
            language_probability=1.0,
            segments=fake_segments,
            dropped_ranges=[(0.0, 30.0), (30.0, 60.0)],
        ),
    )
    monkeypatch.setattr(
        translate,
        "translate_batch",
        lambda texts, *a, **k: (["translated text"] * len(texts), "fake-engine"),
    )

    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"fake video")

    result = orchestrator.run_job(str(input_file), None, "hin", False, False)

    warnings = result["quality"]["warnings"]
    assert len(warnings) == 1
    assert "0s-30s" in warnings[0]
    assert "30s-60s" in warnings[0]
    assert "60s of audio" in warnings[0]


def test_run_job_preserves_asr_dropped_warning_when_voiceover_also_runs(
    monkeypatch, tmp_path
):
    """
    Regression test for a second, more consequential instance of the same
    underlying complaint: the ASR dropped-ranges warning above was only
    ever reaching quality_info["warnings"] when voiceover was OFF. With
    voiceover on (the pipeline's flagship feature, and what every real
    job in practice uses it for), a later `quality_info.update(voiceover_
    quality)` was silently replacing the whole "warnings" key -- built
    from a completely separate dict with no knowledge of the ASR drop --
    discarding it outright. Confirmed against a real 8-video batch: every
    job that requested voiceover and had dropped audio showed only a
    generic voiceover-timing warning, never the dropped-audio one.
    """
    monkeypatch.setattr(
        stage1_preprocess, "run_stage1", lambda *a, **k: _FakePreprocessResult()
    )

    fake_segments = [
        stage2_asr.TranscriptSegment(
            start=60.0, end=71.0, text="असली मजकूर", confidence=0.8
        ),
    ]
    monkeypatch.setattr(
        stage2_asr,
        "run_stage2",
        lambda *a, **k: stage2_asr.ASRResult(
            detected_language="mr",
            language_probability=1.0,
            segments=fake_segments,
            dropped_ranges=[(0.0, 30.0), (30.0, 60.0)],
        ),
    )
    monkeypatch.setattr(
        translate,
        "translate_batch",
        lambda texts, *a, **k: (["translated text"] * len(texts), "fake-engine"),
    )
    monkeypatch.setattr(delivery, "voiceover_available", lambda lang: True)
    monkeypatch.setattr(
        delivery,
        "build_voiceover_track",
        lambda *a, **k: (
            "/tmp/fake_voiceover.wav",
            {
                "warnings": [
                    "Voiceover timing differs noticeably from the "
                    "original segment timing."
                ],
                "tts_success_segments": 1,
                "tts_failed_segments": 0,
                "tts_coverage_percent": 100.0,
                "numeric_segments": 0,
                "numeric_expressions": 0,
                "decimal_expressions": 0,
                "numeric_normalization_success": 0,
                "numeric_normalization_failures": 0,
                "numeric_normalization_percent": 100.0,
                "alignment_scores": [80.0],
                "average_alignment_score": 80.0,
                "overall_quality_score": 80.0,
                "segment_details": [],
            },
        ),
    )
    monkeypatch.setattr(
        delivery,
        "mux_voiceover_onto_video",
        lambda *a, **k: "/tmp/fake_voiceover_out.mp4",
    )

    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"fake video")

    result = orchestrator.run_job(str(input_file), None, "hin", False, True)

    warnings = result["quality"]["warnings"]
    assert any("60s of audio" in w for w in warnings), (
        "ASR dropped-audio warning was lost once voiceover ran: "
        f"{warnings}"
    )
    assert any("Voiceover timing differs" in w for w in warnings), (
        "voiceover's own warning should also survive the merge: "
        f"{warnings}"
    )
