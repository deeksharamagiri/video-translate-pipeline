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

from pipeline import orchestrator, stage1_preprocess, stage2_asr, translate


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

