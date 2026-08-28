"""
Test evidence for the fail-fast check added to pipeline/orchestrator.py:
if ASR produces zero usable segments (e.g. whisper.cpp misdetected the
source language and every segment degraded into hallucinated repetition,
correctly dropped by the hallucination filter), the job must fail
immediately with a clear, actionable JobError -- not proceed silently
through translation and subtitle generation to produce an empty SRT/VTT
with no error at all.

Reproduced from a real failure: a 7-minute English clip auto-detected as
Marathi produced 14 raw whisper.cpp segments, all hallucinated repetition,
0 usable segments -- which previously only surfaced as a confusing
"No segments produced TTS audio." error, and only when voiceover was
requested at all.
"""
import pytest

from pipeline import orchestrator, stage1_preprocess, stage2_asr


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


def test_run_job_fails_fast_when_asr_produces_zero_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(
        stage1_preprocess, "run_stage1", lambda *a, **k: _FakePreprocessResult()
    )
    monkeypatch.setattr(
        stage2_asr,
        "run_stage2",
        lambda *a, **k: stage2_asr.ASRResult(
            detected_language="mr", language_probability=1.0, segments=[]
        ),
    )

    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"fake video")

    with pytest.raises(orchestrator.JobError, match="No usable speech"):
        orchestrator.run_job(str(input_file), None, "hin", False, False)
