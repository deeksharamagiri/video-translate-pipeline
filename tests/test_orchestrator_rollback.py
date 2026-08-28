"""
Test evidence for the rollback behavior added to pipeline/orchestrator.py:
if any stage raises, the partially-written jobs/<job_id>/ directory must
be deleted rather than left behind as half-finished output that could
later be confused for a real job.
"""
import os

import pytest

from pipeline import orchestrator
from config import JOBS_DIR


class _FakeUUID:
    hex = "deadbeef0000"


def test_run_job_rolls_back_job_dir_on_stage_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.uuid, "uuid4", lambda: _FakeUUID())

    def boom(*args, **kwargs):
        raise RuntimeError("stage1 exploded")

    monkeypatch.setattr(orchestrator.stage1_preprocess, "run_stage1", boom)

    input_file = tmp_path / "input.mp3"
    input_file.write_bytes(b"fake audio")

    job_dir = os.path.join(JOBS_DIR, _FakeUUID.hex)
    # Sanity: make sure a stale directory from a previous failed test run
    # doesn't produce a false pass.
    assert not os.path.exists(job_dir)

    with pytest.raises(RuntimeError, match="stage1 exploded"):
        orchestrator.run_job(str(input_file), None, "hin", False, False)

    assert not os.path.exists(job_dir), (
        "Failed job's directory should be rolled back (deleted), not left "
        "as partial output."
    )


def test_run_job_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "input.xyz"
    bad_file.write_bytes(b"not a real media file")

    with pytest.raises(orchestrator.JobError):
        orchestrator.run_job(str(bad_file), None, "hin", False, False)
