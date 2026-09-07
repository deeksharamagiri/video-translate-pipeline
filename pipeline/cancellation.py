"""
Shared job-cancellation primitives.

`JobCancelled` is the single exception type raised anywhere a running job
notices the user cancelled it -- whether that's the app's progress
callback (checked between pipeline stages) or `run_cancellable` (checked
every `poll_interval` seconds while a blocking subprocess like
whisper.cpp or ffmpeg is running). Using one shared type means
orchestrator.py's rollback and app.py's status handling catch it
regardless of where it was raised.
"""
import subprocess
import threading
from typing import Optional


class JobCancelled(Exception):
    pass


def check_cancelled(cancel_event: Optional[threading.Event]):
    """Raise JobCancelled if the given event is set. No-op if None."""
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Cancelled by user.")


def run_cancellable(
    cmd,
    cancel_event: Optional[threading.Event] = None,
    poll_interval: float = 0.25,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    subprocess.run()-alike for a command whose stdout/stderr should be
    captured, except it polls `cancel_event` while the process runs and
    kills it promptly (within ~poll_interval seconds) instead of
    blocking until the process would have finished on its own.

    Returns a subprocess.CompletedProcess, same as subprocess.run(cmd,
    stdout=PIPE, stderr=PIPE), so it's a drop-in replacement.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )

    stdout, stderr = b"", b""

    while True:
        try:
            stdout, stderr = proc.communicate(timeout=poll_interval)
            break

        except subprocess.TimeoutExpired:

            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                try:
                    proc.kill()
                    proc.communicate()
                except OSError:
                    # Process already exited between the timeout and
                    # the kill (e.g. it finished right as we cancelled).
                    pass

                raise JobCancelled("Cancelled by user.")

    return subprocess.CompletedProcess(
        cmd,
        proc.returncode,
        stdout,
        stderr,
    )
