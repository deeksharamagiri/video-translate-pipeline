"""
qa_analyze.py -- automated QA pass over batch_qa_manifest.json outputs.

For every successfully-completed job, inspects:
  - ffprobe stream integrity (video/audio presence, codec, duration) for
    burned_in_mp4 and voiceover_mp4, compared against the source video.
  - ffmpeg silencedetect on each output's audio track.
  - SRT structural validation (overlaps, gaps, duplicates, empty entries,
    out-of-bounds timestamps, extreme reading speeds).
  - The pipeline's own self-reported quality_info (alignment scores, TTS
    coverage, ASR dropped ranges, translation warnings).

Writes a JSON report to qa_report.json and prints a human-readable summary.
"""
import json
import os
import re
import subprocess
import sys

FFPROBE = "/opt/homebrew/bin/ffprobe"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

PIPE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(PIPE_DIR, "batch_qa_manifest.json")
REPORT = os.path.join(PIPE_DIR, "qa_report.json")


def ffprobe_json(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return {"error": out.stderr.strip()}
    return json.loads(out.stdout)


def silencedetect(path, noise_db=-45, min_dur=1.0):
    out = subprocess.run(
        [FFMPEG, "-i", path, "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\-\d.]+)", out.stderr)]
    ends_durs = re.findall(r"silence_end:\s*([\-\d.]+)\s*\|\s*silence_duration:\s*([\-\d.]+)", out.stderr)
    intervals = []
    for i, s in enumerate(starts):
        if i < len(ends_durs):
            e, d = ends_durs[i]
            intervals.append({"start": s, "end": float(e), "duration": float(d)})
        else:
            intervals.append({"start": s, "end": None, "duration": None})
    return intervals


TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def parse_srt(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        m = TS_RE.search(lines[1]) if not TS_RE.search(lines[0]) else TS_RE.search(lines[0])
        ts_line_idx = 1 if TS_RE.search(lines[1]) else 0
        m = TS_RE.search(lines[ts_line_idx])
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text = "\n".join(lines[ts_line_idx + 1:]).strip()
        entries.append({"start": start, "end": end, "text": text})
    return entries


def validate_srt(entries, video_duration):
    issues = []
    for i, e in enumerate(entries):
        if e["end"] <= e["start"]:
            issues.append({"type": "zero_or_negative_duration", "index": i, "detail": e})
        dur = e["end"] - e["start"]
        if dur < 0.15:
            issues.append({"type": "extremely_short_duration", "index": i, "duration": dur, "text": e["text"][:60]})
        if dur > 15:
            issues.append({"type": "extremely_long_duration", "index": i, "duration": dur, "text": e["text"][:60]})
        if not e["text"].strip():
            issues.append({"type": "empty_text", "index": i})
        if i > 0:
            prev = entries[i - 1]
            if e["start"] < prev["end"] - 0.01:
                issues.append({"type": "overlap", "index": i, "prev_end": prev["end"], "this_start": e["start"]})
            if prev["text"].strip() and e["text"].strip() == prev["text"].strip():
                issues.append({"type": "duplicate_consecutive_text", "index": i, "text": e["text"][:60]})
        if e["start"] < 0:
            issues.append({"type": "negative_start", "index": i})
        if video_duration and e["end"] > video_duration + 1.0:
            issues.append({"type": "end_beyond_video_duration", "index": i, "end": e["end"], "video_duration": video_duration})
        # reading speed (chars/sec) -- rough heuristic for burn-in/subtitle legibility
        chars = len(e["text"].replace("\n", " "))
        if dur > 0 and chars > 0:
            cps = chars / dur
            if cps > 30:
                issues.append({"type": "high_reading_speed_cps", "index": i, "cps": round(cps, 1), "text": e["text"][:60]})
    last_end = entries[-1]["end"] if entries else 0
    coverage_gap_to_video_end = (video_duration - last_end) if video_duration else None
    return issues, last_end, coverage_gap_to_video_end


def analyze_job(name, entry):
    result = entry.get("result", {})
    stats = result.get("stats", {})
    job_id = result.get("job_id")
    video_path = entry.get("video_path")

    report = {"name": name, "job_id": job_id, "video_path": video_path}

    src_probe = ffprobe_json(video_path)
    src_duration = float(src_probe.get("format", {}).get("duration", 0)) if "format" in src_probe else None
    report["source_duration_sec"] = src_duration

    report["quality_info"] = stats.get("quality", {})
    report["timing_seconds"] = stats.get("timing_seconds", {})
    report["engine"] = stats.get("engine")
    report["source_lang_detected"] = stats.get("source_lang")
    report["preprocess"] = stats.get("preprocess", {})

    outputs = {}
    for kind in ["burned_in_mp4", "voiceover_mp4"]:
        path = result.get(kind)
        info = {"path": path}
        if path and os.path.exists(path):
            probe = ffprobe_json(path)
            info["ffprobe_error"] = probe.get("error")
            fmt = probe.get("format", {})
            info["container_duration_sec"] = float(fmt.get("duration", 0)) if fmt.get("duration") else None
            streams = probe.get("streams", [])
            v_streams = [s for s in streams if s.get("codec_type") == "video"]
            a_streams = [s for s in streams if s.get("codec_type") == "audio"]
            info["video_stream_count"] = len(v_streams)
            info["audio_stream_count"] = len(a_streams)
            if v_streams:
                v = v_streams[0]
                info["video_codec"] = v.get("codec_name")
                info["video_duration_sec"] = float(v.get("duration", 0)) if v.get("duration") else None
                info["nb_frames"] = v.get("nb_frames")
                info["avg_frame_rate"] = v.get("avg_frame_rate")
            if a_streams:
                a = a_streams[0]
                info["audio_codec"] = a.get("codec_name")
                info["audio_duration_sec"] = float(a.get("duration", 0)) if a.get("duration") else None
                info["audio_sample_rate"] = a.get("sample_rate")
                info["audio_channels"] = a.get("channels")
            else:
                info["MISSING_AUDIO_STREAM"] = True

            if src_duration and info.get("container_duration_sec"):
                info["duration_delta_vs_source_sec"] = round(info["container_duration_sec"] - src_duration, 3)

            if a_streams:
                info["silence_intervals"] = silencedetect(path)
                long_silence = [iv for iv in info["silence_intervals"] if (iv["duration"] or 0) >= 3.0]
                info["long_silence_gt_3s_count"] = len(long_silence)
                info["long_silences"] = long_silence
        else:
            info["MISSING_FILE"] = True
        outputs[kind] = info
    report["outputs"] = outputs

    srt_path = result.get("srt")
    if srt_path and os.path.exists(srt_path):
        entries = parse_srt(srt_path)
        vid_dur = outputs.get("burned_in_mp4", {}).get("container_duration_sec") or src_duration
        issues, last_end, gap_to_end = validate_srt(entries, vid_dur)
        report["srt"] = {
            "path": srt_path,
            "entry_count": len(entries),
            "last_subtitle_end_sec": last_end,
            "video_duration_sec": vid_dur,
            "gap_last_subtitle_to_video_end_sec": gap_to_end,
            "issues": issues,
            "issue_count": len(issues),
        }
    else:
        report["srt"] = {"MISSING_FILE": True}

    return report


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    full_report = {}
    for name, entry in manifest.items():
        if entry.get("status") != "ok":
            full_report[name] = {"status": "error", "error": entry.get("error")}
            continue
        print(f"Analyzing {name} ...", file=sys.stderr)
        full_report[name] = analyze_job(name, entry)

    with open(REPORT, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
