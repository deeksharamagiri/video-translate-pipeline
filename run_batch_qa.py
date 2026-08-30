import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import run_job

VIDEOS = [
    ("401.1", "/Users/nikhil/Downloads/Videos/401.1/401.1.mp4"),
    ("401.2_HOUSING_OF_GOAT", "/Users/nikhil/Downloads/Videos/401.2 HOUSING OF GOAT/401.2 HOUSING OF GOAT.mp4"),
    ("401.3", "/Users/nikhil/Downloads/Videos/401.3/401.3.mp4"),
    ("401.4_storyboard", "/Users/nikhil/Downloads/Videos/401.4 storyboard/401.4 storyboard.mp4"),
    ("401.5", "/Users/nikhil/Downloads/Videos/401.5/401.5.mp4"),
    ("401.6", "/Users/nikhil/Downloads/Videos/401.6/401.6.mp4"),
    ("401.7_Field_Guide_Women_Group", "/Users/nikhil/Downloads/Videos/401.7 Field Gude_ Women Group/401.7 Field Gude_ Women Group.mp4"),
    ("604.3", "/Users/nikhil/Downloads/Videos/604.3/604.3.mp4"),
]

manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_qa_manifest.json")

results = {}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        results = json.load(f)

for name, path in VIDEOS:
    if results.get(name, {}).get("status") == "ok":
        print(f"\nSKIP {name} (already done)", flush=True)
        continue

    print(f"\n{'='*80}\nSTARTING {name} ({path})\n{'='*80}", flush=True)
    t0 = time.time()
    try:
        result = run_job(
            path,
            None,       # source_lang_hint -- auto-detect (explicit hint is broken, see QA report)
            "hin",      # target_lang
            True,       # want_burned_in
            True,       # want_voiceover
            None,       # progress_cb
            "auto",     # engine_override
            "whisper",  # asr_engine
            None,       # tts_speaker
        )
        elapsed = time.time() - t0
        results[name] = {
            "status": "ok",
            "elapsed_sec": elapsed,
            "video_path": path,
            "result": result,
        }
        print(f"\nDONE {name} in {elapsed:.1f}s -> job_id={result.get('job_id')}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        results[name] = {
            "status": "error",
            "elapsed_sec": elapsed,
            "video_path": path,
            "error": str(e),
        }
        print(f"\nFAILED {name} after {elapsed:.1f}s: {e}", flush=True)

    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

print("\nALL DONE", flush=True)
