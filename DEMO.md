# Demo Script (time-boxed walkthrough)

Goal: show the end-to-end journey and its value in ~5 minutes, with a
fallback plan if live processing is too slow or something breaks on the
day.

## Before the demo (do this the night before / morning of)

Live model downloads mid-demo will eat your time budget — pre-warm
everything once so every job on demo day is fully offline and fast:

1. Start the app once and run one full job (video, both output boxes
   ticked) for each engine path you plan to demo, so ASR / IndicTrans2 /
   NLLB / Indic-TTS models are all downloaded and cached.
2. Confirm `HF_TOKEN` is set if you're demoing an Indic↔Indic pair (see
   README "The one manual step") — a 401 error live is avoidable.
3. Pick and stage 1-2 short (30-60s) source clips in advance — short clips
   keep the live processing time inside the demo window. Know their
   approximate processing time from step 1's dry run.
4. **Fallback**: keep a completed job's output files (SRT/VTT/DOCX/MP4s)
   from that dry run on hand, e.g. in a separate folder, so you can show
   finished output immediately if live processing stalls or the venue's
   power/time runs short.

## Walkthrough (~5 min)

| Time | Step | What to say |
|---|---|---|
| 0:00–0:30 | Open `http://127.0.0.1:5000` | One laptop, no internet needed once set up — built for field teams without connectivity. |
| 0:30–1:00 | Drop in the staged clip, pick a target language, tick "burned-in" and "voiceover" | 22 Indian languages, routes automatically between IndicTrans2 (Indic↔Indic) and NLLB-200 (everything else). |
| 1:00–1:15 | Hit **Run Pipeline** | Progress bar shows the actual stage (ASR → segmentation → translation → subtitles → voiceover) live. |
| 1:15–3:00 | While it processes | Talk through the Job Report DOCX from a *previous* completed job (fallback file) — numeric-preservation score, ASR confidence, translation warnings — so the audience isn't just staring at a progress bar. |
| 3:00–4:00 | Job finishes | Download and play the burned-in MP4 and/or voiceover MP4. Show the SRT/VTT are also there for editors who just want subtitle files. |
| 4:00–4:30 | Re-run the *same* file | Point out the "Translation Memory — Recent Jobs" table and that the second run is faster — cache reuse, not re-translating identical content. |
| 4:30–5:00 | Wrap | Mention offline/USB delivery: everything under `jobs/<job_id>/output/` is a plain file, no server needed to view it later. |

## Fallback plan

If live processing is too slow for the room (e.g. a longer clip, cold
cache, or a flaky demo laptop):

- Skip straight to the **pre-generated output** from the pre-warm run —
  play the MP4s, open the DOCX report, show the SRT — and narrate the
  upload step without waiting for it to finish live.
- If asked "how long does this actually take," give the real number from
  your dry run rather than guessing live.
- If a job errors on stage, `jobs/pipeline.log` has the traceback — don't
  try to debug live; switch to the fallback output and mention it's logged
  for follow-up.

## Edge cases worth having answers for (from the manual testing checklist)

- **Video with embedded subtitles**: ASR is skipped entirely — much
  faster. Worth mentioning if a judge asks about handling already-captioned
  content.
- **Noisy/quiet source audio**: the Job Report flags a denoise warning
  automatically (SNR-based) rather than silently producing a bad
  transcript.
- **A language with subtitles but no voiceover model yet** (9 of the 22 —
  see the UI's grouped dropdown): the job still completes; voiceover is
  skipped with a clear warning instead of failing.
