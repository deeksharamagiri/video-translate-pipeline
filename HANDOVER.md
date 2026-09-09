# System Handover Document

| | |
|---|---|
| **System name** | Offline Field Translator (Video/Audio Translation Pipeline) |
| **Prepared for** | BAIF |
| **Document type** | Operational Handover & Runbook |
| **Version** | 1.0 |
| **Status** | Ready for handover |

This document is written for whoever takes ownership of operating and
supporting this system after the build team hands it off. It assumes no
prior involvement in building the system — only familiarity with running
software on a laptop. Setup instructions and a guided demo walkthrough
exist as separate companion documents (the Setup Guide and the Demo
Script); this document covers everything needed to **own, run, and
support** the system day to day.

---

## 1. Purpose & Scope

The system converts a field-recorded video or audio file into translated,
usable outputs for outreach and training purposes — without requiring an
internet connection at the point of use. It was built to remove the
dependency on external translation services or connectivity when working
in the field.

**In scope:**
- Translating spoken content from a recording into any of 22 Indian
  languages.
- Producing subtitles, a dubbed voiceover, a captioned video, and a
  written transcript from a single upload.
- Running entirely on one laptop, with no ongoing internet requirement
  after initial setup.

**Out of scope:**
- Multi-user or networked operation (see Section 3).
- Real-time/live translation (the system processes recorded files only).
- Human review or certification of translation quality — outputs are
  machine-generated and should be spot-checked per Section 7.

---

## 2. System Overview

A user uploads a video or audio recording through a simple web page
running on their own laptop, selects a target language, and starts the
job. The system then automatically:

1. Extracts and cleans the audio (removing background noise where needed).
2. Transcribes the spoken content, with timing for each phrase.
3. Translates the transcript into the selected language, using
   translation memory to avoid re-translating repeated content across
   jobs.
4. Produces:
   - Subtitle files (for use with standard video players),
   - A written transcript document,
   - A video with subtitles burned directly into the picture,
   - A dubbed voiceover track replacing the original audio,
   - A quality report summarising confidence levels and any issues found.

A job typically completes in a few minutes, depending on the length of
the recording and the machine's processing power. Every job's outputs are
saved to that machine's local storage and are ready for transfer to a USB
drive or shared folder for distribution.

---

## 3. Ownership & Responsibility Model

- **Deployment unit**: one installation per laptop. There is no shared
  server, central database, or cloud component — each machine operates
  independently.
- **Users**: designed for a single operator per machine, working through
  the web page in a browser on that same laptop. It is not intended to
  be accessed by other devices over a network or the internet.
- **Access control**: there are no user accounts or logins. Anyone with
  access to the laptop can use the tool. If this needs to change (e.g.
  shared office use), that is a configuration change the technical
  point of contact can make — see Section 9.
- **Data ownership**: all translation history and output files reside
  only on the machine that produced them. Nothing is transmitted
  elsewhere during normal operation.

---

## 4. How the System Works (Architecture Summary)

For anyone who needs a working mental model without technical depth, the
system processes each job through five stages:

| Stage | What happens |
|---|---|
| 1. Preparation | Audio is extracted from the upload and cleaned up (noise reduction applied automatically if the recording is unclear). |
| 2. Transcription | Spoken words are converted to timestamped text. |
| 3. Translation | Text is translated into the target language, reusing previously translated phrases where possible. |
| 4. Output generation | Subtitles, transcript, dubbed audio, and captioned video are produced. |
| 5. Quality check & archive | Each job is scored for translation confidence and completeness, and the translated content is saved for reuse in future jobs. |

Every job also produces a **Quality Report** (a document) summarising how
confident the system was in the transcription and translation, and
flagging anything that may need manual review — for example, a noisy
section of audio, or a phrase the system could not translate. This report
is the first place to check when assessing whether a specific job's
output can be trusted as-is or should be reviewed by a human before
distribution.

---

## 5. Operations & Support Runbook

**Where to look when something goes wrong, in order:**

1. **The activity log** — a running record of every job (start, finish,
   failure) with full technical detail on any failure. This is the
   single most useful place to diagnose an issue.
2. **The terminal/console window** the application is running in — shows
   the same information live, plus a timing breakdown for whichever job
   most recently ran.
3. **The Quality Report** for a specific job — per-job detail on
   confidence scores and warnings, useful when a specific output looks
   wrong but the system otherwise ran fine.

**Common issues and how to resolve them:**

| Symptom | Likely cause | What to do |
|---|---|---|
| A job fails immediately with an access/permission error | One of the translation models requires a one-time, free account approval that hasn't been completed on this machine yet | Follow the one-time model access step in the Setup Guide |
| Dubbed voiceover is skipped, with a warning in the report | The selected target language doesn't yet have a voice available | Expected for some of the 22 supported languages; subtitles are still produced normally. Additional voices can be added later — see Section 9 |
| Voiceover fails outright rather than being skipped | The voiceover component hasn't been set up on this machine | Complete the one-time voiceover setup step in the Setup Guide |
| A file is rejected on upload | The file exceeds the configured size or length limit | Limits can be raised by the technical point of contact if field recordings are routinely longer |
| A job fails with a technical error | Check the activity log for the specific cause | The system automatically removes any partial/incomplete output from a failed job — see "Rollback behaviour" below. It is always safe to simply try again |
| The application won't start at all | Underlying software environment issue | Re-run the initial setup steps in the Setup Guide |

**Rollback behaviour:** if any stage of a job fails for any reason, the
system automatically deletes that job's partial output rather than
leaving an incomplete or corrupted result behind that could be mistaken
for a finished one. The failure is fully recorded in the activity log
before anything is removed, so the cause is never lost — only the
unfinished files are cleaned up.

**Cancelling a job:** an in-progress job can be stopped manually; the
system responds within about a second and cleans up any partial output
the same way a failed job would be cleaned up.

**Restarting the application:** the system holds no state in memory
between restarts beyond what is already saved to disk (translation
history and completed job outputs). Stopping and restarting the
application is always safe. Any job that was actively running at the
moment of a restart is lost and should simply be re-run — it will not
resume automatically.

---

## 6. Data Management

- **What is stored locally**: a translation history/cache (so repeated
  phrases across jobs translate faster and more cheaply over time), a
  glossary of protected terms (Section 9), and the output files for
  every job that has been run.
- **Backing up**: to preserve translation history and glossary terms,
  back up the system's data folder. To also preserve past job outputs
  for record-keeping, back up the jobs folder as well.
- **Migrating to a new machine**: the data and job folders above can be
  copied directly to a new installation. The underlying language models
  do not need to be copied — they will download automatically the first
  time each is used on the new machine (see Section 8 for a fully
  offline alternative that avoids this).
- **Retention**: there is no automatic deletion of old job output. If
  disk space becomes a concern over time, older job folders can be
  archived or removed manually — this does not affect the translation
  history cache.

---

## 7. Known Limitations & Risks

- **Machine-generated output requires spot-checking.** Translation and
  transcription quality vary by recording clarity, background noise, and
  language. The per-job Quality Report should be reviewed before
  distributing sensitive or high-stakes content, particularly where the
  report flags low confidence.
- **No built-in human review workflow.** Approving content for
  distribution is a manual process today; this system does not gate or
  route outputs for sign-off.
- **Single-operator design.** The system is not built for multiple people
  to use the same installation concurrently over a network.
- **Some languages have partial voice support.** All 22 supported
  languages produce subtitles and transcripts; dubbed voiceover is
  currently available for a subset of them (see Section 9 to extend
  this).
- **First-time setup requires internet.** Ongoing day-to-day use does
  not, but the initial installation and first use of each language
  /feature needs a one-time internet connection to download the required
  components.

---

## 8. Deployment Options

Two ways to get the system running are available, chosen based on the
target machine:

1. **Standard installation** — for a machine that already has a general
   software development environment available. Setup takes a small
   number of steps, detailed in the Setup Guide, and downloads what it
   needs automatically on first use.
2. **Containerized offline package (Docker)** — for a machine with no
   existing development environment and no internet access at all,
   intended for field deployment. The entire system — including every
   speech-recognition, translation, and voice model it depends on — is
   packaged in advance into a single, self-contained **Docker container
   image** on a machine that has internet access. That image is then
   transferred to the target machine (e.g. via USB) and run there with
   Docker as the only prerequisite; no Python, no model downloads, and
   no internet connection are needed on the field machine itself.

Both options result in the same system with the same capabilities; the
choice only affects how it gets installed. The containerized package is
the recommended path for field deployment specifically because it
removes environment drift: the same image produces an identical, working
installation on every machine it's loaded onto, rather than depending on
what happens to already be installed there.

**What this means in practice for the Deployment criteria:**

| Aspect | Standard installation | Containerized (Docker) package |
|---|---|---|
| Target machine prerequisites | A general software development environment | Docker only |
| Internet required at deployment site | Yes, for first-time model downloads | No — fully offline |
| Setup time on target machine | A short list of setup steps | Load the pre-built image and start it — a matter of minutes |
| Repeatability | Consistent, but depends on the target machine's existing environment | Identical on every machine, by construction |
| Format | Source installation | A single portable image file, transferable on a USB drive or external disk |

This option is prepared and maintained by the technical point of contact
and is available as a separate packaged build on request.

---

## 9. Maintenance & Extensibility

The following can be done by a non-developer technical point of contact,
without any code changes or redeployment:

- **Adding a protected term** (e.g. an organisation name, a technical
  term that should never be freely translated): add it to the glossary
  data file with its exact translation for each language. Once added, it
  is guaranteed to be used consistently rather than left to the
  translation model's judgement.
- **Adding voiceover support for an additional language**: register the
  new language in the voice-language configuration; the corresponding
  voice data downloads automatically the first time it's used. Requesting
  voiceover for a language without a registered voice does not fail a
  job — it is skipped with a note in that job's Quality Report, and
  subtitles are still produced normally.
- **Raising upload size/length limits**: adjustable in the system's
  configuration file if field recordings routinely exceed the current
  limits.

Anything beyond the above (e.g. adding a new pipeline stage, changing
core translation logic) should go back to the build/technical team.

---

## 10. Adoption Plan for BAIF

1. **Pilot phase** — run 3–5 real field recordings through the system
   using the manual testing checklist in the Setup Guide before wider
   rollout, so that any language- or content-specific issues surface
   early, on known material.
2. **Deployment model** — one machine, one operator, per field site. The
   system is not designed for multiple concurrent operators sharing a
   single installation over a network.
3. **Operator training should cover**:
   - The upload → language selection → download workflow through the web
     page (the Demo Script provides a guided walkthrough). Note that the
     web page itself only ever offers two downloads — the captioned
     video and the subtitle PDF
   - How to open the job's output folder directly on the laptop
4. **Extending coverage over time** — adding a new voiceover language or
   a glossary term (Section 9) is a configuration change an operator's
   technical point of contact can make without any redeployment or
   specialist involvement.

---

## 11. Support Escalation Path

| Level | Who | When |
|---|---|---|
| 1 | Trained operator | Day-to-day use; reading Quality Report warnings; retrying a failed job |
| 2 | Site technical point of contact | Configuration changes (Section 9), setup issues, interpreting the activity log |
| 3 | Build/technical team | Failures not resolved by the activity log's guidance, or requests for new capabilities |

When escalating to Level 3, include: the job ID (if available), the
relevant section of the activity log, and the input file's format and
approximate length — this is normally enough to diagnose an issue without
needing the machine itself.

---

## Appendix A: Key File & Folder Reference

| Item | Location | Purpose |
|---|---|---|
| Translation history / glossary | System's data folder | Cache of past translations and protected terms |
| Job outputs | System's jobs folder, one subfolder per job | All generated files for a completed job |
| Activity log | Inside the jobs folder | Full job history and failure diagnostics |
| Per-job Quality Report | Inside each job's output subfolder | Confidence scores and warnings for that specific job |

## Appendix B: Glossary of Terms

| Term | Meaning |
|---|---|
| Job | One complete run of the system against a single uploaded file |
| Translation memory | The cache of previously translated phrases, reused to save time on repeat content |
| Quality Report | The per-job document summarising confidence and flagging issues |
| Voiceover | The dubbed audio track generated to replace the original spoken audio |
| Burned-in subtitles | Subtitles rendered directly into the video picture, rather than as a separate file |
| Rollback | Automatic removal of incomplete output after a failed job |
