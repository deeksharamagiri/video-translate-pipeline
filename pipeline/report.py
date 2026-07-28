"""
Generates the "SRT + VTT + Job Report" always-generated output:
  Full transcript, translation, QC flags for low-confidence segments, DOCX format.
"""
from typing import List

from docx import Document
from docx.shared import Pt, RGBColor

from pipeline.stage3_segment_tm import Segment

LOW_CONFIDENCE_THRESHOLD = 0.55


def generate_job_report(job_id: str, segments: List[Segment], source_lang: str,
                         target_lang: str, engine_name: str, preprocess_info: dict,
                         out_docx_path: str) -> str:
    doc = Document()

    doc.add_heading("Video/Audio Translation — Job Report", level=1)

    meta = doc.add_paragraph()
    meta.add_run(f"Job ID: ").bold = True
    meta.add_run(f"{job_id}\n")
    meta.add_run(f"Source language: ").bold = True
    meta.add_run(f"{source_lang}\n")
    meta.add_run(f"Target language: ").bold = True
    meta.add_run(f"{target_lang}\n")
    meta.add_run(f"Translation engine: ").bold = True
    meta.add_run(f"{engine_name}\n")
    meta.add_run(f"Total segments: ").bold = True
    meta.add_run(f"{len(segments)}\n")
    low_conf = [s for s in segments if s.confidence < LOW_CONFIDENCE_THRESHOLD]
    meta.add_run(f"Low-confidence segments flagged: ").bold = True
    meta.add_run(f"{len(low_conf)}")

    doc.add_heading("Pre-Processing Notes", level=2)
    p = doc.add_paragraph()
    p.add_run(f"Input type: {preprocess_info.get('input_kind')}\n")
    p.add_run(f"Subtitle track found: {preprocess_info.get('subtitle_track_found')}\n")
    p.add_run(f"Denoise applied: {preprocess_info.get('denoise_applied')}\n")
    if preprocess_info.get("snr_db") is not None:
        p.add_run(f"Estimated SNR: {preprocess_info.get('snr_db')} dB\n")
    for w in preprocess_info.get("warnings", []):
        warn_p = doc.add_paragraph()
        run = warn_p.add_run(f"⚠ {w}")
        run.font.color.rgb = RGBColor(0xC0, 0x50, 0x00)

    doc.add_heading("Full Transcript & Translation", level=2)

    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text = "#"
    hdr[1].text = "Time"
    hdr[2].text = "Source Text"
    hdr[3].text = "Translated Text"
    hdr[4].text = "Confidence / Flag"

    for i, seg in enumerate(segments, start=1):
        row = table.add_row().cells
        row[0].text = str(i)
        row[1].text = f"{_fmt_time(seg.start)} - {_fmt_time(seg.end)}"
        row[2].text = seg.text
        row[3].text = seg.translated_text or ""
        flag = "⚠ LOW CONFIDENCE" if seg.confidence < LOW_CONFIDENCE_THRESHOLD else "OK"
        cache_note = " (cache)" if seg.from_cache else ""
        row[4].text = f"{seg.confidence:.2f} — {flag}{cache_note}"

    doc.save(out_docx_path)
    return out_docx_path


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
