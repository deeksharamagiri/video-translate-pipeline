const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dzContent = document.querySelector('.dz-content');
const dzFile = document.getElementById('dzFile');
const dzFileName = document.getElementById('dzFileName');
const dzClear = document.getElementById('dzClear');
const submitBtn = document.getElementById('submitBtn');
const formError = document.getElementById('formError');

const pipelinePanel = document.getElementById('pipelinePanel');
const progressBarInner = document.getElementById('progressBarInner');
const progressMsg = document.getElementById('progressMsg');
const resultsPanel = document.getElementById('resultsPanel');
const statsRow = document.getElementById('statsRow');
const downloadsEl = document.getElementById('downloads');

let selectedFile = null;

// Maps backend progress "stage" values to the UI stage-track node ids.
const STAGE_ORDER = ['stage1', 'asr', 'stage3', 'translate', 'stage4', 'stage5', 'done'];
const STAGE_ALIASES = { skip_asr: 'asr', stage2: 'asr', burned_in: 'done', voiceover: 'done', queued: 'stage1' };

function resolveStageId(rawStage) {
  return STAGE_ALIASES[rawStage] || rawStage;
}

function highlightStage(rawStage) {
  const resolved = resolveStageId(rawStage);
  const idx = STAGE_ORDER.indexOf(resolved);
  document.querySelectorAll('.stage').forEach(el => {
    const stId = el.dataset.stage;
    const stIdx = STAGE_ORDER.indexOf(stId);
    el.classList.remove('active', 'complete');
    if (stIdx < idx) el.classList.add('complete');
    else if (stIdx === idx) el.classList.add('active');
  });
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});
dzClear.addEventListener('click', (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = '';
  dzContent.hidden = false;
  dzFile.hidden = true;
  submitBtn.disabled = true;
});

function setFile(file) {
  selectedFile = file;
  dzFileName.textContent = `${file.name}  (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
  dzContent.hidden = true;
  dzFile.hidden = false;
  submitBtn.disabled = false;
  formError.hidden = true;
}

submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  formError.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = 'Uploading...';

  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('source_lang', document.getElementById('sourceLang').value);
  fd.append('target_lang', document.getElementById('targetLang').value);
  fd.append('burned_in', document.getElementById('wantBurnedIn').checked ? 'true' : 'false');
  fd.append('voiceover', document.getElementById('wantVoiceover').checked ? 'true' : 'false');
  fd.append('engine', document.getElementById('engineChoice').value);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed.');

    pipelinePanel.hidden = false;
    resultsPanel.hidden = true;
    submitBtn.textContent = 'Processing...';
    pollStatus(data.job_id);
  } catch (err) {
    formError.textContent = err.message;
    formError.hidden = false;
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run Pipeline';
  }
});

function pollStatus(jobId) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${jobId}`);
      const data = await res.json();

      if (data.progress) {
        progressBarInner.style.width = `${data.progress.pct}%`;
        progressMsg.textContent = data.progress.message;
        highlightStage(data.progress.stage);
      }

      if (data.error) {
        clearInterval(interval);
        progressMsg.textContent = `Error: ${data.error}`;
        progressMsg.style.color = 'var(--red)';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Run Pipeline';
        return;
      }

      if (data.result) {
        clearInterval(interval);
        renderResults(jobId, data.result);
        submitBtn.disabled = false;
        submitBtn.textContent = 'Run Pipeline';
        loadArchive();
      }
    } catch (err) {
      clearInterval(interval);
      progressMsg.textContent = `Connection error: ${err.message}`;
    }
  }, 1200);
}

const DOWNLOAD_META = {
  srt: { key: 'srt', name: 'Subtitles (.srt)', desc: 'Standard subtitle file' },
  vtt: { key: 'vtt', name: 'Subtitles (.vtt)', desc: 'Web subtitle format' },
  job_report_docx: { key: 'report', name: 'Job Report (.docx)', desc: 'Transcript + QC flags' },
  burned_in_mp4: { key: 'burned_in', name: 'Burned-in Video (.mp4)', desc: 'Subtitles hard-coded' },
  voiceover_mp4: { key: 'voiceover', name: 'Voiceover Video (.mp4)', desc: 'Dubbed audio track' },
};

function renderResults(jobId, result) {
  resultsPanel.hidden = false;
  const s = result.stats;

  statsRow.innerHTML = `
    <div class="stat-chip">SRC <b>${s.source_lang}</b></div>
    <div class="stat-chip">TGT <b>${s.target_lang}</b></div>
    <div class="stat-chip">ENGINE <b>${s.engine}</b></div>
    <div class="stat-chip">SEGMENTS <b>${s.segment_count}</b></div>
    <div class="stat-chip">CACHE HITS <b>${s.cache_hit_count}</b></div>
    <div class="stat-chip">DENOISE <b>${s.preprocess.denoise_applied ? 'YES' : 'no'}</b></div>
  `;

  downloadsEl.innerHTML = '';
  result.available_downloads.forEach(key => {
    const meta = DOWNLOAD_META[key];
    if (!meta) return;
    const card = document.createElement('div');
    card.className = 'dl-card';
    card.innerHTML = `
      <div class="dl-name">${meta.name}</div>
      <div class="dl-desc">${meta.desc}</div>
      <a href="/api/download/${jobId}/${meta.key}" download>Download</a>
    `;
    downloadsEl.appendChild(card);
  });
}

async function loadArchive() {
  try {
    const res = await fetch('/api/archive');
    const data = await res.json();
    const el = document.getElementById('archiveTable');
    if (!data.jobs || !data.jobs.length) {
      el.innerHTML = '<p class="dim">No jobs archived yet.</p>';
      return;
    }
    let html = `<table><thead><tr>
      <th>Job</th><th>Src→Tgt</th><th>Segments</th><th>Cache hit %</th><th>When</th>
    </tr></thead><tbody>`;
    data.jobs.forEach(j => {
      html += `<tr>
        <td>${j.job_id}</td>
        <td>${j.source_lang} → ${j.target_lang}</td>
        <td>${j.segment_count}</td>
        <td>${j.cache_hit_pct}%</td>
        <td>${j.created_at}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    el.innerHTML = html;
  } catch (err) {
    // silent - archive dashboard is a nice-to-have
  }
}

loadArchive();
