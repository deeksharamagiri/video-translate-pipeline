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
  submitBtn.innerHTML = '<span class="btn-icon">⏳</span> Uploading...';

  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('source_lang', '');
  fd.append('target_lang', document.getElementById('targetLang').value);
  fd.append('burned_in', 'true');
  fd.append('voiceover', 'true');
  fd.append('engine', 'auto');

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed.');

    pipelinePanel.hidden = false;
    resultsPanel.hidden = true;
    submitBtn.innerHTML = '<span class="btn-icon">🔄</span> Processing...';
    pollStatus(data.job_id);
  } catch (err) {
    formError.textContent = err.message;
    formError.hidden = false;
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="btn-icon">🚜</span> Run Pipeline';
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
        pipelinePanel.hidden = true;
        renderResults(jobId, data.result);
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span class="btn-icon">🚜</span> Run Pipeline';
      }
    } catch (err) {
      clearInterval(interval);
      progressMsg.textContent = `Connection error: ${err.message}`;
    }
  }, 1200);
}

const DOWNLOAD_META = {
  vtt: { key: 'vtt', icon: '📝', name: 'Subtitles (.vtt)', desc: 'Web subtitle format' },
  voiceover_mp4: { key: 'voiceover', icon: '🎙️', name: 'Voiceover Video (.mp4)', desc: 'Dubbed audio track' },
};

const LANG_NAMES = {
  hin: 'Hindi', eng: 'English', mar: 'Marathi', ben: 'Bengali', tam: 'Tamil',
  tel: 'Telugu', kan: 'Kannada', mal: 'Malayalam', guj: 'Gujarati',
  pan: 'Punjabi', urd: 'Urdu', asm: 'Assamese', mai: 'Maithili',
};

function renderResults(jobId, result) {
  resultsPanel.hidden = false;
  const s = result.stats;
  const srcName = LANG_NAMES[s.source_lang] || s.source_lang;
  const tgtName = LANG_NAMES[s.target_lang] || s.target_lang;

  statsRow.innerHTML = `
    <div class="stat-chip"><span class="chip-icon">🌐</span> TRANSLATED <b>${srcName} → ${tgtName}</b></div>
  `;

  downloadsEl.innerHTML = '';
  result.available_downloads.forEach(key => {
    const meta = DOWNLOAD_META[key];
    if (!meta) return;
    const card = document.createElement('div');
    card.className = 'dl-card';
    card.innerHTML = `
      <div class="dl-name"><span class="chip-icon">${meta.icon}</span> ${meta.name}</div>
      <div class="dl-desc">${meta.desc}</div>
      <a href="/api/download/${jobId}/${meta.key}" download>Download</a>
    `;
    downloadsEl.appendChild(card);
  });
}
