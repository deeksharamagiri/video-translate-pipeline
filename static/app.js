const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dzContent = document.querySelector('.dz-content');
const dzFile = document.getElementById('dzFile');
const dzFileName = document.getElementById('dzFileName');
const dzClear = document.getElementById('dzClear');
const submitBtn = document.getElementById('submitBtn');
const cancelBtn = document.getElementById('cancelBtn');
const formError = document.getElementById('formError');

const progressBarInner = document.getElementById('progressBarInner');
const progressMsg = document.getElementById('progressMsg');
const progressPct = document.getElementById('progressPct');
const resultsPanel = document.getElementById('resultsPanel');
const statsRow = document.getElementById('statsRow');
const downloadsEl = document.getElementById('downloads');

let selectedFile = null;
let currentJobId = null;
let isCancelling = false;


// ============================================================
// PAGE REFRESH / LEAVE PROTECTION
// ============================================================
//
// While a translation is running, the browser will ask the
// user for confirmation if they:
//
//   - Refresh the page
//   - Close the tab
//   - Close the browser window
//   - Navigate away from the page
//   - Press Back / Forward
//
// The browser controls the exact wording of the popup.
//
// The warning is ONLY enabled after the backend has accepted
// the translation job.
//
// It is disabled when the job:
//
//   - Completes
//   - Is cancelled
//   - Fails
// ============================================================

let jobIsRunning = false;


// ------------------------------------------------------------
// Enable warning
// ------------------------------------------------------------

function enablePageLeaveWarning() {

  jobIsRunning = true;

}


// ------------------------------------------------------------
// Disable warning
// ------------------------------------------------------------

function disablePageLeaveWarning() {

  jobIsRunning = false;

}


// ------------------------------------------------------------
// Browser refresh / close / navigation handler
// ------------------------------------------------------------

window.addEventListener(
  'beforeunload',
  (event) => {

    if (!jobIsRunning) {
      return;
    }

    // Required by modern browsers.
    //
    // The browser will display its own native confirmation
    // dialog. Custom text is intentionally not supplied because
    // modern browsers ignore custom beforeunload messages.
    event.preventDefault();

    event.returnValue = '';

  }
);


// ============================================================
// Reset the run button / cancel button back to idle
// ============================================================

function resetRunUI() {

  submitBtn.disabled = false;

  submitBtn.classList.remove(
    'is-processing'
  );

  cancelBtn.hidden = true;

  cancelBtn.disabled = false;

  currentJobId = null;

  isCancelling = false;

}


// ============================================================
// File selection
// ============================================================

dropzone.addEventListener(
  'click',
  () => fileInput.click()
);


dropzone.addEventListener(
  'dragover',
  (e) => {

    e.preventDefault();

    dropzone.classList.add(
      'dragover'
    );

  }
);


dropzone.addEventListener(
  'dragleave',
  () => {

    dropzone.classList.remove(
      'dragover'
    );

  }
);


dropzone.addEventListener(
  'drop',
  (e) => {

    e.preventDefault();

    dropzone.classList.remove(
      'dragover'
    );

    if (
      e.dataTransfer.files.length
    ) {

      setFile(
        e.dataTransfer.files[0]
      );

    }

  }
);


fileInput.addEventListener(
  'change',
  () => {

    if (
      fileInput.files.length
    ) {

      setFile(
        fileInput.files[0]
      );

    }

  }
);


// ============================================================
// Clear selected file
// ============================================================

dzClear.addEventListener(
  'click',
  (e) => {

    e.stopPropagation();

    selectedFile = null;

    fileInput.value = '';

    dzContent.hidden = false;

    dzFile.hidden = true;

    submitBtn.disabled = true;

  }
);


// ============================================================
// Set selected file
// ============================================================

function setFile(file) {

  selectedFile = file;

  dzFileName.textContent =
    `${file.name}  (${(
      file.size /
      (1024 * 1024)
    ).toFixed(1)} MB)`;

  dzContent.hidden = true;

  dzFile.hidden = false;

  submitBtn.disabled = false;

  formError.hidden = true;

}


// ============================================================
// Start processing
// ============================================================

submitBtn.addEventListener(
  'click',
  async () => {

    if (!selectedFile) {
      return;
    }

    formError.hidden = true;

    submitBtn.disabled = true;

    submitBtn.classList.add(
      'is-processing'
    );

    progressMsg.textContent =
      'Uploading...';

    progressPct.textContent =
      '0%';

    progressBarInner.style.width =
      '0%';


    const fd =
      new FormData();


    fd.append(
      'file',
      selectedFile
    );


    fd.append(
      'source_lang',
      ''
    );


    fd.append(
      'target_lang',
      document.getElementById(
        'targetLang'
      ).value
    );


    // --------------------------------------------------------
    // Always request both required outputs.
    // --------------------------------------------------------

    fd.append(
      'burned_in',
      'true'
    );


    fd.append(
      'voiceover',
      'true'
    );


    fd.append(
      'engine',
      'auto'
    );


    fd.append(
      'asr_engine',
      'indic_conformer'
    );


    try {

      const res =
        await fetch(
          '/api/upload',
          {
            method: 'POST',
            body: fd,
          }
        );


      const data =
        await res.json();


      if (!res.ok) {

        throw new Error(
          data.error ||
          'Upload failed.'
        );

      }


      // ------------------------------------------------------
      // IMPORTANT:
      //
      // The job now exists on the Flask backend.
      //
      // From this point onward, warn the user before they
      // refresh or leave the page.
      // ------------------------------------------------------

      enablePageLeaveWarning();


      resultsPanel.hidden =
        true;


      currentJobId =
        data.job_id;


      cancelBtn.hidden =
        false;


      cancelBtn.disabled =
        false;


      pollStatus(
        data.job_id
      );


    } catch (err) {

      // The upload failed, so no backend job is running.
      disablePageLeaveWarning();


      formError.textContent =
        err.message;


      formError.hidden =
        false;


      resetRunUI();

    }

  }
);


// ============================================================
// Cancel a running job
// ============================================================

cancelBtn.addEventListener(
  'click',
  async () => {

    if (!currentJobId) {
      return;
    }


    cancelBtn.disabled =
      true;


    isCancelling =
      true;


    progressMsg.textContent =
      'Cancelling — stopping the running step...';


    try {

      const res =
        await fetch(
          `/api/cancel/${currentJobId}`,
          {
            method: 'POST'
          }
        );


      const data =
        await res.json();


      // ------------------------------------------------------
      // If the backend successfully accepted the cancellation
      // request, keep the page-leave warning enabled until
      // the status endpoint confirms that the job is actually
      // cancelled.
      // ------------------------------------------------------

      if (!res.ok) {

        throw new Error(
          data.error ||
          'Unable to cancel the job.'
        );

      }


    } catch (err) {

      // Cancellation request itself failed.
      //
      // The job may still be running, so KEEP the page-leave
      // warning enabled.
      isCancelling = false;

      cancelBtn.disabled =
        false;

      formError.textContent =
        `Cancel error: ${err.message}`;

      formError.hidden =
        false;

    }

  }
);


// ============================================================
// Poll job status
// ============================================================

function pollStatus(jobId) {

  const interval =
    setInterval(
      async () => {

        try {

          const res =
            await fetch(
              `/api/status/${jobId}`
            );


          const data =
            await res.json();


          // --------------------------------------------------
          // Progress
          // --------------------------------------------------

          if (
            data.progress &&
            !isCancelling
          ) {

            progressBarInner.style.width =
              `${data.progress.pct}%`;


            progressMsg.textContent =
              data.progress.message;


            progressPct.textContent =
              `${data.progress.pct}%`;

          }


          // --------------------------------------------------
          // Cancelled
          // --------------------------------------------------

          if (data.cancelled) {

            clearInterval(
              interval
            );


            // The backend confirmed that the pipeline
            // has stopped.
            disablePageLeaveWarning();


            formError.textContent =
              'Pipeline cancelled.';


            formError.hidden =
              false;


            resetRunUI();


            return;

          }


          // --------------------------------------------------
          // Error
          // --------------------------------------------------

          if (data.error) {

            clearInterval(
              interval
            );


            // The backend reported a terminal error.
            disablePageLeaveWarning();


            formError.textContent =
              `Error: ${data.error}`;


            formError.hidden =
              false;


            resetRunUI();


            return;

          }


          // --------------------------------------------------
          // Completed
          // --------------------------------------------------

          if (data.result) {

            clearInterval(
              interval
            );


            // The backend confirmed completion.
            // Refreshing is now safe.
            disablePageLeaveWarning();


            renderResults(
              jobId,
              data.result
            );


            resetRunUI();


            return;

          }

        } catch (err) {

          clearInterval(
            interval
          );


          // --------------------------------------------------
          // IMPORTANT:
          //
          // DO NOT disable the page-leave warning here.
          //
          // A network/polling error does NOT prove that the
          // backend pipeline stopped. The Flask worker could
          // still be processing the video.
          //
          // Therefore we keep jobIsRunning = true.
          // --------------------------------------------------

          formError.textContent =
            `Connection error: ${err.message}`;


          formError.hidden =
            false;


          submitBtn.disabled =
            false;


          submitBtn.classList.remove(
            'is-processing'
          );


          cancelBtn.hidden =
            false;


          cancelBtn.disabled =
            false;


          // Keep currentJobId so the user can continue
          // attempting to interact with the running job.


          return;

        }

      },
      1200
    );

}


// ============================================================
// Dashboard download metadata
// ============================================================

/*
 * IMPORTANT
 *
 * These are the ONLY files displayed on the dashboard.
 *
 * The pipeline may generate:
 *
 *   SRT
 *   VTT
 *   PDF
 *   DOCX
 *   voiceover MP4
 *   burned-in MP4
 *
 * But only the following two are user-facing.
 */

const DOWNLOAD_META = {

  burned_in_mp4: {

    key: 'burned_in',

    icon: '🎬',

    name:
      'Burned-in Voiceover + Subtitles (.mp4)',

    desc:
      'Translated voiceover with subtitles burned into the video',

  },


  subtitles_pdf: {

    key: 'subtitles_pdf',

    icon: '📄',

    name:
      'Subtitle PDF (.pdf)',

    desc:
      'Translated subtitles with timestamps',

  },

};


// ============================================================
// Required dashboard order
// ============================================================

const DOWNLOAD_ORDER = [

  'burned_in_mp4',

  'subtitles_pdf',

];


// ============================================================
// Language names
// ============================================================

const LANG_NAMES = {

  hin: 'Hindi',

  eng: 'English',

  mar: 'Marathi',

  ben: 'Bengali',

  tam: 'Tamil',

  tel: 'Telugu',

  kan: 'Kannada',

  mal: 'Malayalam',

  guj: 'Gujarati',

  pan: 'Punjabi',

  urd: 'Urdu',

  asm: 'Assamese',

  mai: 'Maithili',

  nep: 'Nepali',

  san: 'Sanskrit',

  ori: 'Odia',

  fra: 'French',

  spa: 'Spanish',

  deu: 'German',

  zho: 'Chinese',

  ara: 'Arabic',

  por: 'Portuguese',

  rus: 'Russian',

  jpn: 'Japanese',

};


// ============================================================
// Render results
// ============================================================

function renderResults(
  jobId,
  result
) {

  resultsPanel.hidden =
    false;


  const s =
    result.stats;


  const srcName =
    LANG_NAMES[s.source_lang] ||
    s.source_lang;


  const tgtName =
    LANG_NAMES[s.target_lang] ||
    s.target_lang;


  // ==========================================================
  // Translation summary
  // ==========================================================

  statsRow.innerHTML = `

    <div class="stat-chip">

      <span class="chip-icon">
        🌐
      </span>

      TRANSLATED

      <b>
        ${srcName} → ${tgtName}
      </b>

    </div>

  `;


  // ==========================================================
  // Quality warnings are intentionally not shown here.
  // ==========================================================

  /*
   * Quality warnings remain in:
   *
   *   pipeline.log
   *
   * and the generated Job Report DOCX.
   *
   * They are intentionally not shown in the
   * user-facing dashboard.
   */


  // ==========================================================
  // Clear previous download cards
  // ==========================================================

  downloadsEl.innerHTML =
    '';


  // ==========================================================
  // Render ONLY the two approved downloads
  // ==========================================================

  DOWNLOAD_ORDER.forEach(
    (key) => {

      // ------------------------------------------------------
      // Do not render a file that the backend says
      // does not exist.
      // ------------------------------------------------------

      if (
        !result.available_downloads ||
        !result.available_downloads.includes(
          key
        )
      ) {

        return;

      }


      const meta =
        DOWNLOAD_META[key];


      if (!meta) {
        return;
      }


      // ------------------------------------------------------
      // Create card
      // ------------------------------------------------------

      const card =
        document.createElement(
          'div'
        );


      card.className =
        'dl-card';


      card.innerHTML = `

        <div class="dl-name">

          <span class="chip-icon">
            ${meta.icon}
          </span>

          ${meta.name}

        </div>


        <div class="dl-desc">

          ${meta.desc}

        </div>


        <a
          href="/api/download/${jobId}/${meta.key}"
          download
        >
          Download
        </a>

      `;


      downloadsEl.appendChild(
        card
      );

    }
  );

}
