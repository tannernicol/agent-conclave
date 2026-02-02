const statusEl = document.getElementById('status');
const smokeEl = document.getElementById('smoke');
const latestEl = document.getElementById('latest');
const runsEl = document.getElementById('runs');
const form = document.getElementById('query-form');
const rerunBtn = document.getElementById('rerun-latest');

let currentRunId = null;
let pollTimer = null;

async function fetchJSON(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status}`);
  }
  return await resp.json();
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setSmoke(active) {
  if (active) {
    smokeEl.classList.add('active');
  } else {
    smokeEl.classList.remove('active');
  }
}

function renderLatest(latest) {
  if (!latest || !latest.consensus) {
    latestEl.textContent = 'No consensus yet.';
    return;
  }
  latestEl.textContent = latest.consensus.answer || 'No consensus yet.';
}

function renderRuns(runs) {
  runsEl.innerHTML = '';
  runs.forEach((run) => {
    const li = document.createElement('li');
    const status = run.status || 'unknown';
    const pope = run.consensus && run.consensus.pope ? ` - ${run.consensus.pope}` : '';
    li.textContent = `${run.id} [${status}]${pope}`;
    runsEl.appendChild(li);
  });
}

async function refresh() {
  try {
    const status = await fetchJSON('/api/status');
    renderLatest(status.latest);
  } catch (err) {
    console.error(err);
  }
  try {
    const runs = await fetchJSON('/api/runs?limit=5');
    renderRuns(runs.runs || []);
  } catch (err) {
    console.error(err);
  }
}

async function startRun(query) {
  setStatus('Locked doors. Deliberation in progress...');
  setSmoke(false);
  const inputTitle = document.getElementById('input-title').value.trim();
  const inputNotes = document.getElementById('input-notes').value.trim();
  let inputId = null;
  if (inputNotes) {
    const inputResp = await fetchJSON('/api/inputs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: inputTitle, content: inputNotes, question: query }),
    });
    inputId = inputResp.input_id;
  }
  const payload = { query };
  if (inputId) {
    payload.input_id = inputId;
  }
  const resp = await fetchJSON('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  currentRunId = resp.run_id;
  if (!currentRunId) {
    setStatus('Unable to start run.');
    return;
  }
  pollRun(currentRunId);
}

async function pollRun(runId) {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
  pollTimer = setInterval(async () => {
    try {
      const run = await fetchJSON(`/api/runs/${runId}`);
      if (run.status === 'complete') {
        clearInterval(pollTimer);
        setSmoke(true);
        setStatus('White smoke. Consensus reached.');
        renderLatest(run);
        refresh();
      } else if (run.status === 'failed') {
        clearInterval(pollTimer);
        setStatus(`Conclave failed: ${run.error || 'unknown error'}`);
      } else {
        setStatus('Deliberating behind locked doors...');
      }
    } catch (err) {
      console.error(err);
    }
  }, 2000);
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const query = document.getElementById('query').value.trim();
  if (!query) return;
  startRun(query);
});

rerunBtn.addEventListener('click', async () => {
  try {
    const latest = await fetchJSON('/api/runs/latest');
    if (!latest || !latest.query) {
      setStatus('No prior consensus to re-run.');
      return;
    }
    const payload = { query: latest.query };
    if (latest.meta && latest.meta.input_path) {
      payload.input_path = latest.meta.input_path;
    }
    setStatus('Locked doors. Deliberation in progress...');
    setSmoke(false);
    const resp = await fetchJSON('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    currentRunId = resp.run_id;
    if (!currentRunId) {
      setStatus('Unable to start run.');
      return;
    }
    pollRun(currentRunId);
  } catch (err) {
    console.error(err);
  }
});

refresh();
