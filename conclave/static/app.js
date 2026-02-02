const statusEl = document.getElementById('status');
const smokeEl = document.getElementById('smoke');
const latestEl = document.getElementById('latest');
const latestTimeEl = document.getElementById('latest-time');
const latestPromptEl = document.getElementById('latest-prompt');
const progressEl = document.getElementById('pipeline-progress');
const runsEl = document.getElementById('run-rows');
const form = document.getElementById('query-form');
const rerunBtn = document.getElementById('rerun-latest');
const clearBtn = document.getElementById('clear-form');
const savePromptBtn = document.getElementById('save-prompt');
const promptStatusEl = document.getElementById('prompt-status');
const promptListEl = document.getElementById('prompt-list');
const promptCountEl = document.getElementById('prompt-count');
const smokeStatusEl = document.getElementById('smoke-status');
const copyLatestBtn = document.getElementById('copy-latest');
const promptExamplesEl = document.getElementById('prompt-examples');

let currentRunId = null;
let pollTimer = null;
let currentPromptId = null;

async function fetchJSON(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status}`);
  }
  return await resp.json();
}

function setStatus(text) {
  statusEl.textContent = text;
  statusEl.setAttribute('aria-busy', text.toLowerCase().includes('running') ? 'true' : 'false');
}

function setPromptStatus(text) {
  if (promptStatusEl) {
    promptStatusEl.textContent = text;
  }
}

function setSmoke(active) {
  if (active) {
    smokeEl.classList.add('active');
    if (smokeStatusEl) smokeStatusEl.textContent = 'Consensus reached.';
  } else {
    smokeEl.classList.remove('active');
    if (smokeStatusEl) smokeStatusEl.textContent = 'No consensus in progress.';
  }
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function sanitizeUrl(url) {
  try {
    const parsed = new URL(url, window.location.origin);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.href;
    }
  } catch (err) {
    return '#';
  }
  return '#';
}

function renderInline(text) {
  let value = text;
  value = value.replace(/`([^`]+)`/g, '<code>$1</code>');
  value = value.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  value = value.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  value = value.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
    const safe = sanitizeUrl(url);
    return `<a href="${safe}" target="_blank" rel="noreferrer">${label}</a>`;
  });
  return value;
}

function renderMarkdown(text) {
  if (!text) {
    return '<div class="empty-state">No output yet.</div>';
  }
  const escaped = escapeHtml(text);
  const lines = escaped.split('\n');
  let html = '';
  let inCode = false;
  let inUl = false;
  let inOl = false;
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html += `<p>${renderInline(paragraph.join(' '))}</p>`;
      paragraph = [];
    }
  };

  const closeLists = () => {
    if (inUl) {
      html += '</ul>';
      inUl = false;
    }
    if (inOl) {
      html += '</ol>';
      inOl = false;
    }
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('```')) {
      if (inCode) {
        html += '</code></pre>';
        inCode = false;
      } else {
        flushParagraph();
        closeLists();
        inCode = true;
        html += '<pre><code>';
      }
      return;
    }

    if (inCode) {
      html += `${line}\n`;
      return;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      closeLists();
      const level = headingMatch[1].length;
      html += `<h${level}>${renderInline(headingMatch[2])}</h${level}>`;
      return;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (ulMatch) {
      flushParagraph();
      if (inOl) {
        html += '</ol>';
        inOl = false;
      }
      if (!inUl) {
        html += '<ul>';
        inUl = true;
      }
      html += `<li>${renderInline(ulMatch[1])}</li>`;
      return;
    }

    const olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
    if (olMatch) {
      flushParagraph();
      if (inUl) {
        html += '</ul>';
        inUl = false;
      }
      if (!inOl) {
        html += '<ol>';
        inOl = true;
      }
      html += `<li>${renderInline(olMatch[1])}</li>`;
      return;
    }

    if (!trimmed) {
      flushParagraph();
      closeLists();
      return;
    }
    paragraph.push(trimmed);
  });

  flushParagraph();
  closeLists();
  if (inCode) {
    html += '</code></pre>';
  }
  return html;
}

function renderEmptyState() {
  return `
    <div class="empty-state">
      <p>Ask a question about <strong>health</strong>, <strong>tax</strong>, <strong>money</strong>, or <strong>bounty</strong> to get a multi-model consensus.</p>
      <p class="ui-muted">Example: "What vitamins should I take daily?"</p>
    </div>
  `;
}

function formatTime(value) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function extractRecommendations(text, limit = 6) {
  if (!text) return [];
  const lines = text.split('\n');
  const recs = [];
  let inSection = false;
  let sectionFound = false;
  const sectionKeys = ['recommend', 'schedule', 'allocation', 'plan', 'summary', 'daily'];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    const lower = line.toLowerCase();
    if (line.startsWith('#')) {
      if (sectionKeys.some((key) => lower.includes(key))) {
        inSection = true;
        sectionFound = true;
        continue;
      }
      if (sectionFound && inSection) break;
      continue;
    }
    const bulletMatch = line.match(/^[-*•]\s+(.*)$/);
    const numMatch = line.match(/^\d+\.\s+(.*)$/);
    if (bulletMatch) {
      recs.push(bulletMatch[1]);
    } else if (numMatch) {
      recs.push(numMatch[1]);
    } else if (line.includes('|') && !line.includes('---')) {
      const parts = line.split('|').map((p) => p.trim()).filter(Boolean);
      if (parts.length >= 3 && parts[0] !== 'Time') {
        recs.push(`${parts[0]} — ${parts[1]} (${parts[2]})`);
      }
    } else if (!sectionFound && recs.length < 2) {
      recs.push(line);
    }
    if (recs.length >= limit) break;
  }
  return recs;
}

function renderRecommendations(text) {
  const recs = extractRecommendations(text);
  if (!recs.length) {
    return '<div class="empty-state">No recommendations yet.</div>';
  }
  return `<ul class="output-list">${recs.map((item) => `<li>${renderInline(escapeHtml(item))}</li>`).join('')}</ul>`;
}

function buildPhaseState(run) {
  const phases = [
    { key: 'route', label: 'Route' },
    { key: 'retrieve', label: 'RAG' },
    { key: 'reasoner', label: 'Reason' },
    { key: 'critic', label: 'Critic' },
    { key: 'summarizer', label: 'Summarize' },
  ];
  if (!run) return phases;
  const events = run.events || [];
  const done = new Set();
  events.forEach((event) => {
    if (event.phase === 'route' && event.status === 'done') done.add('route');
    if (event.phase === 'retrieve' && event.status === 'done') done.add('retrieve');
    if (event.phase === 'model' && event.role) done.add(event.role);
  });
  if (run.status === 'complete') {
    phases.forEach((phase) => done.add(phase.key));
  }
  let active = null;
  if (run.status === 'running') {
    active = phases.find((phase) => !done.has(phase.key))?.key || null;
  }
  return phases.map((phase) => ({
    ...phase,
    done: done.has(phase.key),
    active: active === phase.key,
  }));
}

function renderProgress(run) {
  if (!progressEl) return;
  if (!run || (!run.events && run.status !== 'running')) {
    progressEl.innerHTML = '';
    return;
  }
  const phases = buildPhaseState(run);
  progressEl.innerHTML = phases.map((phase) => {
    const classes = ['phase'];
    if (phase.done) classes.push('done');
    if (phase.active) classes.push('active');
    return `<span class="${classes.join(' ')}">${phase.label}</span>`;
  }).join('');
}

function renderEvidence(run) {
  const evidence = run?.artifacts?.context?.evidence || [];
  const collections = run?.artifacts?.route?.collections || [];
  if (!evidence.length) {
    return '<div class="empty-state">No evidence captured.</div>';
  }
  const items = evidence.slice(0, 8).map((item) => {
    const title = item.title || item.name || item.path || item.file_path || 'Evidence';
    const path = item.path || item.file_path || '';
    const score = typeof item.signal_score === 'number' ? item.signal_score.toFixed(2) : '—';
    const collection = item.collection || item.source || 'unknown';
    const detail = path ? `${title} (${path})` : title;
    return `<li>${escapeHtml(detail)} <span class="ui-muted">[${escapeHtml(collection)} | signal ${score}]</span></li>`;
  }).join('');
  const collectionLine = collections.length ? `<div class="ui-muted">Collections: ${escapeHtml(collections.join(', '))}</div>` : '';
  return `${collectionLine}<ul class="evidence-list">${items}</ul>`;
}

function renderModelFooter(run) {
  const plan = run?.artifacts?.route?.plan;
  if (!plan) return '';
  const parts = [
    plan.router ? `Router: ${plan.router}` : null,
    plan.reasoner ? `Reasoner: ${plan.reasoner}` : null,
    plan.critic ? `Critic: ${plan.critic}` : null,
    plan.summarizer ? `Summarizer: ${plan.summarizer}` : null,
  ].filter(Boolean);
  if (!parts.length) return '';
  return `<div class="model-footer">Models: ${escapeHtml(parts.join(' · '))}</div>`;
}

function summarizeEvent(event) {
  const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';
  const phase = event.phase || event.event || 'event';
  let detail = '';
  if (event.status) detail = event.status;
  if (event.role && event.model_id) detail = `${event.role} → ${event.model_id}`;
  if (event.role && event.ok === false) detail += ' (failed)';
  if (event.phase === 'route' && event.status === 'done') {
    const plan = event.models || (event.route && event.route.plan) || {};
    const parts = [];
    if (plan.reasoner) parts.push(`reasoner:${plan.reasoner}`);
    if (plan.critic) parts.push(`critic:${plan.critic}`);
    if (plan.summarizer) parts.push(`summarizer:${plan.summarizer}`);
    if (parts.length) detail = parts.join(' ');
  }
  if (event.phase === 'quality' && event.issues && event.issues.length) {
    detail = `issues: ${event.issues.join(', ')}`;
  }
  return `${time} ${phase}${detail ? ` · ${detail}` : ''}`.trim();
}

function renderLogs(run) {
  const events = run?.events || [];
  if (!events.length) {
    return '<div class="empty-state">No logs yet.</div>';
  }
  const preview = events.slice(-4).map((event) => `<li>${summarizeEvent(event)}</li>`).join('');
  const full = events.map((event) => `<li>${summarizeEvent(event)}</li>`).join('');
  const outputText = run?.consensus?.answer || run?.error || '';
  const outputBody = outputText ? renderMarkdown(outputText) : '<div class="empty-state">No output.</div>';
  const evidenceBody = renderEvidence(run);
  const modelFooter = renderModelFooter(run);
  return `
    <ul class="log-list">${preview}</ul>
    <details class="log-detail">
      <summary>View full log + output</summary>
      <div class="log-detail-body">
        <div>
          <div class="log-section-title">Full Log</div>
          <ul class="log-list">${full}</ul>
        </div>
        <div>
          <div class="log-section-title">Full Output</div>
          <div class="markdown">${outputBody}</div>
          ${modelFooter}
        </div>
        <div>
          <div class="log-section-title">Evidence</div>
          ${evidenceBody}
        </div>
      </div>
    </details>
  `;
}

function resolveTitle(run) {
  if (run.meta && run.meta.input_title) return run.meta.input_title;
  if (run.meta && run.meta.topic) return run.meta.topic;
  if (run.consensus && run.consensus.pope) return run.consensus.pope;
  return run.query || run.id;
}

function renderLatest(latest) {
  if (!latest) {
    latestEl.innerHTML = renderEmptyState();
    latestTimeEl.textContent = '—';
    if (latestPromptEl) latestPromptEl.textContent = 'Prompt: —';
    renderProgress(null);
    return;
  }
  if (latest.status === 'running') {
    latestEl.innerHTML = '<div class="empty-state">Running consensus... check the Decision Log for live events.</div>';
    latestTimeEl.textContent = formatTime(latest.created_at);
    if (latestPromptEl) {
      const input = latest.meta && latest.meta.input_title ? ` | Input: ${latest.meta.input_title}` : '';
      latestPromptEl.textContent = `Running: ${latest.query || '—'}${input}`;
    }
    renderProgress(latest);
    return;
  }
  if (latest.status === 'failed') {
    latestEl.innerHTML = `<div class="empty-state">Conclave failed: ${escapeHtml(latest.error || 'unknown error')}</div>`;
    latestTimeEl.textContent = formatTime(latest.completed_at || latest.created_at);
    renderProgress(latest);
    return;
  }
  if (!latest.consensus) {
    latestEl.innerHTML = renderEmptyState();
    latestTimeEl.textContent = formatTime(latest.completed_at || latest.created_at);
    renderProgress(latest);
    return;
  }
  latestEl.innerHTML = renderRecommendations(latest.consensus.answer || '');
  latestTimeEl.textContent = formatTime(latest.completed_at || latest.created_at);
  if (latestPromptEl) {
    const prompt = latest.query || '—';
    const input = latest.meta && latest.meta.input_title ? ` | Input: ${latest.meta.input_title}` : '';
    latestPromptEl.textContent = `Prompt: ${prompt}${input}`;
  }
  renderProgress(latest);
}

function renderRuns(runs) {
  runsEl.innerHTML = '';
  if (!runs.length) {
    runsEl.innerHTML = '<div class="empty-state">No runs yet. Start with a prompt to see decisions logged here.</div>';
    return;
  }
  runs.forEach((run) => {
    const row = document.createElement('div');
    row.className = 'table-row';
    const title = resolveTitle(run);
    const status = run.status || 'unknown';
    const output = run.consensus && run.consensus.answer ? run.consensus.answer : run.error || 'No output.';
    const inputInfo = run.meta && run.meta.input_title ? run.meta.input_title : (run.meta && run.meta.input_path ? run.meta.input_path.split('/').pop() : '—');
    row.innerHTML = `
      <div class="cell cell-title">
        ${escapeHtml(title)}
        <div class="ui-chip run-chip">${escapeHtml(status)}</div>
        <div class="ui-dim ui-mono run-id">${escapeHtml(run.id || '')}</div>
      </div>
      <div class="cell cell-prompt">
        <div class="clamp-2">${escapeHtml(run.query || '')}</div>
        <div class="prompt-meta ui-mono">Input: ${escapeHtml(inputInfo)}</div>
      </div>
      <div class="cell">${renderLogs(run)}</div>
      <div class="cell markdown clamp-3">${renderRecommendations(output)}</div>
      <div class="cell cell-time ui-mono">${formatTime(run.completed_at || run.created_at)}</div>
    `;
    runsEl.appendChild(row);
  });
}

function renderPromptList(prompts) {
  if (!promptListEl) return;
  promptListEl.innerHTML = '';
  if (!prompts.length) {
    promptListEl.innerHTML = '<div class="empty-state">No saved prompts yet.</div>';
    if (promptCountEl) promptCountEl.textContent = '0 saved';
    return;
  }
  if (promptCountEl) promptCountEl.textContent = `${prompts.length} saved`;
  prompts.forEach((prompt) => {
    const card = document.createElement('div');
    card.className = 'prompt-card';
    const title = prompt.title || prompt.query || prompt.id;
    const updated = prompt.updated_at ? formatTime(prompt.updated_at) : '—';
    card.innerHTML = `
      <div class="prompt-card-title">${escapeHtml(title)}</div>
      <div class="prompt-card-meta">Updated: ${escapeHtml(updated)}</div>
      <div class="prompt-card-meta clamp-2">${escapeHtml(prompt.query || '')}</div>
      <div class="prompt-card-actions">
        <button class="ui-button" data-action="load" data-id="${prompt.id}">Load</button>
        <button class="ui-button primary" data-action="run" data-id="${prompt.id}">Run</button>
      </div>
    `;
    promptListEl.appendChild(card);
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
    const runs = await fetchJSON('/api/runs?limit=8');
    renderRuns(runs.runs || []);
  } catch (err) {
    console.error(err);
  }
  try {
    const prompts = await fetchJSON('/api/prompts?limit=20');
    renderPromptList(prompts.prompts || []);
  } catch (err) {
    console.error(err);
  }
}

async function startRun(query) {
  setStatus('Running...');
  setSmoke(false);
  renderProgress({ status: 'running', events: [] });
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
  if (inputTitle) {
    payload.input_title = inputTitle;
  }
  if (currentPromptId) {
    payload.prompt_id = currentPromptId;
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

async function savePrompt() {
  const query = document.getElementById('query').value.trim();
  const title = document.getElementById('input-title').value.trim();
  const notes = document.getElementById('input-notes').value.trim();
  if (!query) {
    setPromptStatus('Prompt required');
    return;
  }
  const payload = { title, query, notes };
  let resp;
  if (currentPromptId) {
    resp = await fetchJSON(`/api/prompts/${currentPromptId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } else {
    resp = await fetchJSON('/api/prompts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }
  currentPromptId = resp.prompt.id;
  setPromptStatus(`Saved: ${currentPromptId}`);
  refresh();
}

async function loadPrompt(promptId) {
  const prompt = await fetchJSON(`/api/prompts/${promptId}`);
  document.getElementById('query').value = prompt.query || '';
  document.getElementById('input-title').value = prompt.title || '';
  document.getElementById('input-notes').value = prompt.notes || '';
  currentPromptId = prompt.id;
  setPromptStatus(`Loaded: ${prompt.id}`);
}

async function runPrompt(promptId) {
  const resp = await fetchJSON(`/api/prompts/${promptId}/run`, {
    method: 'POST',
  });
  currentRunId = resp.run_id;
  currentPromptId = promptId;
  setStatus('Running...');
  setSmoke(false);
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
        setStatus('Complete');
        renderLatest(run);
        refresh();
      } else if (run.status === 'failed') {
        clearInterval(pollTimer);
        setStatus(`Conclave failed: ${run.error || 'unknown error'}`);
        renderLatest(run);
      } else {
        setStatus('Running...');
        renderLatest(run);
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
    if (latest.meta && latest.meta.prompt_id) {
      await runPrompt(latest.meta.prompt_id);
      return;
    }
    const payload = { query: latest.query };
    if (latest.meta && latest.meta.input_path) {
      payload.input_path = latest.meta.input_path;
    }
    if (latest.meta && latest.meta.input_title) {
      payload.input_title = latest.meta.input_title;
    }
    setStatus('Running...');
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

clearBtn.addEventListener('click', () => {
  document.getElementById('query').value = '';
  document.getElementById('input-title').value = '';
  document.getElementById('input-notes').value = '';
  currentPromptId = null;
  setPromptStatus('Not saved');
});

if (savePromptBtn) {
  savePromptBtn.addEventListener('click', () => {
    savePrompt().catch((err) => console.error(err));
  });
}

if (copyLatestBtn) {
  copyLatestBtn.addEventListener('click', async () => {
    try {
      const latest = await fetchJSON('/api/runs/latest');
      const text = latest?.consensus?.answer || '';
      if (!text) return;
      await navigator.clipboard.writeText(text);
      setStatus('Copied consensus output.');
    } catch (err) {
      console.error(err);
    }
  });
}

if (promptExamplesEl) {
  promptExamplesEl.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const example = target.dataset.example;
    if (!example) return;
    document.getElementById('query').value = example;
  });
}

if (promptListEl) {
  promptListEl.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const action = target.dataset.action;
    const promptId = target.dataset.id;
    if (!action || !promptId) return;
    if (action === 'load') {
      loadPrompt(promptId).catch((err) => console.error(err));
    }
    if (action === 'run') {
      runPrompt(promptId).catch((err) => console.error(err));
    }
  });
}

refresh();
