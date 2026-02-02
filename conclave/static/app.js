const statusEl = document.getElementById('status');
const smokeEl = document.getElementById('smoke');
const latestEl = document.getElementById('latest');
const latestTimeEl = document.getElementById('latest-time');
const latestPromptEl = document.getElementById('latest-prompt');
const runsEl = document.getElementById('run-rows');
const form = document.getElementById('query-form');
const rerunBtn = document.getElementById('rerun-latest');
const clearBtn = document.getElementById('clear-form');

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

function summarizeEvent(event) {
  const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';
  const phase = event.phase || event.event || 'event';
  let detail = '';
  if (event.status) detail = event.status;
  if (event.role && event.model_id) detail = `${event.role} → ${event.model_id}`;
  if (event.role && event.ok === false) detail += ' (failed)';
  if (event.phase === 'quality' && event.issues && event.issues.length) {
    detail = `issues: ${event.issues.join(', ')}`;
  }
  return `${time} ${phase}${detail ? ` · ${detail}` : ''}`.trim();
}

function renderLogs(events, outputText) {
  if (!events || !events.length) {
    return '<div class="empty-state">No logs yet.</div>';
  }
  const items = events.slice(-10).map((event) => `<li>${summarizeEvent(event)}</li>`).join('');
  let detail = '';
  if (outputText) {
    const detailText = outputText.length > 1200 ? `${outputText.slice(0, 1200)}…` : outputText;
    detail = `<div class="log-detail markdown">${renderMarkdown(detailText)}</div>`;
  }
  return `<ul class="log-list">${items}</ul>${detail}`;
}

function resolveTitle(run) {
  if (run.meta && run.meta.input_title) return run.meta.input_title;
  if (run.meta && run.meta.topic) return run.meta.topic;
  if (run.consensus && run.consensus.pope) return run.consensus.pope;
  return run.query || run.id;
}

function renderLatest(latest) {
  if (!latest || !latest.consensus) {
    latestEl.innerHTML = '<div class="empty-state">No consensus yet.</div>';
    latestTimeEl.textContent = '—';
    if (latestPromptEl) latestPromptEl.textContent = 'Prompt: —';
    return;
  }
  latestEl.innerHTML = renderRecommendations(latest.consensus.answer || '');
  latestTimeEl.textContent = formatTime(latest.completed_at || latest.created_at);
  if (latestPromptEl) {
    const prompt = latest.query || '—';
    const input = latest.meta && latest.meta.input_title ? ` | Input: ${latest.meta.input_title}` : '';
    latestPromptEl.textContent = `Prompt: ${prompt}${input}`;
  }
}

function renderRuns(runs) {
  runsEl.innerHTML = '';
  if (!runs.length) {
    runsEl.innerHTML = '<div class="empty-state">No runs yet.</div>';
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
        ${escapeHtml(run.query || '')}
        <div class="prompt-meta ui-mono">Input: ${escapeHtml(inputInfo)}</div>
      </div>
      <div class="cell">${renderLogs(run.events || [], output)}</div>
      <div class="cell markdown">${renderRecommendations(output)}</div>
      <div class="cell cell-time ui-mono">${formatTime(run.completed_at || run.created_at)}</div>
    `;
    runsEl.appendChild(row);
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
}

async function startRun(query) {
  setStatus('Running...');
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
  if (inputTitle) {
    payload.input_title = inputTitle;
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
        setStatus('Complete');
        renderLatest(run);
        refresh();
      } else if (run.status === 'failed') {
        clearInterval(pollTimer);
        setStatus(`Conclave failed: ${run.error || 'unknown error'}`);
      } else {
        setStatus('Running...');
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
});

refresh();
