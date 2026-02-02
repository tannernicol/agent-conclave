// Conclave UI - Decision Pipeline Dashboard

const statusEl = document.getElementById('status');
const smokeEl = document.getElementById('smoke');
const latestEl = document.getElementById('latest');
const latestTimeEl = document.getElementById('latest-time');
const latestPromptEl = document.getElementById('latest-prompt');
const latestModelsEl = document.getElementById('latest-models');
const latestEvidenceEl = document.getElementById('latest-evidence');
const progressEl = document.getElementById('pipeline-progress');
const runActionEl = document.getElementById('run-action');
const runActionTitleEl = document.getElementById('run-action-title');
const runActionDetailEl = document.getElementById('run-action-detail');
const runListEl = document.getElementById('run-list');
const runCountEl = document.getElementById('run-count');
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
const collectionPickerEl = document.getElementById('input-collections');
const toggleEvidenceBtn = document.getElementById('toggle-evidence');
const outputTypeEl = document.getElementById('output-type');

let currentRunId = null;
let pollTimer = null;
let currentPromptId = null;
let collectionsLoaded = false;
let runActionTimer = null;
let runListTimer = null;

// Utilities
async function fetchJSON(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
  return await resp.json();
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatTime(value) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit'
  });
}

function formatTimeShort(value) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}

// Status management
function setStatus(text, type = '') {
  statusEl.textContent = text;
  statusEl.className = 'status-pill ui-mono';
  if (type) statusEl.classList.add(type);
  statusEl.setAttribute('aria-busy', type === 'running' ? 'true' : 'false');
}

function setPromptStatus(text) {
  if (promptStatusEl) promptStatusEl.textContent = text;
}

function setSmoke(active) {
  smokeEl.classList.toggle('active', active);
  if (smokeStatusEl) smokeStatusEl.textContent = active ? 'Consensus reached.' : 'No consensus in progress.';
}

function setRunAction(title, detail = '', type = 'info') {
  if (!runActionEl) return;
  if (runActionTimer) {
    clearTimeout(runActionTimer);
    runActionTimer = null;
  }
  if (!title) {
    runActionEl.classList.add('hidden');
    runActionEl.classList.remove('running', 'error', 'success');
    if (runActionTitleEl) runActionTitleEl.textContent = '';
    if (runActionDetailEl) runActionDetailEl.textContent = '';
    return;
  }
  runActionEl.classList.remove('hidden', 'running', 'error', 'success');
  if (type) runActionEl.classList.add(type);
  if (runActionTitleEl) runActionTitleEl.textContent = title;
  if (runActionDetailEl) runActionDetailEl.textContent = detail;
}

function setRunActionEphemeral(title, detail, type = 'success', ms = 4000) {
  setRunAction(title, detail, type);
  runActionTimer = setTimeout(() => setRunAction(''), ms);
}

function summarizeQuery(text, max = 80) {
  if (!text) return '';
  const trimmed = text.trim();
  if (trimmed.length <= max) return trimmed;
  return `${trimmed.slice(0, max)}…`;
}

function updatePromptButton() {
  if (savePromptBtn) {
    savePromptBtn.textContent = currentPromptId ? 'Update' : 'Save';
  }
}

// Markdown rendering
function renderInline(text) {
  let value = escapeHtml(text);
  value = value.replace(/`([^`]+)`/g, '<code>$1</code>');
  value = value.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  value = value.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  return value;
}

function renderMarkdown(text) {
  if (!text) return '';
  const lines = text.split('\n');
  let html = '';
  let inCode = false;
  let inUl = false;
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html += `<p>${renderInline(paragraph.join(' '))}</p>`;
      paragraph = [];
    }
  };

  const closeLists = () => {
    if (inUl) { html += '</ul>'; inUl = false; }
  };

  lines.forEach((line) => {
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      if (inCode) { html += '</code></pre>'; inCode = false; }
      else { flushParagraph(); closeLists(); inCode = true; html += '<pre><code>'; }
      return;
    }
    if (inCode) { html += `${escapeHtml(line)}\n`; return; }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph(); closeLists();
      const level = headingMatch[1].length;
      html += `<h${level}>${renderInline(headingMatch[2])}</h${level}>`;
      return;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (ulMatch) {
      flushParagraph();
      if (!inUl) { html += '<ul>'; inUl = true; }
      html += `<li>${renderInline(ulMatch[1])}</li>`;
      return;
    }

    if (!trimmed) { flushParagraph(); closeLists(); return; }
    paragraph.push(trimmed);
  });

  flushParagraph();
  closeLists();
  if (inCode) html += '</code></pre>';
  return html;
}

// Check if output indicates insufficient evidence
function isInsufficientEvidence(consensus) {
  if (!consensus) return false;
  return Boolean(consensus.insufficient_evidence);
}

// Parse issue details from consensus
function parseIssueDetails(consensus) {
  const details = {
    evidenceCount: '?',
    minEvidence: 2,
    maxSignal: '?',
    pdfRatio: '?',
    issues: []
  };

  if (!consensus || !consensus.answer) return details;

  const text = consensus.answer;
  const evMatch = text.match(/Evidence count:\s*(\d+)\s*\(min\s*(\d+)\)/i);
  if (evMatch) {
    details.evidenceCount = evMatch[1];
    details.minEvidence = evMatch[2];
  }

  const sigMatch = text.match(/Max signal score:\s*([\d.]+)/i);
  if (sigMatch) details.maxSignal = sigMatch[1];

  const pdfMatch = text.match(/PDF ratio:\s*([\d.]+)/i);
  if (pdfMatch) details.pdfRatio = pdfMatch[1];

  const issuesMatch = text.match(/Issues:\s*([^\n]+)/i);
  if (issuesMatch) {
    details.issues = issuesMatch[1].split(',').map(s => s.trim());
  }

  return details;
}

// Render error state for insufficient evidence
function renderInsufficientEvidence(consensus, run) {
  const details = parseIssueDetails(consensus);
  const collections = run?.artifacts?.route?.collections || [];

  return `
    <div class="output-error">
      <div class="output-error-title">Insufficient Evidence</div>
      <div class="output-error-details">
        <p>Conclave couldn't find enough high-quality evidence to provide a confident answer.</p>
        <ul>
          <li>Evidence found: <strong>${details.evidenceCount}</strong> (need ${details.minEvidence}+)</li>
          <li>Max signal score: <strong>${details.maxSignal}</strong></li>
          ${details.issues.length ? `<li>Issues: ${details.issues.join(', ')}</li>` : ''}
        </ul>
      </div>
      <div class="output-error-help">
        <strong>Try:</strong> Add supporting notes above, specify a collection (tax-rag, health-rag, bounty-rag), or run <code>conclave index</code> to refresh NAS content.
      </div>
    </div>
  `;
}

// Render success output
function renderSuccessOutput(text) {
  const lines = text.split('\n');
  const bullets = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('###')) continue; // Skip headers
    const bulletMatch = trimmed.match(/^[-*•]\s+(.*)$/) || trimmed.match(/^\d+\.\s+(.*)$/);
    if (bulletMatch) bullets.push(bulletMatch[1]);
    else if (trimmed && !trimmed.startsWith('#') && bullets.length < 6) {
      bullets.push(trimmed);
    }
    if (bullets.length >= 6) break;
  }

  if (!bullets.length) {
    return `<div class="markdown">${renderMarkdown(text)}</div>`;
  }

  return `
    <div class="output-success">
      <ul class="output-list">
        ${bullets.map(b => `<li>${renderInline(b)}</li>`).join('')}
      </ul>
    </div>
  `;
}

// Render answer preview for run cards - show actual content
function renderAnswerPreview(text) {
  if (!text) return '';

  const lines = text.split('\n');
  const content = [];
  let charCount = 0;
  const maxChars = 400;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Skip "Insufficient Evidence" header if present
    if (trimmed.includes('Insufficient Evidence')) continue;

    // Skip markdown headers but keep the content
    const headerMatch = trimmed.match(/^#{1,3}\s+(.*)$/);
    const bulletMatch = trimmed.match(/^[-*•]\s+(.*)$/) || trimmed.match(/^\d+\.\s+(.*)$/);

    let lineContent = '';
    if (headerMatch) {
      lineContent = `<strong>${escapeHtml(headerMatch[1])}</strong>`;
    } else if (bulletMatch) {
      lineContent = `• ${renderInline(bulletMatch[1])}`;
    } else {
      lineContent = renderInline(trimmed);
    }

    charCount += trimmed.length;
    content.push(lineContent);

    if (charCount > maxChars || content.length >= 8) break;
  }

  if (!content.length) return '';

  return content.map(c => `<div class="answer-line">${c}</div>`).join('');
}

function normalizeHeading(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s/]+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractSection(markdown, titles) {
  if (!markdown) return '';
  const wanted = new Set(titles.map(normalizeHeading));
  const lines = markdown.split('\n');
  let collecting = false;
  const buffer = [];
  for (const line of lines) {
    const headingMatch = line.match(/^#{1,3}\s*(.+)$/);
    const boldMatch = line.match(/^\*\*(.+)\*\*$/);
    if (headingMatch || boldMatch) {
      const headingText = headingMatch ? headingMatch[1] : boldMatch[1];
      if (collecting) break;
      collecting = wanted.has(normalizeHeading(headingText));
      continue;
    }
    if (collecting) buffer.push(line);
  }
  return buffer.join('\n').trim();
}

function extractCodeBlock(markdown) {
  if (!markdown) return '';
  const match = markdown.match(/```(?:bash|sh)?\n([\s\S]*?)```/i);
  return match ? match[1].trim() : '';
}

function extractBullets(markdown, pattern, maxItems = 3) {
  if (!markdown) return [];
  const lines = markdown.split('\n').map((line) => line.trim());
  const items = [];
  for (const line of lines) {
    if (!line.startsWith('-')) continue;
    if (!pattern.test(line)) continue;
    items.push(line.replace(/^-+\s*/, '').trim());
    if (items.length >= maxItems) break;
  }
  return items;
}

function renderBountySummary(run) {
  if (!run) return '';
  const domain = run.artifacts?.route?.domain;
  if (domain !== 'bounty') return '';
  const answer = run.consensus?.answer || '';
  if (!answer) return '';

  const summary = extractSection(answer, ['summary', 'tl;dr', 'overview']);
  const findings = extractSection(answer, [
    'existing bug reports / internal findings',
    'existing bug reports',
    'internal findings',
    'findings',
    'issues'
  ]);
  const launch = extractSection(answer, ['launch commands', 'launch command', 'launch approach', 'launch']);
  const code = extractCodeBlock(answer);
  const findingBullets = findings ? [] : extractBullets(answer, /(finding|issue|bug|vuln|risk)/i, 3);

  if (!summary && !findings && !launch && !code && !findingBullets.length) return '';

  const sections = [];
  if (summary) {
    sections.push(`
      <div class="bounty-section">
        <div class="bounty-section-title">Summary</div>
        <div class="bounty-section-body">${renderMarkdown(summary)}</div>
      </div>
    `);
  }
  if (launch || code) {
    sections.push(`
      <div class="bounty-section">
        <div class="bounty-section-title">Launch</div>
        ${launch ? `<div class="bounty-section-body">${renderMarkdown(launch)}</div>` : ''}
        ${code ? `<pre class="bounty-code">${escapeHtml(code)}</pre>` : ''}
      </div>
    `);
  }
  if (findings || findingBullets.length) {
    const body = findings
      ? renderMarkdown(findings)
      : `<ul>${findingBullets.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
    sections.push(`
      <div class="bounty-section">
        <div class="bounty-section-title">Findings</div>
        <div class="bounty-section-body">${body}</div>
      </div>
    `);
  }

  return `
    <div class="bounty-summary">
      <div class="bounty-title">Bounty Summary</div>
      ${sections.join('')}
    </div>
  `;
}

// Pipeline progress
function buildPhaseState(run) {
  const phases = [
    { key: 'route', label: 'Route' },
    { key: 'retrieve', label: 'RAG' },
    { key: 'reasoner', label: 'Reason' },
    { key: 'critic', label: 'Critic' },
    { key: 'summarizer', label: 'Summary' },
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

function renderRunProgress(run) {
  const phases = buildPhaseState(run);
  const active = phases.find((phase) => phase.active) || phases.find((phase) => !phase.done);
  const label = active ? active.label : (run.status === 'complete' ? 'Complete' : 'Queued');
  const steps = phases.map((phase) => {
    const classes = ['phase'];
    if (phase.done) classes.push('done');
    if (phase.active) classes.push('active');
    return `<span class="${classes.join(' ')}">${phase.label}</span>`;
  }).join('');
  return `
    <div class="run-progress">
      <div class="run-progress-label">Phase: ${label}</div>
      <div class="run-progress-steps">${steps}</div>
    </div>
  `;
}

// Render models used
function renderModels(latest) {
  if (!latestModelsEl) return;

  const models = latest?.artifacts?.route?.plan_details || latest?.artifacts?.route?.plan || {};
  if (!Object.keys(models).length) {
    latestModelsEl.innerHTML = '';
    return;
  }

  const tags = Object.entries(models).map(([role, info]) => {
    const label = typeof info === 'object' ? (info.label || info.id || role) : info;
    return `<span class="model-tag"><span class="model-tag-role">${escapeHtml(role)}</span><span class="model-tag-name">${escapeHtml(label)}</span></span>`;
  }).join('');

  latestModelsEl.innerHTML = tags;
}

// Render evidence panel
function renderEvidence(latest) {
  if (!latestEvidenceEl) return;

  const evidence = latest?.artifacts?.context?.evidence || [];
  if (!evidence.length) {
    latestEvidenceEl.innerHTML = '';
    return;
  }

  const items = evidence.slice(0, 8).map((item) => {
    const source = item.source || 'rag';
    const sourceClass = source === 'mcp' ? 'mcp' : (source === 'user' ? 'user' : '');
    const title = item.title || item.path?.split('/').pop() || 'Evidence';
    const snippet = item.snippet || '';
    const score = typeof item.signal_score === 'number' ? item.signal_score.toFixed(2) : '—';

    return `
      <div class="evidence-item">
        <span class="evidence-item-source ${sourceClass}">${escapeHtml(source)}</span>
        <div class="evidence-item-body">
          <div class="evidence-item-title">${escapeHtml(title)}</div>
          <div class="evidence-item-snippet">${escapeHtml(snippet.slice(0, 150))}</div>
        </div>
        <span class="evidence-item-score">${score}</span>
      </div>
    `;
  }).join('');

  latestEvidenceEl.innerHTML = `
    <div class="evidence-header">
      <span class="evidence-title">Evidence Sources</span>
      <span class="evidence-count">${evidence.length} items</span>
    </div>
    <div class="evidence-items">${items}</div>
  `;
}

// Render Latest Decision
function renderLatest(latest) {
  if (!latest) {
    latestEl.innerHTML = `
      <div class="empty-state">
        <p>Ask a question to get a multi-model consensus.</p>
        <p class="ui-muted">Conclave routes to the best models, gathers evidence, and deliberates.</p>
      </div>
    `;
    latestTimeEl.textContent = '—';
    if (latestPromptEl) latestPromptEl.textContent = '';
    if (latestModelsEl) latestModelsEl.innerHTML = '';
    if (latestEvidenceEl) latestEvidenceEl.innerHTML = '';
    renderProgress(null);
    return;
  }

  latestTimeEl.textContent = formatTime(latest.completed_at || latest.created_at);

  if (latestPromptEl) {
    const prompt = latest.query || '';
    const outputType = latest.meta?.output_type || '';
    const promptText = prompt ? `"${prompt.slice(0, 100)}${prompt.length > 100 ? '...' : ''}"` : '';
    const outputHtml = outputType ? `<span class="latest-output">Output: ${escapeHtml(outputType)}</span>` : '';
    latestPromptEl.innerHTML = `${escapeHtml(promptText)} ${outputHtml}`.trim();
  }

  // Render models used
  renderModels(latest);

  if (latest.status === 'running') {
    latestEl.innerHTML = `
      <div class="empty-state">
        <p>Running consensus...</p>
        <p class="ui-muted">Check the pipeline stages above for progress.</p>
      </div>
    `;
    if (latestEvidenceEl) latestEvidenceEl.innerHTML = '';
    renderProgress(latest);
    return;
  }

  if (latest.status === 'failed') {
    latestEl.innerHTML = `
      <div class="output-error">
        <div class="output-error-title">Pipeline Failed</div>
        <div class="output-error-details">${escapeHtml(latest.error || 'Unknown error')}</div>
      </div>
    `;
    if (latestEvidenceEl) latestEvidenceEl.innerHTML = '';
    renderProgress(latest);
    return;
  }

  if (!latest.consensus) {
    latestEl.innerHTML = `<div class="empty-state">No consensus generated.</div>`;
    if (latestEvidenceEl) latestEvidenceEl.innerHTML = '';
    renderProgress(latest);
    return;
  }

  // Render evidence panel (hidden by default)
  renderEvidence(latest);

  if (isInsufficientEvidence(latest.consensus)) {
    latestEl.innerHTML = renderInsufficientEvidence(latest.consensus, latest);
  } else {
    // Show full markdown output for successful consensus
    latestEl.innerHTML = `
      <div class="output-success">
        <div class="markdown">${renderMarkdown(latest.consensus.answer || '')}</div>
      </div>
    `;
  }

  renderProgress(latest);
}

// Render Run History as cards
function renderRuns(runs) {
  if (!runListEl) return;

  if (runCountEl) {
    const runningCount = runs.filter((run) => run.status === 'running').length;
    const runningLabel = runningCount ? ` • ${runningCount} running` : '';
    runCountEl.textContent = `${runs.length} run${runs.length !== 1 ? 's' : ''}${runningLabel}`;
  }

  if (!runs.length) {
    runListEl.innerHTML = `<div class="empty-state">No runs yet. Ask a question to see history here.</div>`;
    return;
  }

  runListEl.innerHTML = runs.map((run) => {
    const title = run.meta?.input_title || run.query?.slice(0, 60) || run.id;
    const status = run.status || 'unknown';
    const isInsufficient = isInsufficientEvidence(run.consensus);
    const hasError = status === 'failed' || isInsufficient;
    const outputType = run.meta?.output_type || '';

    let statusChip = '';
    if (status === 'running') {
      statusChip = '<span class="status-chip running">Running</span>';
    } else if (status === 'failed') {
      statusChip = '<span class="status-chip failed">Failed</span>';
    } else if (isInsufficient) {
      statusChip = '<span class="status-chip insufficient">Low Evidence</span>';
    } else if (status === 'complete') {
      statusChip = '<span class="status-chip complete">Complete</span>';
    }
    const outputChip = outputType ? `<span class="status-chip output">Output: ${escapeHtml(outputType)}</span>` : '';

    let outputHtml = '';
    if (run.consensus?.answer) {
      if (isInsufficient) {
        outputHtml = `<div class="run-card-output error">Insufficient evidence to provide confident answer</div>`;
      } else {
        // Show the actual answer content (first 500 chars or key bullets)
        outputHtml = `<div class="run-card-answer">${renderAnswerPreview(run.consensus.answer)}</div>`;
      }
    } else if (run.error) {
      outputHtml = `<div class="run-card-output error">${escapeHtml(run.error)}</div>`;
    }

    const cardClass = status === 'running' ? 'running' : (hasError ? 'error' : 'success');
    const progressHtml = status === 'running' ? renderRunProgress(run) : '';
    const bountySummary = renderBountySummary(run);

    return `
      <div class="run-card ${cardClass}" data-run-id="${escapeHtml(run.id || '')}" data-run-title="${escapeHtml(title)}" data-run-query="${escapeHtml(run.query || '')}" data-run-output="${escapeHtml(outputType)}">
        <div class="run-card-main">
          <div class="run-card-title">
            ${escapeHtml(title.slice(0, 60))}${title.length > 60 ? '...' : ''}
            ${statusChip}
            ${outputChip}
          </div>
          <div class="run-card-query">${escapeHtml(run.query || '')}</div>
          ${progressHtml}
          ${bountySummary}
          ${outputHtml}
          <details class="run-card-expand">
            <summary>View full details</summary>
            <div class="run-card-detail">
              ${renderRunDetail(run)}
            </div>
          </details>
        </div>
        <div class="run-card-side">
          <div class="run-card-time">${formatTime(run.completed_at || run.created_at)}</div>
          <div class="run-card-actions">
            <button class="ui-button ghost small edit-run" data-run-id="${escapeHtml(run.id || '')}">Edit</button>
            <button class="ui-button ghost small rerun-run" data-run-id="${escapeHtml(run.id || '')}">Rerun</button>
            <button class="ui-button ghost small delete-run" data-run-id="${escapeHtml(run.id || '')}">×</button>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderRunDetail(run) {
  const events = run.events || [];
  const lastEvents = events.slice(-6);
  const outputType = run.meta?.output_type;

  const logItems = lastEvents.map((event) => {
    const time = event.timestamp ? formatTimeShort(event.timestamp) : '';
    const phase = event.phase || event.event || 'event';
    let detail = event.status || '';
    if (event.role && event.model_id) {
      detail = `${event.role} → ${event.model_label || event.model_id}`;
    }
    if (event.phase === 'quality' && event.issues?.length) {
      detail = event.issues.join(', ');
    }
    return `<li>${time} ${phase}${detail ? ': ' + detail : ''}</li>`;
  }).join('');

  const evidence = run.artifacts?.context?.evidence || [];
  const evidenceItems = evidence.slice(0, 5).map((item) => {
    const name = item.title || item.name || item.path?.split('/').pop() || 'Evidence';
    const score = typeof item.signal_score === 'number' ? item.signal_score.toFixed(2) : '—';
    return `<li>${escapeHtml(name)} <span class="ui-muted">[${score}]</span></li>`;
  }).join('');

  const metaItems = [];
  if (outputType) metaItems.push(`<li>Output: ${escapeHtml(outputType)}</li>`);

  return `
    ${metaItems.length ? `
    <div class="log-section">
      <div class="log-section-title">Run Meta</div>
      <ul class="log-list">${metaItems.join('')}</ul>
    </div>
    ` : ''}
    <div class="log-section">
      <div class="log-section-title">Pipeline Log</div>
      <ul class="log-list">${logItems || '<li>No events</li>'}</ul>
    </div>
    ${evidence.length ? `
    <div class="log-section">
      <div class="log-section-title">Evidence (${evidence.length})</div>
      <ul class="log-list">${evidenceItems}</ul>
    </div>
    ` : ''}
  `;
}

// Render Saved Prompts
function renderPromptList(prompts) {
  if (!promptListEl) return;

  if (promptCountEl) {
    promptCountEl.textContent = `${prompts.length} saved`;
  }

  if (!prompts.length) {
    promptListEl.innerHTML = `<div class="empty-state">No saved prompts yet.</div>`;
    return;
  }

  promptListEl.innerHTML = prompts.map((prompt) => {
    const title = prompt.title || prompt.query?.slice(0, 40) || prompt.id;
    const outputType = prompt.output_type ? ` • Output: ${prompt.output_type}` : '';
    return `
      <div class="prompt-card">
        <div class="prompt-card-title">${escapeHtml(title)}</div>
        <div class="prompt-card-meta">Updated ${formatTime(prompt.updated_at)}${escapeHtml(outputType)}</div>
        <div class="prompt-card-query">${escapeHtml(prompt.query || '')}</div>
        <div class="prompt-card-actions">
          <button class="ui-button small" data-action="load" data-id="${prompt.id}">Load</button>
          <button class="ui-button small primary" data-action="run" data-id="${prompt.id}">Run</button>
          <button class="ui-button ghost small" data-action="delete" data-id="${prompt.id}">×</button>
        </div>
      </div>
    `;
  }).join('');
}

// Load collections for picker
async function loadCollections() {
  if (collectionsLoaded || !collectionPickerEl) return;

  try {
    const resp = await fetchJSON('/api/collections');
    const collections = resp.collections || [];

    // Keep the auto-detect option
    let html = '<option value="">Auto-detect domain</option>';

    // Group by type/reliability
    const grouped = { high: [], medium: [], other: [] };
    collections.forEach((c) => {
      const reliability = c.reliability || 'other';
      if (grouped[reliability]) {
        grouped[reliability].push(c);
      } else {
        grouped.other.push(c);
      }
    });

    // Add grouped options
    if (grouped.high.length) {
      html += '<optgroup label="High Reliability">';
      grouped.high.forEach((c) => {
        html += `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} (${c.file_count || 0})</option>`;
      });
      html += '</optgroup>';
    }

    if (grouped.medium.length) {
      html += '<optgroup label="Medium Reliability">';
      grouped.medium.forEach((c) => {
        html += `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} (${c.file_count || 0})</option>`;
      });
      html += '</optgroup>';
    }

    if (grouped.other.length) {
      html += '<optgroup label="Other">';
      grouped.other.forEach((c) => {
        html += `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} (${c.file_count || 0})</option>`;
      });
      html += '</optgroup>';
    }

    collectionPickerEl.innerHTML = html;
    collectionsLoaded = true;
  } catch (err) {
    console.error('Failed to load collections:', err);
  }
}

// Data fetching
async function refresh() {
  try {
    const status = await fetchJSON('/api/status');
    renderLatest(status.latest);
  } catch (err) {
    console.error('Status fetch failed:', err);
  }

  try {
    const runs = await fetchJSON('/api/runs?limit=10');
    renderRuns(runs.runs || []);
  } catch (err) {
    console.error('Runs fetch failed:', err);
  }

  try {
    const prompts = await fetchJSON('/api/prompts?limit=20');
    renderPromptList(prompts.prompts || []);
  } catch (err) {
    console.error('Prompts fetch failed:', err);
  }

  // Load collections for picker (once)
  loadCollections();

  if (!runListTimer) {
    runListTimer = setInterval(async () => {
      try {
        const runs = await fetchJSON('/api/runs?limit=10');
        renderRuns(runs.runs || []);
      } catch (err) {
        console.error('Runs refresh failed:', err);
      }
    }, 5000);
  }
}

// Run management
async function startRun(query) {
  setStatus('Running...', 'running');
  setSmoke(false);
  renderProgress({ status: 'running', events: [] });
  const outputType = outputTypeEl?.value || '';
  setRunAction(
    'Running new decision',
    `Source: prompt editor • ${summarizeQuery(query)}${outputType ? ' • Output: ' + outputType : ''}`,
    'running'
  );

  const inputTitle = document.getElementById('input-title').value.trim();
  const inputNotes = document.getElementById('input-notes').value.trim();
  const inputArtifacts = document.getElementById('input-artifacts').value
    .split('\n').map(l => l.trim()).filter(Boolean);

  // Get selected collections
  let selectedCollections = [];
  if (collectionPickerEl) {
    selectedCollections = Array.from(collectionPickerEl.selectedOptions)
      .map(opt => opt.value)
      .filter(v => v); // Filter out empty "auto-detect" option
  }

  let inputId = null;
  if (inputNotes) {
    const inputResp = await fetchJSON('/api/inputs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: inputTitle,
        content: inputNotes,
        question: query,
        artifacts: inputArtifacts,
        output_type: outputType,
      }),
    });
    inputId = inputResp.input_id;
  }

  const payload = { query };
  if (inputId) payload.input_id = inputId;
  if (inputTitle) payload.input_title = inputTitle;
  if (currentPromptId) payload.prompt_id = currentPromptId;
  if (selectedCollections.length) payload.collections = selectedCollections;
  if (outputType) payload.output_type = outputType;

  const resp = await fetchJSON('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  currentRunId = resp.run_id;
  if (currentRunId) pollRun(currentRunId);
  else setStatus('Failed to start', 'error');
}

async function pollRun(runId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const run = await fetchJSON(`/api/runs/${runId}`);

      if (run.status === 'complete') {
        clearInterval(pollTimer);
        setSmoke(true);
        setStatus('Complete', 'success');
        renderLatest(run);
        refresh();
        setRunActionEphemeral('Run complete', 'Latest Decision updated.', 'success');
      } else if (run.status === 'failed') {
        clearInterval(pollTimer);
        setStatus('Failed', 'error');
        renderLatest(run);
        refresh();
        setRunActionEphemeral('Run failed', 'See process logs for details.', 'error', 5000);
      } else {
        renderLatest(run);
      }
    } catch (err) {
      console.error('Poll error:', err);
    }
  }, 2000);
}

// Prompt management
async function savePrompt() {
  const query = document.getElementById('query').value.trim();
  const title = document.getElementById('input-title').value.trim();
  const notes = document.getElementById('input-notes').value.trim();
  const artifacts = document.getElementById('input-artifacts').value
    .split('\n').map(l => l.trim()).filter(Boolean);
  const outputType = outputTypeEl?.value || '';

  if (!query) {
    setPromptStatus('Enter a question first');
    return;
  }

  const payload = { title, query, notes, artifacts };
  if (outputType) payload.output_type = outputType;
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
  setPromptStatus('Saved');
  updatePromptButton();
  refresh();
}

async function loadPrompt(promptId) {
  const prompt = await fetchJSON(`/api/prompts/${promptId}`);
  document.getElementById('query').value = prompt.query || '';
  document.getElementById('input-title').value = prompt.title || '';
  document.getElementById('input-notes').value = prompt.notes || '';
  document.getElementById('input-artifacts').value = (prompt.artifacts || []).join('\n');
  if (outputTypeEl) outputTypeEl.value = prompt.output_type || '';
  currentPromptId = prompt.id;
  setPromptStatus('Loaded');
  updatePromptButton();
}

async function runPrompt(promptId) {
  const resp = await fetchJSON(`/api/prompts/${promptId}/run`, { method: 'POST' });
  currentRunId = resp.run_id;
  currentPromptId = promptId;
  setStatus('Running...', 'running');
  setSmoke(false);
  setRunAction('Re-running saved prompt', `Prompt ID ${promptId} • Output will appear in Latest Decision.`, 'running');
  updatePromptButton();
  pollRun(currentRunId);
}

async function loadRunForEdit(run) {
  if (!run) return;
  if (run.meta?.prompt_id) {
    await loadPrompt(run.meta.prompt_id);
    return;
  }
  document.getElementById('query').value = run.query || '';
  document.getElementById('input-title').value = run.meta?.input_title || '';
  if (outputTypeEl) outputTypeEl.value = run.meta?.output_type || '';
  currentPromptId = null;
  setPromptStatus('Loaded from run');
  updatePromptButton();
}

// Event listeners
form.addEventListener('submit', (e) => {
  e.preventDefault();
  const query = document.getElementById('query').value.trim();
  if (query) startRun(query);
});

rerunBtn.addEventListener('click', async () => {
  try {
    const latest = await fetchJSON('/api/runs/latest');
    if (!latest?.query) {
      setStatus('No run to rerun');
      return;
    }
    const latestLabel = latest.meta?.input_title || latest.query || latest.id;
    const latestOutput = latest.meta?.output_type;
    setRunAction(
      'Re-running latest decision',
      `Source: ${summarizeQuery(latestLabel)}${latestOutput ? ' • Output: ' + latestOutput : ''} • Output will appear in Latest Decision.`,
      'running'
    );
    if (latest.meta?.prompt_id) {
      await runPrompt(latest.meta.prompt_id);
      return;
    }
    setStatus('Running...', 'running');
    setSmoke(false);
    const resp = await fetchJSON('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: latest.query,
        input_path: latest.meta?.input_path,
        input_title: latest.meta?.input_title
      }),
    });
    if (resp.run_id) pollRun(resp.run_id);
  } catch (err) {
    console.error(err);
  }
});

clearBtn.addEventListener('click', () => {
  document.getElementById('query').value = '';
  document.getElementById('input-title').value = '';
  document.getElementById('input-notes').value = '';
  document.getElementById('input-artifacts').value = '';
  if (outputTypeEl) outputTypeEl.value = '';
  currentPromptId = null;
  setPromptStatus('Ready');
  updatePromptButton();
});

if (savePromptBtn) {
  savePromptBtn.addEventListener('click', () => savePrompt().catch(console.error));
}

if (copyLatestBtn) {
  copyLatestBtn.addEventListener('click', async () => {
    try {
      const latest = await fetchJSON('/api/runs/latest');
      const text = latest?.consensus?.answer || '';
      if (text) {
        await navigator.clipboard.writeText(text);
        setStatus('Copied');
        setTimeout(() => setStatus('Idle'), 2000);
      }
    } catch (err) {
      console.error(err);
    }
  });
}

if (toggleEvidenceBtn && latestEvidenceEl) {
  toggleEvidenceBtn.addEventListener('click', () => {
    const isVisible = latestEvidenceEl.classList.contains('visible');
    latestEvidenceEl.classList.toggle('visible', !isVisible);
    toggleEvidenceBtn.textContent = isVisible ? 'Show Evidence' : 'Hide Evidence';
  });
}

if (promptExamplesEl) {
  promptExamplesEl.addEventListener('click', (e) => {
    const example = e.target.dataset?.example;
    if (example) document.getElementById('query').value = example;
  });
}

if (promptListEl) {
  promptListEl.addEventListener('click', async (e) => {
    const action = e.target.dataset?.action;
    const promptId = e.target.dataset?.id;
    if (!action || !promptId) return;

    if (action === 'load') await loadPrompt(promptId);
    if (action === 'run') await runPrompt(promptId);
    if (action === 'delete') {
      if (!confirm('Delete this prompt?')) return;
      await fetchJSON(`/api/prompts/${promptId}`, { method: 'DELETE' });
      if (currentPromptId === promptId) {
        currentPromptId = null;
        setPromptStatus('Ready');
        updatePromptButton();
      }
      refresh();
    }
  });
}

if (runListEl) {
  runListEl.addEventListener('click', async (e) => {
    const runId = e.target.dataset?.runId;
    if (!runId) return;

    if (e.target.classList.contains('edit-run')) {
      const run = await fetchJSON(`/api/runs/${runId}`);
      loadRunForEdit(run);
    }

    if (e.target.classList.contains('rerun-run')) {
      const card = e.target.closest('.run-card');
      const label = card?.dataset?.runTitle || card?.dataset?.runQuery || runId;
      const outputType = card?.dataset?.runOutput || '';
      setRunAction(
        'Re-running from history',
        `Source: ${summarizeQuery(label)}${outputType ? ' • Output: ' + outputType : ''} • Output will appear in Latest Decision.`,
        'running'
      );
      setStatus('Running...', 'running');
      setSmoke(false);
      const resp = await fetchJSON(`/api/runs/${runId}/rerun`, { method: 'POST' });
      if (resp.run_id) pollRun(resp.run_id);
    }

    if (e.target.classList.contains('delete-run')) {
      if (!confirm('Delete this run?')) return;
      await fetchJSON(`/api/runs/${runId}`, { method: 'DELETE' });
      refresh();
    }
  });
}

// Initialize
refresh();
updatePromptButton();
