// Conclave UI - Decision Pipeline Dashboard

const statusEl = document.getElementById('status');
const smokeEl = document.getElementById('smoke');
const latestEl = document.getElementById('latest');
const latestTimeEl = document.getElementById('latest-time');
const latestPromptEl = document.getElementById('latest-prompt');
const latestModelsEl = document.getElementById('latest-models');
const latestEvidenceEl = document.getElementById('latest-evidence');
const latestArtifactsEl = document.getElementById('latest-artifacts');
const viewLatestContextBtn = document.getElementById('view-latest-context');
const latestTitleEl = document.getElementById('latest-title');
const clearSelectionBtn = document.getElementById('clear-selection');
const progressEl = document.getElementById('pipeline-progress');
const liveProgressEl = document.getElementById('live-progress');
const reconcilePanelEl = document.getElementById('reconcile-panel');
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
const artifactFieldEl = document.getElementById('input-artifacts');
const fileInputEl = document.getElementById('input-files');
const artifactListEl = document.getElementById('artifact-list');
const useCaseEl = document.getElementById('use-case');
const agentSetEl = document.getElementById('agent-set');
const roleOverridesEl = document.getElementById('role-overrides');
const planSummaryEl = document.getElementById('plan-summary');
const refreshPlanBtn = document.getElementById('refresh-plan');
const clearOverridesBtn = document.getElementById('clear-overrides');
const historyTabButtons = document.querySelectorAll('.panel-tabs [data-tab]');
const historyTabPanels = document.querySelectorAll('.tab-panel[data-tab]');
const contextModalEl = document.getElementById('context-modal');
const contextBodyEl = document.getElementById('context-body');
const contextTitleEl = document.getElementById('context-title');
const contextSubtitleEl = document.getElementById('context-subtitle');
const closeContextBtn = document.getElementById('close-context');
const copyContextBtn = document.getElementById('copy-context');
const copyContextCompactBtn = document.getElementById('copy-context-compact');

let currentRunId = null;
let pollTimer = null;
let currentPromptId = null;
let collectionsLoaded = false;
let outputsLoaded = false;
let useCasesLoaded = false;
let agentSetsLoaded = false;
let runActionTimer = null;
let runListTimer = null;
let runWebSocket = null;
let activeContextRun = null;
let latestRunCache = null;
let selectedRun = null;
let latestImagePrompt = '';
let runsCache = [];
const autoPrunedRuns = new Set();
let useCasesCache = [];
let agentSetsCache = [];
let modelsCache = [];
let modelsById = {};
let planCache = null;
let roleOverrides = {};
let planRefreshTimer = null;
let isRenderingPlan = false;
const autoFailedRuns = new Set();
const STUCK_RUN_MS = 30 * 60 * 1000;

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

function formatCost(value) {
  if (value === null || value === undefined) return '';
  const num = Number(value);
  if (Number.isNaN(num)) return '';
  return `$${num.toFixed(2)}`;
}

function normalizeOutputType(value) {
  return String(value || '').trim().toLowerCase().replace(/\s+/g, '_');
}

function getSelectedCollections() {
  if (!collectionPickerEl) return [];
  return Array.from(collectionPickerEl.selectedOptions || [])
    .map((opt) => opt.value)
    .filter((value) => value);
}

function modelLabel(model) {
  return model?.model_label || model?.id || 'model';
}

function modelTags(model) {
  if (!model) return [];
  const caps = model.capabilities || {};
  const tags = [];
  if (caps.image_generation) tags.push('image gen');
  const vision = String(caps.image_understanding || '').toLowerCase();
  if (vision && vision !== 'none') tags.push(`vision ${vision}`);
  if (caps.tool_use) tags.push('tools');
  const codeGen = String(caps.code_generation || '').toLowerCase();
  if (codeGen && codeGen !== 'false' && codeGen !== 'none') {
    tags.push(codeGen === 'low' ? 'code gen (low)' : 'code gen');
  }
  const codeReview = String(caps.code_review || '').toLowerCase();
  if (codeReview && codeReview !== 'false' && codeReview !== 'none') {
    tags.push(codeReview === 'low' ? 'code review (low)' : 'code review');
  }
  const jsonRel = String(caps.json_reliability || '').toLowerCase();
  if (jsonRel) tags.push(`json ${jsonRel}`);
  return tags;
}

function updateModelCache(models) {
  modelsCache = Array.isArray(models) ? models : [];
  modelsById = {};
  modelsCache.forEach((model) => {
    if (model?.id) modelsById[model.id] = model;
  });
}

function extractSection(text, label) {
  const regex = new RegExp(`${label}\\s*[:\\-]?\\s*([\\s\\S]*?)(?:\\n\\w[\\w\\s]*:|$)`, 'i');
  const match = text.match(regex);
  if (!match) return '';
  return match[1].trim();
}

function buildImagePrompt(run) {
  const outputType = normalizeOutputType(run?.artifacts?.context?.output?.type || run?.meta?.output_type || '');
  const answer = run?.consensus?.answer || '';
  if (!answer) return '';
  const visionSummary = run?.artifacts?.context?.vision_summary || '';

  if (outputType === 'image_palette') {
    const colors = (answer.match(/#[0-9a-fA-F]{3,6}/g) || [])
      .map((c) => c.toUpperCase());
    const unique = [];
    colors.forEach((c) => {
      if (!unique.includes(c)) unique.push(c);
    });
    if (!unique.length) return answer.trim();
    const prompts = unique.slice(0, 3).map((color, idx) => (
      `Variant ${idx + 1} (${color}): Using the attached kitchen photos, generate a high-fidelity photorealistic image of the same kitchen with cabinets painted ${color}. Preserve layout, materials, lighting, and camera angle. Do not change counters, floors, appliances, or decor.`
    ));
    const notes = visionSummary ? `Reference notes: ${visionSummary.slice(0, 600)}` : '';
    return `Use the attached photos.\n\n${prompts.join('\n\n')}${notes ? `\n\n${notes}` : ''}`.trim();
  }

  if (outputType === 'image_brief') {
    const prompt = extractSection(answer, 'Image prompt');
    const negative = extractSection(answer, 'Negative prompt');
    if (prompt && negative) {
      return `Prompt: ${prompt}\nNegative: ${negative}`.trim();
    }
    if (prompt) return prompt;
    return answer.trim();
  }

  if (outputType === 'web_prompt_pack') {
    const primary = extractSection(answer, 'Primary Prompt')
      || extractSection(answer, 'Primary')
      || extractSection(answer, 'Prompt');
    const attach = extractSection(answer, 'What to Attach')
      || extractSection(answer, 'Attachments')
      || extractSection(answer, 'Attach');
    const followups = extractSection(answer, 'Follow-up Prompts')
      || extractSection(answer, 'Follow-ups');
    let combined = primary || answer.trim();
    if (attach) combined += `\n\nAttachments:\n${attach}`;
    if (followups) combined += `\n\nFollow-ups:\n${followups}`;
    return combined.trim();
  }

  return '';
}

function parseArtifactPaths() {
  const raw = artifactFieldEl?.value || '';
  return raw.split('\n').map((line) => line.trim()).filter(Boolean);
}

function syncArtifactPaths(paths) {
  const unique = Array.from(new Set(paths));
  if (artifactFieldEl) artifactFieldEl.value = unique.join('\n');
  renderArtifactList(unique);
}

function renderArtifactList(paths) {
  if (!artifactListEl) return;
  if (!paths.length) {
    artifactListEl.innerHTML = '';
    return;
  }
  artifactListEl.innerHTML = paths.map((item) => {
    const label = item.split('/').pop() || item;
    return `
      <div class="artifact-item">
        <div class="artifact-name">${escapeHtml(label)}</div>
        <div class="artifact-path ui-muted">${escapeHtml(item)}</div>
        <button type="button" class="ui-button ghost small artifact-remove" data-path="${escapeHtml(item)}">Remove</button>
      </div>
    `;
  }).join('');
}

function normalizeOverrides(overrides) {
  const result = {};
  if (!overrides || typeof overrides !== 'object') return result;
  Object.entries(overrides).forEach(([role, modelId]) => {
    const key = String(role || '').trim();
    const value = String(modelId || '').trim();
    if (key && value) result[key] = value;
  });
  return result;
}

function setRoleOverrides(overrides) {
  roleOverrides = normalizeOverrides(overrides);
  schedulePlanRefresh();
  if (planCache) renderRoleOverrides(planCache);
}

function effectiveModelId(role, plan) {
  if (roleOverrides && roleOverrides[role]) return roleOverrides[role];
  return plan?.route?.plan?.[role] || '';
}

function renderRoleOverrides(plan) {
  if (!roleOverridesEl) return;
  isRenderingPlan = true;
  const route = plan?.route || {};
  const roles = Array.isArray(route.roles) ? route.roles : [];
  const assignments = route.plan || {};
  const outputMeta = plan?.output || {};
  const outputLabel = outputMeta.label || outputMeta.type || '';

  const roleSet = new Set(roles);
  Object.keys(roleOverrides || {}).forEach((role) => {
    if (!roleSet.has(role)) delete roleOverrides[role];
  });

  if (planSummaryEl) {
    if (!roles.length) {
      planSummaryEl.textContent = 'No roles configured for this agent set.';
    } else {
      const planText = roles.map((role) => {
        const modelId = assignments?.[role];
        const label = modelLabel(modelsById[modelId] || { id: modelId });
        return `${role}→${label}`;
      }).join(' · ');
      const missing = outputMeta.missing || [];
      const missingText = missing.length ? `Missing for ${outputLabel || 'output'}: ${missing.join(', ')}` : '';
      const overrideText = Object.keys(roleOverrides || {}).length ? 'Overrides active' : '';
      const parts = [`Recommended plan: ${planText}`];
      if (missingText) parts.push(missingText);
      if (overrideText) parts.push(overrideText);
      planSummaryEl.textContent = parts.join(' • ');
    }
  }

  if (!roles.length) {
    roleOverridesEl.innerHTML = '<div class="ui-muted">Select an agent set to see roles.</div>';
    isRenderingPlan = false;
    return;
  }

  const optionsByRole = roles.map((role) => {
    const recommendedId = assignments?.[role] || '';
    const recommendedLabel = modelLabel(modelsById[recommendedId] || { id: recommendedId || 'planner' });
    const selectedId = roleOverrides?.[role] || '';
    const optionTags = modelsCache.map((model) => {
      const label = modelLabel(model);
      const tags = modelTags(model);
      const tagText = tags.length ? ` • ${tags.join(', ')}` : '';
      const selected = selectedId && model.id === selectedId ? ' selected' : '';
      return `<option value="${escapeHtml(model.id)}"${selected}>${escapeHtml(label + tagText)}</option>`;
    }).join('');
    const autoSelected = selectedId ? '' : ' selected';
    const autoLabel = `Auto (recommended: ${recommendedLabel})`;
    const capId = `role-cap-${role}`;
    return `
      <div class="role-overrides-row">
        <div class="role-overrides-role">${escapeHtml(role)}</div>
        <select class="collection-picker role-select" data-role="${escapeHtml(role)}">
          <option value=""${autoSelected}>${escapeHtml(autoLabel)}</option>
          ${optionTags}
        </select>
        <div class="role-capabilities" id="${escapeHtml(capId)}"></div>
      </div>
    `;
  }).join('');

  roleOverridesEl.innerHTML = optionsByRole;
  isRenderingPlan = false;
  roles.forEach((role) => updateRoleCapabilities(role, plan));
}

function updateRoleCapabilities(role, plan) {
  const capEl = document.getElementById(`role-cap-${role}`);
  if (!capEl) return;
  const modelId = effectiveModelId(role, plan);
  if (!modelId) {
    capEl.textContent = 'Capabilities: not available';
    return;
  }
  const model = modelsById[modelId];
  const tags = modelTags(model);
  capEl.textContent = tags.length ? `Capabilities: ${tags.join(', ')}` : 'Capabilities: not reported';
}

async function refreshPlan() {
  if (!roleOverridesEl) return;
  const query = document.getElementById('query')?.value?.trim() || '';
  const outputType = outputTypeEl?.value || '';
  const useCase = useCaseEl?.value || '';
  const agentSet = agentSetEl?.value || '';
  const selectedCollections = getSelectedCollections();
  const payload = { query: query || 'general' };
  if (outputType) payload.output_type = outputType;
  if (useCase) payload.use_case = useCase;
  if (agentSet) payload.agent_set = agentSet;
  if (selectedCollections.length) payload.collections = selectedCollections;
  if (Object.keys(roleOverrides || {}).length) payload.role_overrides = roleOverrides;

  try {
    const resp = await fetchJSON('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    planCache = resp;
    if (resp?.models?.length) updateModelCache(resp.models);
    renderRoleOverrides(resp);
  } catch (err) {
    console.error('Failed to load plan:', err);
    if (planSummaryEl && !query) {
      planSummaryEl.textContent = 'Enter a question to get a plan recommendation.';
    }
  }
}

function schedulePlanRefresh() {
  if (!roleOverridesEl) return;
  if (planRefreshTimer) clearTimeout(planRefreshTimer);
  planRefreshTimer = setTimeout(() => {
    refreshPlan().catch(console.error);
  }, 250);
}

function maybeAutoFailRun(run) {
  if (!run || run.status !== 'running' || !run.id) return;
  if (autoFailedRuns.has(run.id)) return;
  const started = Date.parse(run.created_at || run.started_at || '');
  if (!started || Number.isNaN(started)) return;
  if (Date.now() - started < STUCK_RUN_MS) return;
  autoFailedRuns.add(run.id);
  fetchJSON(`/api/runs/${run.id}/fail`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason: 'stuck_running' }),
  }).catch(console.error);
}

function setSelectedRun(run) {
  selectedRun = run || null;
  renderLatest(run);
  refresh();
}

function clearSelectedRun() {
  selectedRun = null;
  refresh();
}

function setHistoryTab(tab) {
  if (!historyTabButtons.length || !historyTabPanels.length) return;
  historyTabButtons.forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle('is-active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  historyTabPanels.forEach((panel) => {
    panel.classList.toggle('is-active', panel.dataset.tab === tab);
  });
  try {
    localStorage.setItem('conclave.historyTab', tab);
  } catch (err) {
    // ignore storage errors
  }
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
  return parsed.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatDuration(ms) {
  if (!ms || Number.isNaN(ms)) return '—';
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes <= 0) return `${seconds}s`;
  return `${minutes}m ${seconds}s`;
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

function hasAgreement(run) {
  return run?.artifacts?.deliberation?.agreement === true;
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

function shouldPruneRun(run) {
  if (!run || run.status === 'running') return false;
  const answer = (run.consensus?.answer || '').trim();
  const error = (run.error || '').trim();
  if (answer || error) return false;
  return true;
}

async function pruneRuns(runs) {
  const targets = (runs || []).filter((run) => run?.id && shouldPruneRun(run) && !autoPrunedRuns.has(run.id));
  if (!targets.length) return false;
  targets.forEach((run) => autoPrunedRuns.add(run.id));
  await Promise.all(targets.map((run) => fetchJSON(`/api/runs/${run.id}`, { method: 'DELETE' }).catch(() => null)));
  return true;
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

  const parseTableRow = (row) => {
    let value = row.trim();
    if (value.startsWith('|')) value = value.slice(1);
    if (value.endsWith('|')) value = value.slice(0, -1);
    return value.split('|').map(cell => cell.trim());
  };

  const isTableSeparator = (row) => {
    const value = row.trim();
    return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(value);
  };

  const flushParagraph = () => {
    if (paragraph.length) {
      html += `<p>${renderInline(paragraph.join(' '))}</p>`;
      paragraph = [];
    }
  };

  const closeLists = () => {
    if (inUl) { html += '</ul>'; inUl = false; }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      if (inCode) { html += '</code></pre>'; inCode = false; }
      else { flushParagraph(); closeLists(); inCode = true; html += '<pre><code>'; }
      continue;
    }
    if (inCode) { html += `${escapeHtml(line)}\n`; continue; }

    // Tables
    if (trimmed.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushParagraph();
      closeLists();
      const header = parseTableRow(line);
      i += 1; // skip separator
      const bodyRows = [];
      while (i + 1 < lines.length && lines[i + 1].trim().includes('|')) {
        const next = lines[i + 1].trim();
        if (!next) break;
        i += 1;
        bodyRows.push(parseTableRow(lines[i]));
      }
      const headHtml = header.map(cell => `<th>${renderInline(cell)}</th>`).join('');
      let bodyHtml = '';
      bodyRows.forEach((row) => {
        const cells = row.map(cell => `<td>${renderInline(cell)}</td>`).join('');
        bodyHtml += `<tr>${cells}</tr>`;
      });
      html += `<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph(); closeLists();
      const level = headingMatch[1].length;
      html += `<h${level}>${renderInline(headingMatch[2])}</h${level}>`;
      continue;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (ulMatch) {
      flushParagraph();
      if (!inUl) { html += '<ul>'; inUl = true; }
      html += `<li>${renderInline(ulMatch[1])}</li>`;
      continue;
    }

    if (!trimmed) { flushParagraph(); closeLists(); continue; }
    paragraph.push(trimmed);
  }

  flushParagraph();
  closeLists();
  if (inCode) html += '</code></pre>';
  return html;
}

function extractUserInput(run) {
  const inputs = run?.artifacts?.context?.user_inputs || [];
  if (inputs.length) {
    return inputs[0].full_text || inputs[0].snippet || '';
  }
  const evidence = run?.artifacts?.context?.evidence || [];
  const userEvidence = evidence.find((item) => item.collection === 'user-input');
  return userEvidence?.snippet || '';
}

function buildContextText(run) {
  const lines = [];
  const title = run?.meta?.input_title || '';
  const outputType = run?.meta?.output_type || '';
  const promptId = run?.meta?.prompt_id || '';
  const query = run?.query || '';
  const userInputs = run?.artifacts?.context?.user_inputs || [];
  const userInput = userInputs[0]?.full_text || userInputs[0]?.snippet || extractUserInput(run);
  const inputPath = userInputs[0]?.path || '';
  const outputPath = run?.artifacts?.output_path || '';
  const inputArtifacts = run?.artifacts?.context?.input_artifacts || [];
  const evidence = run?.artifacts?.context?.evidence || [];
  const disagreements = run?.artifacts?.deliberation?.disagreements || [];
  const panelRounds = run?.artifacts?.deliberation?.panel || [];
  const agentSync = run?.artifacts?.context?.agent_sync || null;
  const answer = run?.consensus?.answer || '';
  const agreement = run?.artifacts?.deliberation?.agreement;
  const plan = run?.artifacts?.route?.plan_details || run?.artifacts?.route?.plan || {};
  const panel = run?.artifacts?.route?.panel_details
    || run?.consensus?.models_used?.panel
    || run?.artifacts?.route?.panel_models
    || [];

  lines.push(`# Recommendation Context`);
  if (title) lines.push(`Title: ${title}`);
  if (run?.id) lines.push(`Run ID: ${run.id}`);
  if (promptId) lines.push(`Prompt ID: ${promptId}`);
  if (outputType) lines.push(`Output Type: ${outputType}`);
  if (agreement !== undefined) lines.push(`Consensus: ${agreement ? 'reached' : 'not reached'}`);
  if (run?.created_at) lines.push(`Created: ${run.created_at}`);
  if (inputPath) lines.push(`Input Path: ${inputPath}`);
  if (outputPath) lines.push(`Output Path: ${outputPath}`);
  if (query) {
    lines.push(`\n## Query`);
    lines.push(query);
  }
  if (userInput) {
    lines.push(`\n## Prompt Input`);
    lines.push(userInput);
  }
  if (inputArtifacts.length) {
    lines.push(`\n## Attached Artifacts`);
    inputArtifacts.forEach((item) => {
      const name = item.title || item.path?.split('/').pop() || 'artifact';
      const path = item.path || '';
      lines.push(`- ${name}${path ? ` (${path})` : ''}`);
    });
  }
  if (plan && Object.keys(plan).length) {
    lines.push(`\n## Model Plan`);
    Object.entries(plan).forEach(([role, info]) => {
      const label = typeof info === 'object' ? (info.label || info.id || role) : info;
      lines.push(`- ${role}: ${label}`);
    });
  }
  const overrides = run?.meta?.role_overrides || run?.artifacts?.route?.role_overrides || {};
  if (overrides && Object.keys(overrides).length) {
    lines.push(`\n## Role Overrides`);
    Object.entries(overrides).forEach(([role, modelId]) => {
      lines.push(`- ${role}: ${modelId}`);
    });
  }
  if (Array.isArray(panel) && panel.length) {
    lines.push(`\n## Panel Models`);
    panel.forEach((item) => {
      const label = typeof item === 'object' ? (item.label || item.id || 'panel') : item;
      lines.push(`- ${label}`);
    });
  }
  if (agentSync && agentSync.enabled !== undefined) {
    lines.push(`\n## Agent Sync`);
    lines.push(`Enabled: ${agentSync.enabled ? 'yes' : 'no'}`);
    if (agentSync.mode) lines.push(`Mode: ${agentSync.mode}`);
    if (agentSync.contexts) {
      Object.entries(agentSync.contexts).forEach(([key, info]) => {
        const chars = info?.chars ?? '—';
        const maxChars = info?.max_chars ?? '—';
        const base = info?.include_base ? 'base+bus' : 'bus-only';
        lines.push(`- ${key}: ${chars}/${maxChars} chars (${base})`);
      });
    }
  }
  if (panelRounds.length) {
    lines.push(`\n## Panel Reviews`);
    panelRounds.forEach((round) => {
      lines.push(`\n### Round ${round.round || '?'} (${round.agreement ? 'agree' : 'disagree'})`);
      (round.reviews || []).forEach((review) => {
        const label = review.label || review.model_id || 'panel';
        if (review.skipped || review.verdict === 'skipped') {
          lines.push(`- ${label}: SKIPPED`);
        } else if (review.ok) {
          lines.push(`- ${label}: ${(review.verdict || 'disagree').toUpperCase()}`);
        } else {
          const err = review.error || 'error';
          lines.push(`- ${label}: ERROR (${err})`);
        }
      });
    });
  }
  if (evidence.length) {
    lines.push(`\n## Evidence`);
    evidence.forEach((item) => {
      const name = item.title || item.name || item.path?.split('/').pop() || 'Evidence';
      const path = item.path || '';
      const score = typeof item.signal_score === 'number' ? item.signal_score.toFixed(2) : '—';
      lines.push(`- ${name} ${path ? `(${path})` : ''} [${score}]`);
      if (item.snippet) lines.push(`  ${item.snippet}`);
    });
  }
  if (disagreements.length) {
    lines.push(`\n## Open Disagreements`);
    disagreements.forEach((item) => lines.push(`- ${item}`));
  }
  if (answer) {
    lines.push(`\n## Recommendation`);
    lines.push(answer);
  }
  return lines.join('\n');
}

function buildContextCompactText(run) {
  const title = run?.meta?.input_title || '';
  const outputType = run?.meta?.output_type || '';
  const query = run?.query || '';
  const answer = run?.consensus?.answer || '';
  const agreement = run?.artifacts?.deliberation?.agreement;
  return [
    `# Recommendation Summary`,
    title ? `Title: ${title}` : null,
    outputType ? `Output Type: ${outputType}` : null,
    agreement !== undefined ? `Consensus: ${agreement ? 'reached' : 'not reached'}` : null,
    query ? `\n## Query\n${query}` : null,
    answer ? `\n## Recommendation\n${answer}` : null,
  ].filter(Boolean).join('\n');
}

function openContextModal(run) {
  if (!contextModalEl || !contextBodyEl) return;
  activeContextRun = run;
  const title = run?.meta?.input_title || run?.query || run?.id || 'Recommendation Context';
  if (contextTitleEl) contextTitleEl.textContent = title;
  if (contextSubtitleEl) contextSubtitleEl.textContent = run?.created_at ? formatTime(run.created_at) : '';
  const contextText = buildContextText(run);
  contextBodyEl.innerHTML = renderMarkdown(contextText);
  contextModalEl.classList.remove('hidden');
}

function closeContextModal() {
  if (!contextModalEl) return;
  contextModalEl.classList.add('hidden');
  activeContextRun = null;
}

// Check if output indicates insufficient evidence
function isInsufficientEvidence(consensus) {
  if (!consensus) return false;
  return Boolean(consensus.insufficient_evidence);
}

function isRequirementsFailure(consensus) {
  if (!consensus) return false;
  if (consensus.requirements_failed) return true;
  const pope = (consensus.pope || '').toLowerCase();
  return pope.includes('required model unavailable') || pope.includes('run cancelled');
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
  const quality = run?.artifacts?.quality || {};
  if (details.evidenceCount === '?' && typeof quality.evidence_count === 'number') {
    details.evidenceCount = String(quality.evidence_count);
  }
  if (details.minEvidence === '?' && typeof quality.min_evidence === 'number') {
    details.minEvidence = String(quality.min_evidence);
  }
  if (details.maxSignal === '?' && typeof quality.max_signal_score === 'number') {
    details.maxSignal = Number(quality.max_signal_score).toFixed(2);
  }
  if (details.pdfRatio === '?' && typeof quality.pdf_ratio === 'number') {
    details.pdfRatio = Number(quality.pdf_ratio).toFixed(2);
  }
  const collections = run?.artifacts?.route?.collections || [];
  const answer = consensus?.answer || '';

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
        <strong>Try:</strong> Add supporting notes above, specify a collection, or run <code>conclave index</code> to refresh content.
      </div>
    </div>
    ${answer ? `<div class="markdown">${renderMarkdown(answer)}</div>` : ''}
  `;
}

function renderRequirementsFailure(consensus, run) {
  const answer = consensus?.answer || '';
  const requirements = run?.artifacts?.requirements || {};
  const failed = requirements.failed || [];
  let detail = '';
  let chip = '';
  const interrupted = Boolean(
    consensus?.interrupted
    || failed.some((item) => item?.interrupted)
    || failed.some((item) => String(item?.reason || '').startsWith('exit -15') || String(item?.reason || '').startsWith('exit -9'))
  );
  if (failed.length) {
    const first = failed[0] || {};
    const output = (first.output || first.stderr || '').trim();
    const snippet = output ? output.split('\n')[0] : '';
    const lower = output.toLowerCase();
    if (/(quota|limit|rate limit|too many requests|exceeded|resets)/.test(lower)) {
      chip = '<span class="status-chip quota">Quota / Limit</span>';
    }
    let resetChip = '';
    const resetMatch = output.match(/resets?\s+([0-9:]+\s*(?:am|pm)?)(?:\s*\\(([^)]+)\\))?/i);
    if (resetMatch) {
      const timePart = (resetMatch[1] || '').trim();
      const zonePart = (resetMatch[2] || '').trim();
      const resetLabel = zonePart ? `Resets ${timePart} (${zonePart})` : `Resets ${timePart}`;
      resetChip = `<span class="status-chip reset">${escapeHtml(resetLabel)}</span>`;
    }
    detail = `${first.id || 'model'}: ${first.reason || 'failed'}${snippet ? ` • ${snippet}` : ''}`;
    if (resetChip) {
      chip = `${chip} ${resetChip}`.trim();
    }
  }
  return `
    <div class="output-error">
      <div class="output-error-title">${interrupted ? 'Run Interrupted' : 'Required Model Unavailable'}</div>
      <div class="output-error-details">
        ${interrupted
          ? 'Conclave was restarting and interrupted the model check. Re-run once the service is stable.'
          : 'Conclave couldn\'t start because a required model is unavailable. Fix the missing CLI/login and re-run.'}
        ${chip ? `<div class="latest-note">${chip}</div>` : ''}
        ${detail ? `<div class="latest-note">${escapeHtml(detail)}</div>` : ''}
      </div>
    </div>
    ${answer ? `<div class="markdown">${renderMarkdown(answer)}</div>` : ''}
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

function renderDomainSummary(run) {
  if (!run) return '';
  const answer = run.consensus?.answer || '';
  if (!answer) return '';

  const summary = extractSection(answer, ['summary', 'tl;dr', 'overview']);
  const findings = extractSection(answer, [
    'findings',
    'issues',
    'results',
    'recommendations'
  ]);
  const steps = extractSection(answer, ['next steps', 'action items', 'approach']);
  const code = extractCodeBlock(answer);
  const keyBullets = findings ? [] : extractBullets(answer, /(finding|issue|result|recommendation|risk)/i, 3);

  if (!summary && !findings && !steps && !code && !keyBullets.length) return '';

  const domain = run.artifacts?.route?.domain || 'general';
  const sections = [];
  if (summary) {
    sections.push(`
      <div class="domain-section">
        <div class="domain-section-title">Summary</div>
        <div class="domain-section-body">${renderMarkdown(summary)}</div>
      </div>
    `);
  }
  if (steps || code) {
    sections.push(`
      <div class="domain-section">
        <div class="domain-section-title">Next Steps</div>
        ${steps ? `<div class="domain-section-body">${renderMarkdown(steps)}</div>` : ''}
        ${code ? `<pre class="domain-code">${escapeHtml(code)}</pre>` : ''}
      </div>
    `);
  }
  if (findings || keyBullets.length) {
    const body = findings
      ? renderMarkdown(findings)
      : `<ul>${keyBullets.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
    sections.push(`
      <div class="domain-section">
        <div class="domain-section-title">Findings</div>
        <div class="domain-section-body">${body}</div>
      </div>
    `);
  }

  return `
    <div class="domain-summary">
      <div class="domain-title">${escapeHtml(domain.charAt(0).toUpperCase() + domain.slice(1))} Summary</div>
      ${sections.join('')}
    </div>
  `;
}

function renderDomainAssets(run) {
  const assets = run?.artifacts?.domain_assets;
  if (!assets) return '';
  const sections = [];
  const target = assets.target?.id || assets.target?.name;
  if (target) {
    sections.push(`
      <div class="domain-section">
        <div class="domain-section-title">Target</div>
        <div class="domain-section-body">${escapeHtml(target)}</div>
      </div>
    `);
  }
  if (assets.reports?.length) {
    const items = assets.reports.slice(0, 6)
      .map((p) => `<li><code>${escapeHtml(p)}</code></li>`).join('');
    sections.push(`
      <div class="domain-section">
        <div class="domain-section-title">Reports</div>
        <div class="domain-section-body"><ul>${items}</ul></div>
      </div>
    `);
  }
  if (assets.working_reports?.length) {
    const items = assets.working_reports.slice(0, 6)
      .map((p) => `<li><code>${escapeHtml(p)}</code></li>`).join('');
    sections.push(`
      <div class="domain-section">
        <div class="domain-section-title">Working Reports (Detailed)</div>
        <div class="domain-section-body"><ul>${items}</ul></div>
      </div>
    `);
  }
  if (assets.locations?.length) {
    const items = assets.locations.slice(0, 8).map((loc) => {
      const prefix = loc.finding ? `${escapeHtml(loc.finding)} ` : '';
      const path = loc.path ? `${escapeHtml(loc.path)}${loc.lines ? ':' + escapeHtml(loc.lines) : ''}` : '';
      return `<li><code>${prefix}${path}</code></li>`;
    }).join('');
    sections.push(`
      <div class="domain-section">
        <div class="domain-section-title">Referenced Locations</div>
        <div class="domain-section-body"><ul>${items}</ul></div>
      </div>
    `);
  }
  if (!sections.length) return '';
  return `
    <div class="domain-summary domain-assets">
      <div class="domain-title">Artifacts</div>
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

function estimateRunTiming(run, phases) {
  const startedAt = Date.parse(run?.created_at || run?.started_at || '');
  if (!startedAt || Number.isNaN(startedAt)) {
    return { elapsedMs: 0, etaMs: null, progress: 0 };
  }
  const elapsedMs = Date.now() - startedAt;
  const total = phases.length || 1;
  const doneCount = phases.filter((phase) => phase.done).length;
  const hasActive = phases.some((phase) => phase.active);
  let progress = (doneCount + (hasActive ? 0.35 : 0)) / total;
  progress = Math.max(progress, 0.05);
  const estTotalMs = elapsedMs / progress;
  const etaMs = Math.max(0, estTotalMs - elapsedMs);
  return { elapsedMs, etaMs, progress };
}

function summarizeEvent(event, run, index) {
  if (!event) return '';
  const phase = event.phase || 'run';
  if (phase === 'model') {
    const role = event.role || 'model';
    const label = event.model_label || event.model_id || 'model';
    const duration = event.duration_ms ? `${Math.round(event.duration_ms / 100) / 10}s` : '';
    const status = event.ok === false ? 'error' : (event.ok === true ? 'ok' : '');

    let description = `${role}: ${label}`;
    if (role === 'reasoner') {
      const prior = (run?.events || []).slice(0, index).reverse().find((item) => item.phase === 'deliberate' && item.disagreements);
      const count = prior?.disagreements?.length || 0;
      description = count > 0
        ? `Reasoner revising after ${count} critique${count === 1 ? '' : 's'}`
        : 'Reasoner drafting initial response';
    } else if (role === 'critic_panel') {
      description = `Panel critique from ${label}`;
    } else if (role === 'critic') {
      description = `Primary critic review (${label})`;
    } else if (role === 'summarizer') {
      description = `Summarizer distilling consensus`;
    }

    return `${description}${duration ? ` • ${duration}` : ''}${status ? ` • ${status}` : ''}`.trim();
  }
  if (phase === 'retrieve') {
    const evidence = event.context?.evidence;
    const ragCount = event.context?.rag;
    const details = [];
    if (typeof evidence === 'number') details.push(`evidence ${evidence}`);
    if (typeof ragCount === 'number') details.push(`rag ${ragCount}`);
    return `Retrieval completed${details.length ? ` • ${details.join(', ')}` : ''}`.trim();
  }
  if (phase === 'route') {
    return event.status === 'done' ? 'Routing complete' : 'Routing models';
  }
  if (phase === 'quality') {
    const count = event.evidence_count;
    const signal = event.max_signal_score;
    const details = [];
    if (typeof count === 'number') details.push(`evidence ${count}`);
    if (typeof signal === 'number') details.push(`signal ${signal.toFixed(2)}`);
    return `Quality check${details.length ? ` • ${details.join(', ')}` : ''}`.trim();
  }
  if (phase === 'deliberate') {
    const roundLabel = event.round ? `Round ${event.round}` : 'Deliberation';
    const label = event.model_label || event.model_id || '';
    const duration = typeof event.duration_s === 'number' ? `${event.duration_s.toFixed(1)}s` : '';
    const summary = (event.summary || '').trim();
    const disagreementCount = (event.disagreements || []).length;
    if (event.status === 'round_start') {
      return `${roundLabel}: started`;
    }
    if (event.status === 'reasoner_start') {
      return `${roundLabel}: reasoner thinking${label ? ` (${label})` : ''}`;
    }
    if (event.status === 'reasoner_done') {
      return `${roundLabel}: reasoner draft${duration ? ` • ${duration}` : ''}${summary ? ` • ${summary}` : ''}`;
    }
    if (event.status === 'critic_start') {
      return `${roundLabel}: critic reviewing${label ? ` (${label})` : ''}`;
    }
    if (event.status === 'critic_done') {
      const verdict = event.verdict ? event.verdict.toUpperCase() : '';
      return `${roundLabel}: critic ${verdict || 'done'}${duration ? ` • ${duration}` : ''}${summary ? ` • ${summary}` : ''}`;
    }
    if (event.status === 'panel_start') {
      return `${roundLabel}: panel reviewing${label ? ` (${label})` : ''}`;
    }
    if (event.status === 'panel_done') {
      const verdict = event.verdict ? event.verdict.toUpperCase() : '';
      return `${roundLabel}: panel ${verdict || 'done'}${duration ? ` • ${duration}` : ''}${summary ? ` • ${summary}` : ''}`;
    }
    if (event.status === 'round_result') {
      const smoke = event.agreement ? '\u2601\ufe0f white smoke' : '\u2587\u2587 black smoke';
      const verdict = event.agreement ? 'AGREE' : 'DISAGREE';
      const issues = disagreementCount ? ` • ${disagreementCount} issue${disagreementCount === 1 ? '' : 's'}` : '';
      return `${roundLabel}: ${smoke} ${verdict}${issues}`;
    }
    if (event.status === 'stable') {
      return `${roundLabel}: \u2587\u2587 black smoke — stable disagreement`;
    }
    if (event.status === 'stop') {
      return `${roundLabel}: stopping (${event.reason || 'done'})`;
    }
    if (event.agreement === true) {
      return `${roundLabel}: consensus reached`;
    }
    if (disagreementCount > 0) {
      return `${roundLabel}: critic pushed back on ${disagreementCount} point${disagreementCount === 1 ? '' : 's'}`;
    }
    if (event.status === 'start') {
      return `${roundLabel}: panel reviewing`;
    }
    return `${roundLabel}: continuing negotiation`;
  }
  if (phase === 'requirements') {
    return event.status === 'ok' ? 'Requirements satisfied' : 'Checking requirements';
  }
  if (phase === 'preflight') {
    return 'Preflight checks';
  }
  return `${phase} ${event.status || ''}`.trim();
}

function renderLiveProgress(run) {
  if (!liveProgressEl) return;
  if (!run || run.status !== 'running') {
    liveProgressEl.classList.add('hidden');
    liveProgressEl.innerHTML = '';
    return;
  }
  const phases = buildPhaseState(run);
  const active = phases.find((phase) => phase.active) || phases.find((phase) => !phase.done);
  const timing = estimateRunTiming(run, phases);
  const percent = Math.min(99, Math.max(2, Math.round(timing.progress * 100)));
  const allEvents = run.events || [];
  const recent = allEvents.slice(-6);
  const offset = allEvents.length - recent.length;
  const events = recent.map((event, idx) => {
    const globalIndex = offset + idx;
    const timestamp = event.timestamp ? formatTimeShort(event.timestamp) : '';
    const summary = summarizeEvent(event, run, globalIndex);
    return `
      <div class="live-event">
        <span class="live-event-time">${escapeHtml(timestamp)}</span>
        <span class="live-event-detail">${escapeHtml(summary || 'Update received')}</span>
      </div>
    `;
  }).join('');

  liveProgressEl.innerHTML = `
    <div class="live-progress-header">
      <div class="live-progress-title">Live Updates</div>
      <div class="live-progress-meta">
        <span>Phase: ${escapeHtml(active?.label || 'Queued')}</span>
        <span>Elapsed: ${formatDuration(timing.elapsedMs)}</span>
        <span>ETA: ${timing.etaMs === null ? '—' : formatDuration(timing.etaMs)}</span>
        <span>Progress: ${percent}%</span>
      </div>
    </div>
    <div class="live-progress-bar">
      <span style="width: ${percent}%"></span>
    </div>
    <div class="live-progress-events">
      ${events || '<div class="live-event muted">Waiting for updates...</div>'}
    </div>
  `;
  liveProgressEl.classList.remove('hidden');
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

function summarizeDecision(run) {
  const answer = run?.consensus?.answer || '';
  if (!answer) return 'No output.';
  const lines = answer.split('\n').map((line) => line.trim()).filter(Boolean);
  return lines.slice(0, 2).join(' ').slice(0, 220);
}

async function renderReconcilePanel(run) {
  if (!reconcilePanelEl) return;
  if (!run) {
    reconcilePanelEl.classList.add('hidden');
    reconcilePanelEl.innerHTML = '';
    return;
  }
  const promptId = run.meta?.prompt_id;
  const reconcile = run.artifacts?.reconcile || {};
  let previousRunId = reconcile.previous_run_id;
  if (previousRunId && typeof previousRunId === 'object') {
    previousRunId = previousRunId.id || previousRunId.run_id || null;
  }
  let priorRuns = [];

  if (promptId) {
    const cached = (runsCache || []).filter((item) => item?.meta?.prompt_id === promptId && item?.id !== run.id);
    priorRuns = cached;
    if (priorRuns.length < 3) {
      try {
        const extra = await fetchJSON('/api/runs?limit=20');
        priorRuns = (extra.runs || []).filter((item) => item?.meta?.prompt_id === promptId && item?.id !== run.id);
      } catch (err) {
        // ignore additional fetch failure
      }
    }
  }

  if (previousRunId && !priorRuns.find((item) => item?.id === previousRunId)) {
    try {
      const prev = await fetchJSON(`/api/runs/${previousRunId}`);
      if (prev) priorRuns.unshift(prev);
    } catch (err) {
      // ignore missing previous run
    }
  }

  priorRuns = priorRuns
    .filter((item) => item?.id)
    .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
    .slice(0, 3);

  if (!priorRuns.length) {
    reconcilePanelEl.classList.add('hidden');
    reconcilePanelEl.innerHTML = '';
    return;
  }

  const heading = previousRunId
    ? `Reconciled with ${previousRunId}`
    : 'Prior Decisions';
  const rows = priorRuns.map((item) => {
    const title = item.meta?.input_title || item.query || item.id;
    const time = formatTime(item.completed_at || item.created_at);
    const outputType = item.meta?.output_type ? `• ${item.meta.output_type}` : '';
    const summary = summarizeDecision(item);
    return `
      <div class="reconcile-item">
        <div class="reconcile-item-header">
          <div class="reconcile-item-title">${escapeHtml(title)}</div>
          <div class="reconcile-item-meta ui-mono">${escapeHtml(time)} ${escapeHtml(outputType)}</div>
        </div>
        <div class="reconcile-item-summary">${escapeHtml(summary)}</div>
        <div class="reconcile-item-actions">
          <button type="button" class="ui-button ghost small" data-action="open" data-run-id="${escapeHtml(item.id)}">Open</button>
        </div>
      </div>
    `;
  }).join('');

  reconcilePanelEl.innerHTML = `
    <div class="reconcile-header">
      <div class="reconcile-title">${escapeHtml(heading)}</div>
      <div class="reconcile-sub ui-muted">Current run is the latest, with prior results reconciled.</div>
    </div>
    <div class="reconcile-list">${rows}</div>
  `;
  reconcilePanelEl.classList.remove('hidden');
}

function buildPendingRun(query, outputType, inputTitle) {
  return {
    id: 'pending',
    status: 'running',
    created_at: new Date().toISOString(),
    query: query || '',
    meta: {
      output_type: outputType || '',
      input_title: inputTitle || '',
    },
    events: [],
  };
}

function primeRunningView(run) {
  selectedRun = null;
  latestRunCache = run || null;
  if (clearSelectionBtn) clearSelectionBtn.classList.add('hidden');
  if (run) {
    renderLatest(run);
  }
}

// Render models used
function renderModels(latest) {
  if (!latestModelsEl) return;

  const models = latest?.artifacts?.route?.plan_details || latest?.artifacts?.route?.plan || {};
  const panelModels = latest?.artifacts?.route?.panel_details
    || latest?.consensus?.models_used?.panel
    || latest?.artifacts?.route?.panel_models
    || [];
  if (!Object.keys(models).length && (!panelModels || panelModels.length === 0)) {
    latestModelsEl.innerHTML = '';
    return;
  }

  const tags = Object.entries(models).map(([role, info]) => {
    const label = typeof info === 'object' ? (info.label || info.id || role) : info;
    return `<span class="model-tag"><span class="model-tag-role">${escapeHtml(role)}</span><span class="model-tag-name">${escapeHtml(label)}</span></span>`;
  }).join('');

  let panelTags = '';
  if (Array.isArray(panelModels) && panelModels.length) {
    panelTags = panelModels.map((item) => {
      const label = typeof item === 'object' ? (item.label || item.id || 'panel') : item;
      return `<span class="model-tag"><span class="model-tag-role">panel</span><span class="model-tag-name">${escapeHtml(label)}</span></span>`;
    }).join('');
  }

  latestModelsEl.innerHTML = tags + panelTags;
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

function renderOutputArtifacts(run) {
  if (!latestArtifactsEl) return;
  const artifacts = run?.artifacts?.output_artifacts || [];
  const outputMeta = run?.artifacts?.context?.output || {};
  if (!artifacts.length) {
    latestArtifactsEl.classList.add('hidden');
    latestArtifactsEl.innerHTML = '';
    latestImagePrompt = '';
    return;
  }
  const outputLabel = outputMeta?.label || outputMeta?.type || run?.meta?.output_type || '';
  const missing = outputMeta?.missing || [];
  const required = outputMeta?.requires || [];
  const chips = [];
  if (outputLabel) {
    chips.push(`<span class="artifact-chip ok">Output: ${escapeHtml(outputLabel)}</span>`);
  }
  required.forEach((cap) => {
    chips.push(`<span class="artifact-chip ok">${escapeHtml(cap)}</span>`);
  });
  missing.forEach((cap) => {
    chips.push(`<span class="artifact-chip warn">Missing: ${escapeHtml(cap)}</span>`);
  });
  const listHtml = artifacts.map((item) => {
    const name = item.name || '';
    if (!name) return '';
    const href = `/api/runs/${encodeURIComponent(run.id)}/artifacts/${encodeURIComponent(name)}`;
    const kind = item.kind ? `<span class="ui-muted">${escapeHtml(item.kind)}</span>` : '';
    return `
      <div class="artifact-item">
        <div class="artifact-name">${escapeHtml(name)}</div>
        <div class="artifact-actions">
          ${kind}
          <a class="ui-button ghost small" href="${href}" target="_blank" rel="noopener">Download</a>
        </div>
      </div>
    `;
  }).join('');
  const note = missing.length
    ? `<div class="artifact-note">Missing capabilities detected. Generated best-effort artifacts only.</div>`
    : '';
  const outputType = normalizeOutputType(outputMeta?.type || run?.meta?.output_type || '');
  const imagePrompt = buildImagePrompt(run);
  latestImagePrompt = imagePrompt || '';
  const buttonLabel = outputType === 'web_prompt_pack' ? 'Copy Web Prompt' : 'Copy Image Prompt';
  const imageButton = imagePrompt
    ? `<button type="button" class="ui-button ghost small copy-image-prompt">${buttonLabel}</button>`
    : '';
  latestArtifactsEl.innerHTML = `
    <div class="artifact-header">
      <div class="artifact-title">Artifacts</div>
      <div class="artifact-meta">${chips.join('')} ${imageButton}</div>
    </div>
    <div class="artifact-list">${listHtml}</div>
    ${note}
  `;
  latestArtifactsEl.classList.remove('hidden');
}

// Render Latest Decision
function renderLatest(latest) {
  if (selectedRun && latest?.id !== selectedRun.id) {
    return renderLatest(selectedRun);
  }
  latestRunCache = latest || null;
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
    if (latestArtifactsEl) {
      latestArtifactsEl.innerHTML = '';
      latestArtifactsEl.classList.add('hidden');
    }
    renderLiveProgress(null);
    renderReconcilePanel(null);
    if (latestTitleEl) latestTitleEl.textContent = 'Latest Decision';
    if (clearSelectionBtn) clearSelectionBtn.classList.add('hidden');
    renderProgress(null);
    return;
  }

  maybeAutoFailRun(latest);

  const startedAt = latest.created_at || latest.started_at || '';
  const endedAt = latest.completed_at || latest.created_at || '';
  const startedMs = Date.parse(startedAt);
  const endedMs = Date.parse(endedAt);
  if (latest.status === 'running' && startedAt && !Number.isNaN(startedMs)) {
    const elapsed = formatDuration(Date.now() - startedMs);
    latestTimeEl.textContent = `Started ${formatTime(startedAt)} • Elapsed ${elapsed}`;
  } else if (startedAt && endedAt && !Number.isNaN(startedMs) && !Number.isNaN(endedMs)) {
    const duration = formatDuration(endedMs - startedMs);
    latestTimeEl.textContent = `${formatTime(endedAt)} • Duration ${duration}`;
  } else {
    latestTimeEl.textContent = formatTime(latest.completed_at || latest.created_at);
  }
  renderLiveProgress(latest);
  renderReconcilePanel(latest).catch(console.error);

  if (latestPromptEl) {
    const prompt = latest.query || '';
    const outputType = latest.meta?.output_type || '';
    const promptText = prompt ? `"${prompt.slice(0, 100)}${prompt.length > 100 ? '...' : ''}"` : '';
    const outputHtml = outputType ? `<span class="latest-output">Output: ${escapeHtml(outputType)}</span>` : '';
    const costValue = latest?.artifacts?.cost_estimate?.total_usd;
    const costHtml = costValue !== undefined ? `<span class="latest-cost">Est. ${formatCost(costValue)}</span>` : '';
    latestPromptEl.innerHTML = `${escapeHtml(promptText)} ${outputHtml} ${costHtml}`.trim();
  }

  // Render models used
  renderModels(latest);

  if (latest.status === 'running') {
    latestEl.innerHTML = `
      <div class="empty-state">
        <p>Running consensus...</p>
        <p class="ui-muted">Live updates and ETA appear above as Conclave works.</p>
      </div>
    `;
    if (latestEvidenceEl) latestEvidenceEl.innerHTML = '';
    if (latestArtifactsEl) {
      latestArtifactsEl.innerHTML = '';
      latestArtifactsEl.classList.add('hidden');
    }
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
    if (latestArtifactsEl) {
      latestArtifactsEl.innerHTML = '';
      latestArtifactsEl.classList.add('hidden');
    }
    renderProgress(latest);
    return;
  }

  if (!latest.consensus) {
    latestEl.innerHTML = `<div class="empty-state">No consensus generated.</div>`;
    if (latestEvidenceEl) latestEvidenceEl.innerHTML = '';
    if (latestArtifactsEl) {
      latestArtifactsEl.innerHTML = '';
      latestArtifactsEl.classList.add('hidden');
    }
    renderProgress(latest);
    return;
  }

  // Render evidence panel (hidden by default)
  renderEvidence(latest);
  renderOutputArtifacts(latest);

  const agreement = latest?.artifacts?.deliberation?.agreement;
  const panelRounds = latest?.artifacts?.deliberation?.panel || [];
  let panelHtml = '';
  let disagreementHtml = '';
  let panelNotesHtml = '';
  if (panelRounds.length) {
    const latestPanel = panelRounds[panelRounds.length - 1].reviews || [];
    const items = latestPanel.map((item) => {
      const label = item.label || item.model_id || 'panel';
      const rawVerdict = (item.verdict || (item.skipped ? 'skipped' : 'disagree'));
      const verdict = rawVerdict.toUpperCase();
      const isAgree = rawVerdict.toLowerCase() === 'agree';
      const isSkipped = rawVerdict.toLowerCase() === 'skipped';
      const verdictClass = isAgree ? 'verdict-agree' : isSkipped ? 'verdict-skipped' : 'verdict-disagree';
      return `<li><span class="verdict-label">${escapeHtml(label)}</span> <span class="verdict-badge ${verdictClass}">${escapeHtml(verdict)}</span></li>`;
    }).join('');
    panelHtml = `
      <div class="consensus-panel">
        <div class="consensus-panel-title">Panel Verdicts</div>
        <ul class="consensus-panel-list">${items}</ul>
      </div>
    `;

    const aggregated = latest?.artifacts?.deliberation?.disagreements || [];
    if (aggregated.length) {
      const sources = new Map();
      latestPanel.forEach((item) => {
        const label = item.label || item.model_id || 'panel';
        (item.disagreements || []).forEach((text) => {
          if (!text) return;
          const key = text.trim();
          if (!sources.has(key)) sources.set(key, []);
          sources.get(key).push(label);
        });
      });
      // Separate resolved (✅) from truly open disagreements
      const resolved = [];
      const open = [];
      aggregated.slice(0, 8).forEach((text) => {
        if (text.startsWith('✅') || text.startsWith('✓')) {
          resolved.push(text);
        } else {
          open.push(text);
        }
      });
      let sections = '';
      if (open.length) {
        const openRows = open.map((text) => {
          const people = sources.get(text) || [];
          const suffix = people.length ? ` — ${people.join(', ')}` : '';
          return `<li>${escapeHtml(text)}${escapeHtml(suffix)}</li>`;
        }).join('');
        sections += `
          <div class="consensus-disagreements">
            <div class="consensus-panel-title">Open Disagreements</div>
            <ul class="consensus-panel-list">${openRows}</ul>
          </div>
        `;
      }
      if (resolved.length) {
        const resolvedRows = resolved.map((text) => {
          return `<li>${escapeHtml(text)}</li>`;
        }).join('');
        sections += `
          <div class="consensus-resolved">
            <div class="consensus-panel-title">Resolved</div>
            <ul class="consensus-panel-list">${resolvedRows}</ul>
          </div>
        `;
      }
      disagreementHtml = sections;
    }

    // Only show panel notes for models that actually disagreed
    const disagreedPanels = latestPanel.filter((item) => {
      const v = (item.verdict || '').toLowerCase();
      return v !== 'agree' && !item.skipped;
    });
    const noteItems = (disagreedPanels.length ? disagreedPanels : latestPanel).map((item, idx) => {
      const text = (item.text || '').trim();
      if (!text) return '';
      const label = item.label || item.model_id || 'panel';
      const rawVerdict = (item.verdict || (item.skipped ? 'skipped' : 'disagree'));
      const verdict = rawVerdict.toUpperCase();
      const noteId = `panel-note-${latest.id || 'latest'}-${idx}`;
      return `
        <div class="panel-note-card">
          <div class="panel-note-header">
            <div class="panel-note-title">${escapeHtml(label)} <span class="verdict-badge verdict-${rawVerdict.toLowerCase() === 'agree' ? 'agree' : rawVerdict.toLowerCase() === 'skipped' ? 'skipped' : 'disagree'}">${escapeHtml(verdict)}</span></div>
            <button type="button" class="ui-button ghost small panel-note-toggle" data-note-id="${escapeHtml(noteId)}">Show notes</button>
          </div>
          <div class="panel-note-body hidden" id="${escapeHtml(noteId)}">
            <div class="markdown">${renderMarkdown(text)}</div>
          </div>
        </div>
      `;
    }).join('');

    if (noteItems) {
      const notesTitle = disagreedPanels.length ? 'Panel Notes' : 'Panel Notes';
      panelNotesHtml = `
        <div class="panel-notes">
          <div class="consensus-panel-title">${notesTitle}</div>
          ${noteItems}
        </div>
      `;
    }
  }
  let agreementHtml = '';
  if (agreement !== undefined) {
    const agreeCount = panelRounds.length
      ? (panelRounds[panelRounds.length - 1].reviews || []).filter(r => (r.verdict || '').toLowerCase() === 'agree').length
      : 0;
    const totalCount = panelRounds.length
      ? (panelRounds[panelRounds.length - 1].reviews || []).length
      : 0;
    const consensusNote = totalCount ? ` (${agreeCount}/${totalCount} agree)` : '';
    const status = agreement ? 'Consensus reached' : 'Consensus not reached';
    const statusClass = agreement ? 'consensus-yes' : 'consensus-no';
    agreementHtml = `<div class="consensus-status ${statusClass}">${escapeHtml(status)}${escapeHtml(consensusNote)}</div>`;
  }

  const isSelected = Boolean(selectedRun);
  if (latestTitleEl) {
    latestTitleEl.textContent = isSelected ? 'Selected Decision' : 'Latest Decision';
  }
  if (clearSelectionBtn) {
    clearSelectionBtn.classList.toggle('hidden', !isSelected);
  }

  const userInput = extractUserInput(latest);
  const outputBlock = `
    <div class="latest-col">
      <div class="latest-col-title">Output</div>
      <div class="markdown">${renderMarkdown(latest.consensus.answer || '')}</div>
    </div>
  `;

  if (isRequirementsFailure(latest.consensus)) {
    latestEl.innerHTML = agreementHtml + panelHtml + disagreementHtml + panelNotesHtml + renderRequirementsFailure(latest.consensus, latest);
  } else if (isInsufficientEvidence(latest.consensus)) {
    latestEl.innerHTML = agreementHtml + panelHtml + disagreementHtml + panelNotesHtml + renderInsufficientEvidence(latest.consensus, latest);
  } else {
    latestEl.innerHTML = `
      <div class="output-success">
        ${agreementHtml}
        ${panelHtml}
        ${disagreementHtml}
        ${panelNotesHtml}
        ${outputBlock}
      </div>
    `;
  }

  renderProgress(latest);
}

// Render Run History as cards
function renderRuns(runs) {
  if (!runListEl) return;

  (runs || []).forEach((run) => maybeAutoFailRun(run));
  pruneRuns(runs).then((deleted) => {
    if (deleted) {
      setTimeout(() => refresh(), 500);
    }
  }).catch(() => null);

  const visibleRuns = (runs || []).filter((run) => !shouldPruneRun(run));

  if (runCountEl) {
    const runningCount = visibleRuns.filter((run) => run.status === 'running').length;
    const runningLabel = runningCount ? ` • ${runningCount} running` : '';
    runCountEl.textContent = `${visibleRuns.length} run${visibleRuns.length !== 1 ? 's' : ''}${runningLabel}`;
  }

  if (!visibleRuns.length) {
    runListEl.innerHTML = `<div class="empty-state">No runs yet. Ask a question to see history here.</div>`;
    return;
  }

  runListEl.innerHTML = visibleRuns.map((run) => {
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
    const domainSummary = renderDomainSummary(run);
    const domainAssets = renderDomainAssets(run);

    const selectedClass = selectedRun?.id === run.id ? ' is-selected' : '';
    return `
      <div class="run-card ${cardClass}${selectedClass}" data-run-id="${escapeHtml(run.id || '')}" data-run-title="${escapeHtml(title)}" data-run-query="${escapeHtml(run.query || '')}" data-run-output="${escapeHtml(outputType)}">
        <button class="card-delete" data-run-id="${escapeHtml(run.id || '')}" aria-label="Delete run">×</button>
        <div class="run-card-main">
          <div class="run-card-title">
            ${escapeHtml(title.slice(0, 60))}${title.length > 60 ? '...' : ''}
            ${statusChip}
            ${outputChip}
          </div>
          <div class="run-card-query">${escapeHtml(run.query || '')}</div>
          ${progressHtml}
          ${domainSummary}
          ${domainAssets}
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
            <button class="ui-button ghost small view-context" data-run-id="${escapeHtml(run.id || '')}">Context</button>
            <button class="ui-button ghost small edit-run" data-run-id="${escapeHtml(run.id || '')}">Edit</button>
            <button class="ui-button ghost small rerun-run" data-run-id="${escapeHtml(run.id || '')}">Rerun</button>
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
  const query = run.query || '';
  const inputText = extractUserInput(run);
  const modelEvents = events.filter((event) => event.phase === 'model' && event.role);
  const roleMap = {};
  modelEvents.forEach((event) => {
    if (!roleMap[event.role]) {
      roleMap[event.role] = event.model_label || event.model_id;
    }
  });
  const modelItems = Object.entries(roleMap).map(([role, model]) => {
    return `<li>${escapeHtml(role)} → ${escapeHtml(model || 'unknown')}</li>`;
  }).join('');

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
  const evidenceItems = evidence.map((item) => {
    const name = item.title || item.name || item.path?.split('/').pop() || 'Evidence';
    const score = typeof item.signal_score === 'number' ? item.signal_score.toFixed(2) : '—';
    return `<li>${escapeHtml(name)} <span class="ui-muted">[${score}]</span></li>`;
  }).join('');

  const metaItems = [];
  if (outputType) metaItems.push(`<li>Output: ${escapeHtml(outputType)}</li>`);
  const cost = run.artifacts?.cost_estimate?.total_usd;
  if (cost !== undefined) metaItems.push(`<li>Estimated cost: ${escapeHtml(formatCost(cost))}</li>`);

  return `
    ${query ? `
    <div class="log-section">
      <div class="log-section-title">Query</div>
      <div class="log-body">${escapeHtml(query)}</div>
    </div>
    ` : ''}
    ${inputText ? `
    <div class="log-section">
      <div class="log-section-title">Prompt Input</div>
      <div class="log-body">${escapeHtml(inputText)}</div>
    </div>
    ` : ''}
    ${metaItems.length ? `
    <div class="log-section">
      <div class="log-section-title">Run Meta</div>
      <ul class="log-list">${metaItems.join('')}</ul>
    </div>
    ` : ''}
    ${modelItems ? `
    <div class="log-section">
      <div class="log-section-title">Models</div>
      <ul class="log-list">${modelItems}</ul>
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
    ${run.consensus?.answer ? `
    <div class="log-section">
      <div class="log-section-title">Output</div>
      <div class="markdown">${renderMarkdown(run.consensus.answer)}</div>
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

  const sorted = [...prompts].sort((a, b) => {
    const aRun = a.latest_run || {};
    const bRun = b.latest_run || {};
    const aTime = aRun.completed_at || aRun.created_at || a.updated_at || '';
    const bTime = bRun.completed_at || bRun.created_at || b.updated_at || '';
    return (bTime || '').localeCompare(aTime || '');
  });

  promptListEl.innerHTML = sorted.map((prompt) => {
    const title = prompt.title || prompt.query?.slice(0, 40) || prompt.id;
    const outputType = prompt.output_type ? ` • Output: ${prompt.output_type}` : '';
    const latest = prompt.latest_run || null;
    if (latest) maybeAutoFailRun(latest);
    const latestAnswer = latest?.consensus?.answer || '';
    const latestStatus = latest?.status;
    const latestAgreement = latest?.artifacts?.deliberation?.agreement;
    const latestTime = latest?.completed_at || latest?.created_at;
    let latestLabel = '';
    if (latestStatus === 'failed') latestLabel = 'Failed';
    else if (latestStatus === 'running') latestLabel = 'Running';
    else if (latestAgreement === true) latestLabel = 'Consensus reached';
    else if (latestAgreement === false) latestLabel = 'Consensus not reached';
    else if (latestStatus) latestLabel = latestStatus;
    const latestMeta = latestTime ? `Last run ${formatTime(latestTime)}${latestLabel ? ' • ' + latestLabel : ''}` : 'No runs yet';
    const latestPreview = latestAnswer ? renderAnswerPreview(latestAnswer) : '';
    const latestError = latest?.error || '';
    const statusChip = latestStatus ? `<span class="status-chip ${escapeHtml(latestStatus)}">${escapeHtml(latestStatus)}</span>` : '';
    const consensusChip = latestAgreement === true
      ? `<span class="status-chip consensus-yes">Consensus</span>`
      : (latestAgreement === false ? `<span class="status-chip consensus-no">No Consensus</span>` : '');
    const selectedClass = latest?.id && selectedRun?.id === latest.id ? ' is-selected' : '';
    return `
      <div class="prompt-card${selectedClass}" data-prompt-id="${escapeHtml(prompt.id)}" data-run-id="${escapeHtml(latest?.id || '')}">
        <button class="card-delete" data-action="delete" data-id="${prompt.id}" aria-label="Delete saved query">×</button>
        <div class="prompt-card-header">
          <div class="prompt-card-title">${escapeHtml(title)}</div>
          <div class="prompt-card-chips">
            ${statusChip}
            ${consensusChip}
          </div>
        </div>
        <div class="prompt-card-meta">Updated ${formatTime(prompt.updated_at)}${escapeHtml(outputType)}</div>
        <div class="prompt-card-query">${escapeHtml(prompt.query || '')}</div>
        <div class="prompt-card-meta">${escapeHtml(latestMeta)}</div>
        ${latestPreview ? `<div class="prompt-card-result">${latestPreview}</div>` : ''}
        ${!latestPreview && latestError ? `<div class="prompt-card-result error">${escapeHtml(latestError)}</div>` : ''}
        <div class="prompt-card-actions">
          <button class="ui-button small" data-action="load" data-id="${prompt.id}">Edit</button>
          <button class="ui-button small primary" data-action="run" data-id="${prompt.id}">Run</button>
          ${latest?.id ? `<button class="ui-button ghost small" data-action="view" data-run-id="${latest.id}">Open</button>` : ''}
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

// Load output types for picker
async function loadOutputTypes() {
  if (outputsLoaded || !outputTypeEl) return;
  try {
    const resp = await fetchJSON('/api/outputs');
    const outputs = resp.outputs || [];
    if (!outputs.length) return;
    const current = outputTypeEl.value;
    let html = '<option value="">Auto (report)</option>';
    outputs.forEach((item) => {
      if (!item?.id) return;
      html += `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label || item.id)}</option>`;
    });
    outputTypeEl.innerHTML = html;
    if (current) outputTypeEl.value = current;
    outputsLoaded = true;
  } catch (err) {
    console.error('Failed to load output types:', err);
  }
}

async function loadUseCases() {
  if (useCasesLoaded || !useCaseEl) return;
  try {
    const resp = await fetchJSON('/api/use-cases');
    const cases = resp.use_cases || [];
    if (!cases.length) return;
    useCasesCache = cases;
    const current = useCaseEl.value;
    let html = '<option value="">Auto (general)</option>';
    cases.forEach((item) => {
      if (!item?.id) return;
      html += `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label || item.id)}</option>`;
    });
    useCaseEl.innerHTML = html;
    if (current) useCaseEl.value = current;
    useCasesLoaded = true;
  } catch (err) {
    console.error('Failed to load use cases:', err);
  }
}

async function loadAgentSets() {
  if (agentSetsLoaded || !agentSetEl) return;
  try {
    const resp = await fetchJSON('/api/agent-sets');
    const sets = resp.agent_sets || [];
    if (!sets.length) return;
    agentSetsCache = sets;
    const current = agentSetEl.value;
    let html = '<option value="">Auto (default)</option>';
    sets.forEach((item) => {
      if (!item?.id) return;
      html += `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label || item.id)}</option>`;
    });
    agentSetEl.innerHTML = html;
    if (current) agentSetEl.value = current;
    agentSetsLoaded = true;
  } catch (err) {
    console.error('Failed to load agent sets:', err);
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
    runsCache = runs.runs || [];
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
  loadOutputTypes();
  loadUseCases();
  loadAgentSets();
  schedulePlanRefresh();

  if (!runListTimer) {
    runListTimer = setInterval(async () => {
      try {
        const runs = await fetchJSON('/api/runs?limit=10');
        runsCache = runs.runs || [];
        renderRuns(runs.runs || []);
      } catch (err) {
        console.error('Runs refresh failed:', err);
      }
    }, 5000);
  }
}

// Run management
async function startRun(query) {
  const outputType = outputTypeEl?.value || '';
  const inputTitle = document.getElementById('input-title').value.trim();
  primeRunningView(buildPendingRun(query, outputType, inputTitle));
  setStatus('Running...', 'running');
  setSmoke(false);
  renderProgress({ status: 'running', events: [] });
  setRunAction(
    'Running new decision',
    `Source: prompt editor • ${summarizeQuery(query)}${outputType ? ' • Output: ' + outputType : ''}`,
    'running'
  );

  const inputNotes = document.getElementById('input-notes').value.trim();
  const inputArtifacts = parseArtifactPaths();
  const useCase = useCaseEl?.value || '';
  const agentSet = agentSetEl?.value || '';
  const overrides = roleOverrides || {};

  // Get selected collections
  let selectedCollections = [];
  if (collectionPickerEl) {
    selectedCollections = Array.from(collectionPickerEl.selectedOptions)
      .map(opt => opt.value)
      .filter(v => v); // Filter out empty "auto-detect" option
  }

  let inputId = null;
  if (inputNotes || inputArtifacts.length || inputTitle) {
    const inputResp = await fetchJSON('/api/inputs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: inputTitle,
        content: inputNotes,
        question: query,
        artifacts: inputArtifacts,
        output_type: outputType,
        use_case: useCase,
        agent_set: agentSet,
        role_overrides: overrides,
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
  if (useCase) payload.use_case = useCase;
  if (agentSet) payload.agent_set = agentSet;
  if (Object.keys(overrides).length) payload.role_overrides = overrides;

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
  // Clean up existing connections
  if (pollTimer) clearInterval(pollTimer);
  if (runWebSocket) {
    runWebSocket.close();
    runWebSocket = null;
  }

  // Try WebSocket first for real-time updates
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/run/${runId}`;

  try {
    runWebSocket = new WebSocket(wsUrl);

    runWebSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'state' || data.type === 'complete') {
          const run = data.run;
          if (run.status === 'complete') {
            setSmoke(hasAgreement(run));
            setStatus('Complete', 'success');
            renderLatest(run);
            refresh();
            setRunActionEphemeral('Run complete', 'Latest Decision updated.', 'success');
            runWebSocket?.close();
          } else if (run.status === 'failed') {
            setStatus('Failed', 'error');
            renderLatest(run);
            refresh();
            setRunActionEphemeral('Run failed', run.error || 'See process logs for details.', 'error', 5000);
            runWebSocket?.close();
          } else {
            renderLatest(run);
          }
        } else if (data.type === 'event') {
          const incoming = data.event;
          if (!latestRunCache || latestRunCache.id !== runId) {
            latestRunCache = latestRunCache && latestRunCache.status === 'running'
              ? latestRunCache
              : { id: runId, status: 'running', created_at: new Date().toISOString(), events: [] };
          }
          latestRunCache.events = [...(latestRunCache.events || []), incoming];
          renderProgress(latestRunCache);
          renderLiveProgress(latestRunCache);
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    runWebSocket.onerror = () => {
      console.log('WebSocket error, falling back to polling');
      runWebSocket = null;
      startPolling(runId);
    };

    runWebSocket.onclose = () => {
      runWebSocket = null;
    };

  } catch (err) {
    console.log('WebSocket failed, using polling:', err);
    startPolling(runId);
  }
}

function startPolling(runId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const run = await fetchJSON(`/api/runs/${runId}`);

      if (run.status === 'complete') {
        clearInterval(pollTimer);
        setSmoke(hasAgreement(run));
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
  const artifacts = parseArtifactPaths();
  const outputType = outputTypeEl?.value || '';
  const useCase = useCaseEl?.value || '';
  const agentSet = agentSetEl?.value || '';
  const overrides = roleOverrides || {};

  if (!query) {
    setPromptStatus('Enter a question first');
    return;
  }

  const payload = { title, query, notes, artifacts };
  if (outputType) payload.output_type = outputType;
  if (useCase) payload.use_case = useCase;
  if (agentSet) payload.agent_set = agentSet;
  if (Object.keys(overrides).length) payload.role_overrides = overrides;
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
  syncArtifactPaths(prompt.artifacts || []);
  if (outputTypeEl) outputTypeEl.value = prompt.output_type || '';
  if (useCaseEl) useCaseEl.value = prompt.use_case || '';
  if (agentSetEl) agentSetEl.value = prompt.agent_set || '';
  setRoleOverrides(prompt.role_overrides || {});
  if (fileInputEl) fileInputEl.value = '';
  currentPromptId = prompt.id;
  setPromptStatus('Loaded');
  updatePromptButton();
}

async function runPrompt(promptId) {
  const queryPreview = document.getElementById('query').value.trim();
  const outputType = outputTypeEl?.value || '';
  const inputTitle = document.getElementById('input-title').value.trim();
  primeRunningView(buildPendingRun(queryPreview || `Saved prompt ${promptId}`, outputType, inputTitle));
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
  if (useCaseEl) useCaseEl.value = run.meta?.use_case || '';
  if (agentSetEl) agentSetEl.value = run.meta?.agent_set || '';
  setRoleOverrides(run.meta?.role_overrides || {});
  if (fileInputEl) fileInputEl.value = '';
  if (run.meta?.input_path) {
    try {
      const input = await fetchJSON(`/api/inputs/${encodeURIComponent(run.meta.input_path.split('/').pop())}`);
      const matches = (input.content || '').match(/## Artifacts([\s\S]*)/i);
      if (matches && matches[1]) {
        const paths = matches[1].split('\n').map((line) => line.replace(/^-\\s*/, '').trim()).filter(Boolean);
        syncArtifactPaths(paths);
      }
    } catch (err) {
      // ignore missing input file
    }
  } else {
    syncArtifactPaths([]);
  }
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

if (artifactFieldEl) {
  artifactFieldEl.addEventListener('input', () => {
    renderArtifactList(parseArtifactPaths());
  });
}

if (artifactListEl) {
  artifactListEl.addEventListener('click', (e) => {
    if (!e.target.classList.contains('artifact-remove')) return;
    const path = e.target.dataset?.path;
    if (!path) return;
    const remaining = parseArtifactPaths().filter((item) => item !== path);
    syncArtifactPaths(remaining);
  });
}

if (fileInputEl) {
  fileInputEl.addEventListener('change', async () => {
    const files = Array.from(fileInputEl.files || []);
    if (!files.length) return;
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    try {
      const resp = await fetch('/api/uploads', {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
      const data = await resp.json();
      const uploaded = (data.files || []).map((item) => item.path).filter(Boolean);
      const merged = parseArtifactPaths().concat(uploaded);
      syncArtifactPaths(merged);
    } catch (err) {
      console.error(err);
    } finally {
      fileInputEl.value = '';
    }
  });
}

rerunBtn.addEventListener('click', async () => {
  try {
    const latest = await fetchJSON('/api/runs/latest');
    if (!latest?.query) {
      setStatus('No run to rerun');
      return;
    }
    const latestLabel = latest.meta?.input_title || latest.query || latest.id;
    const latestOutput = latest.meta?.output_type;
    primeRunningView(buildPendingRun(latest.query || latestLabel, latestOutput || '', latest.meta?.input_title || ''));
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
        input_title: latest.meta?.input_title,
        role_overrides: latest.meta?.role_overrides || undefined,
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
  syncArtifactPaths([]);
  if (outputTypeEl) outputTypeEl.value = '';
  if (useCaseEl) useCaseEl.value = '';
  if (agentSetEl) agentSetEl.value = '';
  setRoleOverrides({});
  if (fileInputEl) fileInputEl.value = '';
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

if (viewLatestContextBtn) {
  viewLatestContextBtn.addEventListener('click', async () => {
    try {
      const latest = latestRunCache || await fetchJSON('/api/runs/latest');
      if (latest) openContextModal(latest);
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

if (contextModalEl) {
  contextModalEl.addEventListener('click', (e) => {
    if (e.target === contextModalEl) closeContextModal();
  });
}

if (latestEl) {
  latestEl.addEventListener('click', (e) => {
    const button = e.target.closest('.panel-note-toggle');
    if (!button) return;
    const noteId = button.dataset?.noteId;
    if (!noteId) return;
    const panel = document.getElementById(noteId);
    if (!panel) return;
    const isHidden = panel.classList.contains('hidden');
    panel.classList.toggle('hidden', !isHidden);
    button.textContent = isHidden ? 'Hide notes' : 'Show notes';
  });
}

if (closeContextBtn) {
  closeContextBtn.addEventListener('click', () => closeContextModal());
}

if (clearSelectionBtn) {
  clearSelectionBtn.addEventListener('click', () => clearSelectedRun());
}

if (copyContextBtn) {
  copyContextBtn.addEventListener('click', async () => {
    if (!activeContextRun) return;
    const text = buildContextText(activeContextRun);
    await navigator.clipboard.writeText(text);
    setRunActionEphemeral('Copied context', 'Full context copied to clipboard.', 'success', 3000);
  });
}

if (copyContextCompactBtn) {
  copyContextCompactBtn.addEventListener('click', async () => {
    if (!activeContextRun) return;
    const text = buildContextCompactText(activeContextRun);
    await navigator.clipboard.writeText(text);
    setRunActionEphemeral('Copied summary', 'Compact context copied to clipboard.', 'success', 3000);
  });
}

if (latestArtifactsEl) {
  latestArtifactsEl.addEventListener('click', async (e) => {
    if (!e.target.classList.contains('copy-image-prompt')) return;
    if (!latestImagePrompt) return;
    await navigator.clipboard.writeText(latestImagePrompt);
    setRunActionEphemeral('Copied image prompt', 'Image prompt copied to clipboard.', 'success', 3000);
  });
}

if (promptExamplesEl) {
  promptExamplesEl.addEventListener('click', (e) => {
    const example = e.target.dataset?.example;
    if (example) {
      document.getElementById('query').value = example;
      schedulePlanRefresh();
    }
  });
}

if (useCaseEl) {
  useCaseEl.addEventListener('change', () => {
    const selected = useCaseEl.value;
    const entry = (useCasesCache || []).find((item) => item.id === selected);
    if (!entry) return;
    if (outputTypeEl && (!outputTypeEl.value || outputTypeEl.value === '')) {
      if (entry.output_type) outputTypeEl.value = entry.output_type;
    }
    if (agentSetEl && (!agentSetEl.value || agentSetEl.value === '')) {
      if (entry.agent_set) agentSetEl.value = entry.agent_set;
    }
    schedulePlanRefresh();
  });
}

const queryFieldEl = document.getElementById('query');
if (queryFieldEl) {
  queryFieldEl.addEventListener('input', () => schedulePlanRefresh());
}

if (outputTypeEl) {
  outputTypeEl.addEventListener('change', () => schedulePlanRefresh());
}

if (agentSetEl) {
  agentSetEl.addEventListener('change', () => schedulePlanRefresh());
}

if (collectionPickerEl) {
  collectionPickerEl.addEventListener('change', () => schedulePlanRefresh());
}

if (refreshPlanBtn) {
  refreshPlanBtn.addEventListener('click', () => refreshPlan().catch(console.error));
}

if (clearOverridesBtn) {
  clearOverridesBtn.addEventListener('click', () => {
    setRoleOverrides({});
    schedulePlanRefresh();
  });
}

if (roleOverridesEl) {
  roleOverridesEl.addEventListener('change', (e) => {
    if (isRenderingPlan) return;
    const select = e.target.closest('select[data-role]');
    if (!select) return;
    const role = select.dataset.role;
    if (!role) return;
    const value = select.value;
    if (value) {
      roleOverrides[role] = value;
    } else {
      delete roleOverrides[role];
    }
    schedulePlanRefresh();
  });
}

if (promptListEl) {
  promptListEl.addEventListener('click', async (e) => {
    const action = e.target.dataset?.action;
    const promptId = e.target.dataset?.id;
    const runId = e.target.dataset?.runId;
    const card = e.target.closest('.prompt-card');
    const cardPromptId = card?.dataset?.promptId;
    const cardRunId = card?.dataset?.runId;

    if (!action) {
      if (card && !e.target.closest('button')) {
        if (cardPromptId) await loadPrompt(cardPromptId);
        if (cardRunId) {
          const run = await fetchJSON(`/api/runs/${cardRunId}`);
          setSelectedRun(run);
        }
      }
      return;
    }

    if (action === 'view') {
      if (!runId) return;
      const run = await fetchJSON(`/api/runs/${runId}`);
      setSelectedRun(run);
      await loadRunForEdit(run);
      return;
    }

    if (!promptId) return;

    if (action === 'load') {
      await loadPrompt(promptId);
      if (cardRunId) {
        const run = await fetchJSON(`/api/runs/${cardRunId}`);
        setSelectedRun(run);
      }
    }
    if (action === 'run') await runPrompt(promptId);
    if (action === 'delete') {
      await fetchJSON(`/api/prompts/${promptId}`, { method: 'DELETE' });
      if (currentPromptId === promptId) {
        currentPromptId = null;
        setPromptStatus('Ready');
        updatePromptButton();
      }
      refresh();
    }

    if (cardRunId && !e.target.closest('button')) {
      const run = await fetchJSON(`/api/runs/${cardRunId}`);
      setSelectedRun(run);
      await loadRunForEdit(run);
    }
  });
}

if (historyTabButtons.length) {
  historyTabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab || 'saved';
      setHistoryTab(tab);
    });
  });
  let initialTab = 'saved';
  try {
    const stored = localStorage.getItem('conclave.historyTab');
    if (stored) initialTab = stored;
  } catch (err) {
    // ignore storage errors
  }
  setHistoryTab(initialTab || 'saved');
}

if (runListEl) {
  runListEl.addEventListener('click', async (e) => {
    const runId = e.target.dataset?.runId;
    const deleteId = e.target.dataset?.runId;
    const card = e.target.closest('.run-card');
    const cardRunId = card?.dataset?.runId;

    if (e.target.classList.contains('view-context')) {
      if (!runId) return;
      const run = await fetchJSON(`/api/runs/${runId}`);
      openContextModal(run);
      return;
    }

    if (e.target.classList.contains('edit-run')) {
      if (!runId) return;
      const run = await fetchJSON(`/api/runs/${runId}`);
      loadRunForEdit(run);
      setSelectedRun(run);
    }

    if (e.target.classList.contains('rerun-run')) {
      if (!runId) return;
      const card = e.target.closest('.run-card');
      const label = card?.dataset?.runTitle || card?.dataset?.runQuery || runId;
      const outputType = card?.dataset?.runOutput || '';
      primeRunningView(buildPendingRun(label, outputType, ''));
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
      if (!runId) return;
      await fetchJSON(`/api/runs/${runId}`, { method: 'DELETE' });
      refresh();
    }

    if (e.target.classList.contains('card-delete')) {
      if (!deleteId) return;
      await fetchJSON(`/api/runs/${deleteId}`, { method: 'DELETE' });
      refresh();
      return;
    }

    if (cardRunId && !e.target.closest('button')) {
      const run = await fetchJSON(`/api/runs/${cardRunId}`);
      setSelectedRun(run);
      await loadRunForEdit(run);
    }
  });
}

if (reconcilePanelEl) {
  reconcilePanelEl.addEventListener('click', async (e) => {
    const button = e.target.closest('button');
    if (!button) return;
    const runId = button.dataset?.runId;
    if (!runId) return;
    try {
      const run = await fetchJSON(`/api/runs/${runId}`);
      setSelectedRun(run);
    } catch (err) {
      console.error(err);
    }
  });
}

// Initialize
refresh();
updatePromptButton();
renderArtifactList(parseArtifactPaths());
