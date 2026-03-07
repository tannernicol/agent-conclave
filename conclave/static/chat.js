// Conclave Chat UI

const AGENTS = {
  claude: { label: 'claude', icon: '✦', color: '#e8845c' },
  codex:  { label: 'codex',  icon: '✧', color: '#4ade80' },
  gemini: { label: 'gemini', icon: '◆', color: '#60a5fa' },
};

const VALID_SENDERS = new Set(['user', ...Object.keys(AGENTS)]);
const USER_ICON = '👤';
const MAX_CLIENT_MESSAGES = 500;

// ── State ──
let ws = null;
let messages = [];
let messageIds = new Set();
let agentStatus = {};
let reconnectTimer = null;
let reconnectDelay = 1000;
let messageIdCounter = 0;
let roomId = 'default';

// ── DOM refs ──
const messagesEl = document.getElementById('messages');
const emptyState = document.getElementById('empty-state');
const inputEl = document.getElementById('message-input');
const sendBtn = document.getElementById('btn-send');
const mentionBar = document.getElementById('mention-bar');
const statusBar = document.getElementById('agent-status');

// ── Initialize ──
function init() {
  buildMentionBar();
  buildStatusBar();
  bindEvents();
  connect();
  inputEl.focus();
}

// ── WebSocket ──
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/chat/${roomId}`);

  ws.onopen = () => {
    reconnectDelay = 1000;
    console.log('[chat] connected');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWsMessage(data);
    } catch (e) {
      console.error('[chat] parse error', e);
    }
  };

  ws.onclose = () => {
    console.log('[chat] disconnected, reconnecting...');
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => {
      reconnectDelay = Math.min(reconnectDelay * 1.5, 10000);
      connect();
    }, reconnectDelay);
  };

  ws.onerror = () => {};
}

function wsSend(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}

function handleWsMessage(data) {
  switch (data.type) {
    case 'message':
      addMessage(data);
      break;
    case 'typing':
      setAgentStatus(data.agent, 'typing');
      break;
    case 'typing_stop':
      setAgentStatus(data.agent, 'online');
      break;
    case 'status':
      if (data.agents) {
        for (const [agent, status] of Object.entries(data.agents)) {
          setAgentStatus(agent, status);
        }
      }
      break;
    case 'history':
      if (Array.isArray(data.messages)) {
        for (const msg of data.messages) {
          if (msg.sender === 'system') {
            addSystemMessage(msg.content || '');
          } else {
            addMessage(msg, false);
          }
        }
        scrollToBottom();
      }
      break;
    case 'system':
      addSystemMessage(data.content || data.message || '');
      break;
    case 'error':
      addSystemMessage(data.message || 'An error occurred');
      break;
  }
}

// ── Status bar ──
function buildStatusBar() {
  statusBar.innerHTML = '';
  for (const [id, agent] of Object.entries(AGENTS)) {
    const el = document.createElement('div');
    el.className = 'agent-indicator';
    el.id = `status-${id}`;
    el.innerHTML = `<span class="dot"></span>${agent.label[0].toUpperCase() + agent.label.slice(1)}`;
    statusBar.appendChild(el);
    agentStatus[id] = 'offline';
  }
}

function setAgentStatus(agent, status) {
  if (!VALID_SENDERS.has(agent)) return;
  agentStatus[agent] = status;
  const el = document.getElementById(`status-${agent}`);
  if (!el) return;
  el.className = `agent-indicator ${status}`;

  if (status !== 'typing') {
    const typingEl = document.getElementById(`typing-${agent}`);
    if (typingEl) typingEl.remove();
  }

  if (status === 'typing') {
    showTypingIndicator(agent);
  }
}

// ── Mention bar ──
function buildMentionBar() {
  mentionBar.innerHTML = '';
  for (const [id, agent] of Object.entries(AGENTS)) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `mention-btn ${id}`;
    btn.textContent = `@${agent.label[0].toUpperCase() + agent.label.slice(1)}`;
    btn.addEventListener('click', () => insertMention(id));
    mentionBar.appendChild(btn);
  }
}

function insertMention(agent) {
  const name = AGENTS[agent]?.label;
  if (!name) return;
  const mention = `@${name}`;
  const val = inputEl.value;
  const pos = inputEl.selectionStart || val.length;
  const before = val.slice(0, pos);
  const after = val.slice(pos);
  const needSpace = before.length > 0 && !before.endsWith(' ') && !before.endsWith('\n');
  inputEl.value = before + (needSpace ? ' ' : '') + mention + ' ' + after;
  inputEl.focus();
  const newPos = pos + mention.length + (needSpace ? 2 : 1);
  inputEl.setSelectionRange(newPos, newPos);
  autoResize();
}

// ── Input handling ──
function bindEvents() {
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputEl.addEventListener('input', autoResize);

  sendBtn.addEventListener('click', sendMessage);

  inputEl.addEventListener('paste', (e) => {
    const files = e.clipboardData?.files;
    if (files && files.length > 0) {
      e.preventDefault();
      addSystemMessage('Image upload coming soon');
    }
  });
}

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

function sendMessage() {
  const content = inputEl.value.trim();
  if (!content) return;

  const mentionPattern = /@(claude|codex|gemini)/gi;
  const mentions = [...content.matchAll(mentionPattern)].map(m => m[1].toLowerCase());
  const uniqueMentions = [...new Set(mentions)];

  // Add user message locally (will be deduped when server broadcasts it back)
  const localId = `local-${++messageIdCounter}`;
  const msg = {
    id: localId,
    sender: 'user',
    content: content,
    timestamp: new Date().toISOString(),
    mentions: uniqueMentions,
  };

  addMessage(msg);

  wsSend({
    type: 'message',
    content: content,
    mentions: uniqueMentions.length > 0 ? uniqueMentions : Object.keys(AGENTS),
  });

  inputEl.value = '';
  autoResize();
  inputEl.focus();
}

// ── Message rendering ──
function addMessage(msg, scroll = true) {
  // Deduplicate by server ID
  if (msg.id && messageIds.has(msg.id)) return;
  if (msg.id) messageIds.add(msg.id);
  messages.push(msg);
  // Trim client-side message history
  if (messages.length > MAX_CLIENT_MESSAGES) {
    const excess = messages.length - MAX_CLIENT_MESSAGES;
    messages = messages.slice(excess);
    // Remove excess DOM nodes
    for (let i = 0; i < excess; i++) {
      const first = messagesEl.querySelector('.msg, .msg-system');
      if (first) first.remove();
    }
  }
  if (emptyState) emptyState.style.display = 'none';

  const el = renderMessage(msg);
  messagesEl.appendChild(el);

  if (scroll) scrollToBottom();
}

function addSystemMessage(text) {
  const el = document.createElement('div');
  el.className = 'msg-system';
  el.innerHTML = `<div class="system-pill">${renderContent(text)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function showTypingIndicator(agent) {
  if (document.getElementById(`typing-${agent}`)) return;
  if (!AGENTS[agent]) return;

  const agentInfo = AGENTS[agent];
  const el = document.createElement('div');
  el.className = `msg ${agent} msg-typing`;
  el.id = `typing-${agent}`;
  el.innerHTML = `
    <div class="msg-avatar">${agentInfo.icon}</div>
    <div class="msg-body">
      <div class="msg-header">
        <span class="msg-sender">${agentInfo.label}</span>
      </div>
      <div class="msg-content">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function safeSenderClass(sender) {
  return VALID_SENDERS.has(sender) ? sender : 'unknown';
}

function renderMessage(msg) {
  const sender = msg.sender || 'user';
  const safeClass = safeSenderClass(sender);
  const isAgent = sender in AGENTS;
  const isUser = sender === 'user';

  const el = document.createElement('div');
  el.className = `msg ${safeClass}`;
  el.dataset.id = msg.id || '';

  let icon = USER_ICON;
  if (isAgent) icon = AGENTS[sender].icon;

  const time = formatTime(msg.timestamp);

  // Quote block
  let quoteHtml = '';
  if (msg.quoting) {
    const qSender = msg.quoting.sender || 'user';
    const qClass = safeSenderClass(qSender);
    const qLabel = qSender === 'user' ? 'User' : (AGENTS[qSender]?.label || qSender);
    const qText = truncate(stripMarkdown(msg.quoting.content || ''), 120);
    quoteHtml = `
      <div class="msg-quote ${qClass}">
        <span class="quote-sender">${escapeHtml(qLabel)}</span>
        ${escapeHtml(qText)}
      </div>
    `;
  }

  const renderedContent = renderContent(msg.content || '');

  // Remove typing indicator for this agent
  if (isAgent) {
    const typingEl = document.getElementById(`typing-${sender}`);
    if (typingEl) typingEl.remove();
  }

  const senderLabel = isUser ? 'User' : (AGENTS[sender]?.label || sender);

  // Role badge (reasoner, critic, panel)
  let roleBadge = '';
  if (msg.role) {
    const roleLabels = { reasoner: 'Reasoner', critic: 'Critic', panel: 'Panel' };
    const roleLabel = roleLabels[msg.role] || msg.role;
    roleBadge = `<span class="role-badge role-${escapeHtml(msg.role)}">${escapeHtml(roleLabel)}</span>`;
  }

  el.innerHTML = `
    <div class="msg-avatar">${icon}</div>
    <div class="msg-body">
      ${quoteHtml}
      <div class="msg-header">
        <span class="msg-sender">${escapeHtml(senderLabel)}</span>
        ${roleBadge}
        <span class="msg-time">${time}</span>
      </div>
      <div class="msg-content">${renderedContent}</div>
    </div>
  `;

  return el;
}

// ── Content rendering (simple markdown) ──
function renderContent(text) {
  const parts = text.split(/(```[\s\S]*?```)/g);
  let html = '';

  for (const part of parts) {
    if (part.startsWith('```') && part.endsWith('```')) {
      const inner = part.slice(3, -3);
      const newlineIdx = inner.indexOf('\n');
      let lang = '';
      let code = inner;
      if (newlineIdx > -1 && newlineIdx < 20 && !/\s/.test(inner.slice(0, newlineIdx))) {
        lang = inner.slice(0, newlineIdx).trim();
        code = inner.slice(newlineIdx + 1);
      }
      html += `<pre${lang ? ` data-lang="${escapeHtml(lang)}"` : ''}><code>${escapeHtml(code)}</code></pre>`;
    } else {
      html += renderInline(part);
    }
  }

  return html;
}

function renderInline(text) {
  // Process raw text — extract code spans first to protect them from escaping
  const codeSpans = [];
  let processed = text.replace(/`([^`]+)`/g, (_match, code) => {
    const idx = codeSpans.length;
    codeSpans.push(`<code>${escapeHtml(code)}</code>`);
    return `\x00CODE${idx}\x00`;
  });

  // Now escape the rest
  processed = escapeHtml(processed);

  // @mentions
  processed = processed.replace(/@(claude|codex|gemini)/gi, (_match, name) => {
    const lower = name.toLowerCase();
    return `<span class="mention ${lower}">@${name}</span>`;
  });

  processed = processed.replace(/@(User|user)/g, '<span class="mention user">@User</span>');

  // Bold
  processed = processed.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic
  processed = processed.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Restore code spans
  processed = processed.replace(/\x00CODE(\d+)\x00/g, (_match, idx) => codeSpans[parseInt(idx, 10)] || '');

  // Convert bullet lists
  const lines = processed.split('\n');
  let inList = false;
  let result = '';

  for (const line of lines) {
    const trimmed = line.trim();
    const isBullet = /^[-•]\s/.test(trimmed);

    if (isBullet) {
      if (!inList) { result += '<ul>'; inList = true; }
      result += `<li>${trimmed.replace(/^[-•]\s/, '')}</li>`;
    } else {
      if (inList) { result += '</ul>'; inList = false; }
      if (trimmed !== '') {
        result += `<p>${line}</p>`;
      }
    }
  }

  if (inList) result += '</ul>';

  return result;
}

// ── Utilities ──
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '';
  }
}

function truncate(str, max) {
  if (str.length <= max) return str;
  return str.slice(0, max) + '...';
}

function stripMarkdown(text) {
  return text
    .replace(/```[\s\S]*?```/g, '[code]')
    .replace(/`[^`]+`/g, '[code]')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/@(claude|codex|gemini)/gi, '@$1');
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

// ── Boot ──
document.addEventListener('DOMContentLoaded', init);
