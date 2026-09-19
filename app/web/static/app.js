// 前端逻辑：会话管理 + 知识库列表 + 基于 SSE 流的问答（含检索资料卡片展示）
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const docListEl = document.getElementById('docList');
const docCountEl = document.getElementById('docCount');
const sessionListEl = document.getElementById('sessionList');
const newSessionBtn = document.getElementById('newSessionBtn');

// 当前会话 id（存 localStorage，刷新后仍能回到同一会话）
let currentSessionId = localStorage.getItem('session_id');

// ---------- 工具：生成 UUID ----------
function genId() {
  return crypto.randomUUID ? crypto.randomUUID()
    : 'sess-' + Date.now() + '-' + Math.random().toString(36).slice(2);
}

// ---------- Markdown 渲染（本地 vendor 库，DOMPurify 消毒防 XSS） ----------
function renderMarkdown(text) {
  if (typeof marked === 'undefined') return null;
  return DOMPurify.sanitize(marked.parse(text || '', { breaks: true, gfm: true }));
}

// 回答专用渲染：
//   withCitations=true（当前问答）：[资料N] → 可点击角标，联动上方资料卡片；
//                                   编号越界（超过实际卡片数）渲染为灰色不可点。
//   withCitations=false（历史会话）：直接移除 [资料N] 标记及末尾"引用：…"整段
//                                   （历史消息未持久化检索资料，角标无卡片可联动）。
function renderAnswerHTML(text, sourcesCount, withCitations) {
  let prepared = text || '';
  if (withCitations) {
    prepared = prepared.replace(/\[资料\s*(\d+)\]/g, (m, n) => {
      const invalid = sourcesCount && Number(n) > sourcesCount;
      return `<sup class="cite${invalid ? ' cite-invalid' : ''}" data-n="${n}">${n}</sup>`;
    });
  } else {
    prepared = prepared
      .replace(/引用[:：]?\s*(?:\[资料\s*\d+\]\s*[,，、]?\s*)+/g, '')
      .replace(/\[资料\s*\d+\]\s*[,，、]?/g, '')
      .replace(/[,，、\s]+$/, '');
  }
  return renderMarkdown(prepared);
}

// ---------- 消息渲染 ----------
// bot/assistant 消息 = 头像 + 全宽正文（Markdown 渲染）；user 消息 = 右侧气泡（纯文本）
function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role === 'user' ? 'user' : 'bot'}`;

  if (role === 'user') {
    div.textContent = text;
  } else {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '✦';
    const body = document.createElement('div');
    body.className = 'msg-body';
    const answer = document.createElement('div');
    answer.className = 'answer-text';
    // 历史消息：无检索资料面板，去掉 [资料N] 引用标记
    const html = renderAnswerHTML(text, 0, false);
    if (html !== null) answer.innerHTML = html;
    else answer.textContent = text;
    body.appendChild(answer);
    div.append(avatar, body);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

// "思考中"占位（发送后立即显示，第一个 token 到达时移除）
function makeThinking() {
  const el = document.createElement('div');
  el.className = 'thinking';
  el.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span> 思考中';
  return el;
}

// 助手消息骨架：头像 + 空正文（内含思考中占位），资料面板和回答后续填充
function buildBotMessage() {
  const div = document.createElement('div');
  div.className = 'msg bot';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '✦';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const answerEl = document.createElement('div');
  answerEl.className = 'answer-text';

  const thinking = makeThinking();
  body.append(answerEl, thinking);

  div.append(avatar, body);

  // 角标点击事件委托（流式过程中 answerEl.innerHTML 反复重绘，委托不受影响）：
  // 展开资料面板 → 平滑滚动到对应卡片 → 高亮闪烁
  div.addEventListener('click', (e) => {
    const sup = e.target.closest('.cite');
    if (!sup || sup.classList.contains('cite-invalid')) return;
    const n = sup.dataset.n;
    const card = div.querySelector(`.source-card[data-cite="${n}"]`);
    if (!card) return;
    const block = div.querySelector('.sources-block');
    if (block) block.classList.remove('collapsed');
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.remove('cite-flash');
    void card.offsetWidth; // 强制 reflow，让动画可以重新触发
    card.classList.add('cite-flash');
    setTimeout(() => card.classList.remove('cite-flash'), 2000);
  });

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { div, body, answerEl, thinking };
}

// 把资料面板插入到回答正文上方
function showSources(body, sources, answerEl) {
  const block = renderSourcesBlock(sources);
  if (block) body.insertBefore(block, answerEl);
  return block;
}

// 渲染检索资料面板（header 可整体折叠，卡片内正文默认 2 行折叠）
function renderSourcesBlock(sources) {
  if (!sources || !sources.length) return null;

  const block = document.createElement('div');
  block.className = 'sources-block';

  const header = document.createElement('div');
  header.className = 'sources-header';
  header.innerHTML = `<span>参考资料（${sources.length}）</span><span class="chevron">▼</span>`;
  header.onclick = () => block.classList.toggle('collapsed');
  block.appendChild(header);

  const list = document.createElement('div');
  list.className = 'sources-list';
  block.appendChild(list);

  sources.forEach((s, idx) => {
    const card = document.createElement('div');
    card.className = 'source-card';
    // 编号与 prompt 中【资料N】一致（从 1 开始），供角标定位
    card.dataset.cite = String(idx + 1);

    // 卡片头部：来源名 + 类型标签 + 相似度
    const head = document.createElement('div');
    head.className = 'source-card-head';

    const nameWrap = document.createElement('div');
    nameWrap.className = 'source-name-wrap';

    const tag = document.createElement('span');
    tag.className = `source-tag ${s.collection === 'faq' ? 'tag-faq' : 'tag-doc'}`;
    tag.textContent = s.collection === 'faq' ? 'FAQ' : '文档';
    nameWrap.appendChild(tag);

    // URL 来源渲染为可点击链接，文件来源为普通文本
    let name;
    if (s.source && /^https?:\/\//.test(s.source)) {
      name = document.createElement('a');
      name.href = s.source;
      name.target = '_blank';
      name.rel = 'noopener';
      name.classList.add('source-name', 'link');
    } else {
      name = document.createElement('span');
      name.className = 'source-name';
    }
    name.textContent = s.source || '(未知来源)';
    name.title = s.source;
    nameWrap.appendChild(name);

    const score = document.createElement('span');
    score.className = 'source-score';
    const pct = Math.round((s.score || 0) * 100);
    score.textContent = `${pct}%`;
    score.title = `相似度 ${pct}%`;
    if (pct >= 80) score.classList.add('score-high');
    else if (pct >= 60) score.classList.add('score-mid');
    else score.classList.add('score-low');

    head.append(nameWrap, score);
    card.appendChild(head);

    // 相似度进度条
    const bar = document.createElement('div');
    bar.className = 'score-bar';
    const fill = document.createElement('div');
    fill.className = 'score-bar-fill';
    fill.style.width = `${pct}%`;
    if (pct >= 80) fill.classList.add('score-high');
    else if (pct >= 60) fill.classList.add('score-mid');
    else fill.classList.add('score-low');
    bar.appendChild(fill);
    card.appendChild(bar);

    // 片段正文（默认折叠前 2 行）
    const text = document.createElement('div');
    text.className = 'source-text collapsed';
    text.textContent = s.text || '';
    card.appendChild(text);

    // 展开/收起按钮
    const toggle = document.createElement('button');
    toggle.className = 'source-toggle';
    toggle.textContent = '展开全文';
    toggle.onclick = () => {
      const collapsed = text.classList.toggle('collapsed');
      toggle.textContent = collapsed ? '展开全文' : '收起';
    };
    card.appendChild(toggle);

    list.appendChild(card);
  });

  return block;
}

// ---------- 会话管理 ----------
async function loadSessions() {
  try {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    renderSessionList(data.sessions || []);
  } catch (e) {
    console.error('加载会话列表失败', e);
  }
}

function renderSessionList(sessions) {
  sessionListEl.innerHTML = '';
  if (!sessions.length) {
    const li = document.createElement('li');
    li.className = 'session-empty';
    li.textContent = '暂无会话';
    sessionListEl.appendChild(li);
    return;
  }
  sessions.forEach(s => {
    const li = document.createElement('li');
    li.className = 'session-item' + (s.session_id === currentSessionId ? ' active' : '');
    li.dataset.id = s.session_id;

    const info = document.createElement('div');
    info.className = 'session-info';
    const title = document.createElement('span');
    title.className = 'session-title';
    title.textContent = s.title;
    title.title = s.title;
    const meta = document.createElement('span');
    meta.className = 'session-meta';
    meta.textContent = s.last_active_at;
    info.append(title, meta);

    const delBtn = document.createElement('button');
    delBtn.className = 'session-delete';
    delBtn.textContent = '✕';
    delBtn.title = '删除会话';
    delBtn.onclick = (e) => {
      e.stopPropagation();
      deleteSession(s.session_id, s.title);
    };

    li.append(info, delBtn);
    li.onclick = () => switchSession(s.session_id);
    sessionListEl.appendChild(li);
  });
}

async function createSession() {
  const res = await fetch('/api/sessions', { method: 'POST' });
  const data = await res.json();
  currentSessionId = data.session_id;
  localStorage.setItem('session_id', currentSessionId);
  inputEl.value = '';
  renderChatWelcome();
  await loadSessions();
}

async function deleteSession(id, title) {
  if (!confirm(`确定删除会话"${title}"吗？`)) return;
  await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
  // 若删的是当前会话，切到列表第一个或新建
  if (id === currentSessionId) {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    if (data.sessions && data.sessions.length) {
      currentSessionId = data.sessions[0].session_id;
      localStorage.setItem('session_id', currentSessionId);
      await switchSession(currentSessionId);
    } else {
      await createSession();
    }
  } else {
    await loadSessions();
  }
}

async function switchSession(id) {
  if (id === currentSessionId) return;
  currentSessionId = id;
  localStorage.setItem('session_id', currentSessionId);
  inputEl.value = '';
  await loadSessionMessages(id);
  await loadSessions();
}

async function loadSessionMessages(id) {
  const res = await fetch(`/api/sessions/${id}/messages`);
  const data = await res.json();
  messagesEl.innerHTML = '';
  const msgs = data.messages || [];
  if (!msgs.length) {
    renderChatWelcome();
    return;
  }
  // 后端返回正序（旧→新），直接逐条 append 即可
  msgs.forEach(m => addMessage(m.role, m.content));
}

function renderChatWelcome() {
  messagesEl.innerHTML = '';
  addMessage('bot', '你好！知识文件已在服务启动时自动入库，直接基于你的资料提问即可。');
}

// ---------- 知识库列表 ----------
async function loadDocs() {
  const res = await fetch('/api/documents');
  const data = await res.json();
  docCountEl.textContent = data.documents.length;
  docListEl.innerHTML = '';
  data.documents.forEach(doc => {
    const li = document.createElement('li');

    const name = document.createElement('span');
    name.className = 'doc-name';
    name.textContent = doc.source;
    name.title = `${doc.source} · ${doc.chunk_count} 个片段 · 入库于 ${doc.indexed_at}`;

    const meta = document.createElement('span');
    meta.className = 'doc-meta';
    meta.textContent = `${doc.chunk_count} 段`;

    li.append(name, meta);
    docListEl.appendChild(li);
  });
}

// ---------- 问答（SSE 流式） ----------
async function sendQuestion() {
  const question = inputEl.value.trim();
  if (!question) return;
  inputEl.value = '';
  sendBtn.disabled = true;

  addMessage('user', question);
  // 立即创建助手消息骨架（含"思考中"占位），检索期间页面不再空白
  const built = buildBotMessage();
  let { body, answerEl, thinking } = built;
  let fullText = '';   // 流式累积的回答原文
  let gotToken = false;
  let sourcesCount = 0; // 本轮实际检索到的资料数（判定角标是否越界）

  const removeThinking = () => { if (thinking) { thinking.remove(); thinking = null; } };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: currentSessionId }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        const evt = JSON.parse(line.slice(6));

        if (evt.type === 'sources') {
          // 检索完成：资料面板插到回答上方，占位继续显示到首个 token
          sourcesCount = (evt.documents || []).length;
          showSources(body, evt.documents, answerEl);
        } else if (evt.type === 'token') {
          if (!gotToken) { removeThinking(); gotToken = true; }
          fullText += evt.content;
          // 每个 token 后重渲染 Markdown + 引用角标（文本量小，性能无压力）
          answerEl.innerHTML = renderAnswerHTML(fullText, sourcesCount, true);
        } else if (evt.type === 'error') {
          removeThinking();
          const errEl = document.createElement('div');
          errEl.className = 'chat-error';
          errEl.textContent = '[错误] ' + evt.message;
          answerEl.appendChild(errEl);
        } else if (evt.type === 'done') {
          removeThinking();
          // 最终完整渲染一次，保证闭合状态正确
          if (fullText) answerEl.innerHTML = renderAnswerHTML(fullText, sourcesCount, true);
          // 回答结束，刷新会话列表（标题可能从"新会话"变为首条问题）
          await loadSessions();
        }
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }
  } catch (e) {
    removeThinking();
    const errEl = document.createElement('div');
    errEl.className = 'chat-error';
    errEl.textContent = '请求失败：' + e.message;
    answerEl.appendChild(errEl);
  } finally {
    sendBtn.disabled = false;
  }
}

// ---------- 事件绑定 ----------
sendBtn.onclick = sendQuestion;
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});
newSessionBtn.onclick = createSession;

// ---------- 初始化 ----------
async function init() {
  // 若无 session_id，先创建一个
  if (!currentSessionId) {
    const res = await fetch('/api/sessions', { method: 'POST' });
    const data = await res.json();
    currentSessionId = data.session_id;
    localStorage.setItem('session_id', currentSessionId);
  }
  renderChatWelcome();
  await Promise.all([loadSessions(), loadDocs()]);
}

init();
