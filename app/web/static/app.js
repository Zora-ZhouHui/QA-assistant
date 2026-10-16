// 前端逻辑：会话管理 + 基于 SSE 流的问答（检索资料卡片 + 右侧执行日志）
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const sessionListEl = document.getElementById('sessionList');
const newSessionBtn = document.getElementById('newSessionBtn');
const webSearchCheck = document.getElementById('webSearchCheck');
const imageBtn = document.getElementById('imageBtn');
const imageInput = document.getElementById('imageInput');
const imagePreviewEl = document.getElementById('imagePreview');
// 截图场景三选一弹窗
const sceneModalEl = document.getElementById('sceneModal');
const sceneOptionsEl = document.getElementById('sceneOptions');
const sceneNoteEl = document.getElementById('sceneNote');
const sceneConfirmBtn = document.getElementById('sceneConfirm');
// 设计稿对比缺图提示（需两张图：设计稿 + 页面截图）
const sceneImageHintEl = document.getElementById('sceneImageHint');
const sceneImageHintTextEl = document.getElementById('sceneImageHintText');
const sceneAddImageBtn = document.getElementById('sceneAddImageBtn');

// 场景 key → 显示名（与后端 app/agent/scenes.py 的 SCENE_LABELS 保持一致）
const SCENE_LABELS = {
  console_error: '控制台报错截图',
  layout_misalign: '布局错位截图',
  design_compare: '设计稿对比截图',
};

// 当前会话 id（存 localStorage，刷新后仍能回到同一会话）
let currentSessionId = localStorage.getItem('session_id');

// 待发送的图片（base64 dataURL 列表），发送成功后清空
let pendingImages = [];

// 弹窗中被选中的场景 key（未选为空）
let selectedScene = '';

// ---------- 工具：生成 UUID ----------
function genId() {
  return crypto.randomUUID ? crypto.randomUUID()
    : 'sess-' + Date.now() + '-' + Math.random().toString(36).slice(2);
}

// ---------- 图片：压缩 + 预览 + 增删 ----------
// 前端压缩：最长边缩到 maxDim，转 JPEG；截图文字密集，分辨率与质量取偏保真值
function compressImage(file, maxDim = 1400, quality = 0.85) {
  return new Promise((resolve, reject) => {
    if (!file || !file.type.startsWith('image/')) {
      reject(new Error('非图片文件'));
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL('image/jpeg', quality));
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('图片加载失败')); };
    img.src = url;
  });
}

async function addImageFiles(files) {
  const list = Array.from(files || []).filter(f => f.type && f.type.startsWith('image/'));
  for (const f of list) {
    try {
      pendingImages.push(await compressImage(f));
    } catch (e) {
      console.error('图片处理失败', e);
    }
  }
  renderImagePreview();
  // 有了待发图片 → 弹出场景选择（已有弹窗打开时不重置用户的选择）
  if (pendingImages.length && sceneModalEl.hidden) {
    openSceneModal();
  } else if (!sceneModalEl.hidden) {
    // 弹窗开着补图（如设计稿对比第二张）→ 刷新缺图提示与发送可用态
    refreshSceneConfirm();
  }
}

// ---------- 截图场景选择弹窗（上传图片后三选一） ----------
// 场景所需最少图片数：设计稿对比需两张（设计稿 + 页面截图），其余一张
function sceneMinImages(scene) {
  return scene === 'design_compare' ? 2 : 1;
}

// 刷新底部缺图提示与"开始解析"可用态：设计稿对比缺图时提示补齐并禁用发送
function refreshSceneConfirm() {
  const need = sceneMinImages(selectedScene);
  const missing = Math.max(0, need - pendingImages.length);
  const showHint = selectedScene === 'design_compare' && missing > 0;

  sceneConfirmBtn.disabled = !selectedScene || missing > 0;
  sceneImageHintEl.hidden = !showHint;
  if (showHint) {
    sceneImageHintTextEl.textContent =
      `设计稿对比需两张图：①设计稿 ②页面截图（当前 ${pendingImages.length}/${need}），请继续粘贴或点「添加图片」`;
  }
}

// 打开弹窗：清空选择，把输入框已有的文字带进"补充说明"（用户可能先打了字再传图）
function openSceneModal() {
  selectedScene = '';
  sceneNoteEl.value = inputEl.value.trim();
  sceneOptionsEl.querySelectorAll('.scene-card').forEach(c => c.classList.remove('selected'));
  sceneModalEl.hidden = false;
  refreshSceneConfirm();
}

function closeSceneModal() {
  sceneModalEl.hidden = true;
}

// 点卡片即选中（高亮），"开始解析"按钮随图片数是否满足而定
sceneOptionsEl.addEventListener('click', (e) => {
  const card = e.target.closest('.scene-card');
  if (!card) return;
  selectedScene = card.dataset.scene;
  sceneOptionsEl.querySelectorAll('.scene-card')
    .forEach(c => c.classList.toggle('selected', c === card));
  refreshSceneConfirm();
});

// 选定场景 → 关闭弹窗 → 带场景发起（补充说明作为 question 传给后端）
sceneConfirmBtn.onclick = () => {
  if (!selectedScene) return;
  // 设计稿对比图片不足时不发送，只刷新提示
  if (sceneMinImages(selectedScene) > pendingImages.length) {
    refreshSceneConfirm();
    return;
  }
  inputEl.value = sceneNoteEl.value.trim();
  closeSceneModal();
  sendQuestion(selectedScene);
};

// 弹窗内补图入口：复用隐藏文件选择（与输入区"图片"按钮一致）
sceneAddImageBtn.onclick = () => imageInput.click();

document.getElementById('sceneCancel').onclick = closeSceneModal;

function removePendingImage(i) {
  pendingImages.splice(i, 1);
  renderImagePreview();
}

function renderImagePreview() {
  imagePreviewEl.innerHTML = '';
  if (!pendingImages.length) {
    imagePreviewEl.hidden = true;
    return;
  }
  imagePreviewEl.hidden = false;
  pendingImages.forEach((dataUrl, i) => {
    const item = document.createElement('div');
    item.className = 'image-preview-item';
    const img = document.createElement('img');
    img.src = dataUrl;
    const del = document.createElement('button');
    del.className = 'image-preview-del';
    del.textContent = '✕';
    del.onclick = () => removePendingImage(i);
    item.append(img, del);
    imagePreviewEl.appendChild(item);
  });
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
// images 为可选附件（dataURL 列表），user 消息在文字下方渲染缩略图
function addMessage(role, text, images) {
  const div = document.createElement('div');
  div.className = `msg ${role === 'user' ? 'user' : 'bot'}`;

  if (role === 'user') {
    div.textContent = text;
    (images || []).forEach(dataUrl => {
      const img = document.createElement('img');
      img.className = 'msg-image';
      img.src = dataUrl;
      div.appendChild(img);
    });
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

// ---------- 右侧执行日志（Agent 决策时间线） ----------
// 每轮问答按时间顺序记录：提问 → 每次工具调用（类型/序号/检索词/结果数/耗时/结果明细）
// → 生成回答 → 总耗时。多次搜索时，不同检索词与各自结果数可解释"为什么会调用多次"。
const logListEl = document.getElementById('logList');
const LOG_EMPTY = '发送问题后，这里会实时显示助手的每一步检索与决策过程';
const logPanelEl = document.getElementById('logPanel');

// 收起/展开执行日志面板（状态记忆到 localStorage）：
// 新建会话/切历史会话 → 自动收起；提问产生日志 → 自动展开；回答中途手动收起不被抢占
function setLogCollapsed(collapsed) {
  logPanelEl.classList.toggle('collapsed', collapsed);
  try { localStorage.setItem('logCollapsed', collapsed ? '1' : '0'); } catch (e) { /* 隐私模式忽略 */ }
}
document.getElementById('logToggle').addEventListener('click', () => setLogCollapsed(true));
document.getElementById('logRail').addEventListener('click', () => setLogCollapsed(false));
if (localStorage.getItem('logCollapsed') === '1') logPanelEl.classList.add('collapsed');

let activeToolStep = null; // 当前进行中的工具步骤 {el, start}
let llmStep = null;        // 进行中的"模型调用"步骤 {el}（决策/生成共用，按后续事件定型）
let answerStep = null;     // 回答步骤 {el, start}
let visionStep = null;     // 进行中的"图片解析"步骤 {el, start}（仅截图问答）
let toolSeq = 0;           // 本轮工具调用序号（回答"第几次搜索"）
let llmCalls = 0;          // 本轮模型调用总次数（决策 + 生成）
let decisionCalls = 0;     // 其中"决策"次数（以调用工具告终的调用）
let qaStart = 0;           // 本轮问答起始时间
let liveStreaming = false; // 是否正在流式接收回答（轨迹回放需避让，避免与实时节点叠加）

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function nowTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

function resetLog() {
  logListEl.innerHTML = `<p class="log-empty">${LOG_EMPTY}</p>`;
  activeToolStep = null;
  answerStep = null;
  visionStep = null;
  llmStep = null;          // 清空进行中的"模型调用"步骤，避免跨轮残留
  llmCalls = 0;            // 计数器随新一轮归零，避免跨问题累计
  decisionCalls = 0;
  toolSeq = 0;
  qaStart = 0;
}

function appendStep(stateClass) {
  const empty = logListEl.querySelector('.log-empty');
  if (empty) empty.remove();
  const step = document.createElement('div');
  step.className = `log-step ${stateClass}`;
  step.innerHTML = '<span class="log-node"></span>';
  logListEl.appendChild(step);
  logListEl.scrollTop = logListEl.scrollHeight;
  return step;
}

function logQuestion(question) {
  resetLog();
  qaStart = performance.now();
  setLogCollapsed(false);  // 日志开始生成 → 自动展开面板
  const step = appendStep('info');
  step.innerHTML =
    '<span class="log-node"></span>' +
    '<div class="log-head"><span class="log-kind question">提问</span>' +
    '<span class="log-action">用户问题</span>' +
    `<span class="log-time">${nowTime()}</span></div>` +
    `<div class="log-detail">${escapeHtml(question)}</div>`;
}

// 截图问答的日志入口：新一轮问答先记"图片解析"（替代文字问答的"提问"节点）
function logSceneStart(label) {
  resetLog();
  qaStart = performance.now();
  setLogCollapsed(false);  // 日志开始生成 → 自动展开面板
  const step = appendStep('running');
  step.innerHTML =
    '<span class="log-node spin"></span>' +
    '<div class="log-head"><span class="log-kind model">多模态</span>' +
    `<span class="log-action">图片解析 · ${escapeHtml(label)}</span>` +
    `<span class="log-time">${nowTime()}</span></div>`;
  visionStep = { el: step, start: performance.now() };
}

// 图片解析完成：收起进行中标记，补上耗时
function logVisionDone() {
  if (!visionStep) return;
  const { el, start } = visionStep;
  const secs = ((performance.now() - start) / 1000).toFixed(1);
  el.classList.remove('running');
  el.classList.add('ok');
  el.querySelector('.log-node').classList.remove('spin');
  el.querySelector('.log-action').textContent = '图片解析完成';
  const meta = document.createElement('div');
  meta.className = 'log-meta';
  meta.textContent = `解析耗时 ${secs}s`;
  el.appendChild(meta);
  visionStep = null;
  logListEl.scrollTop = logListEl.scrollHeight;
}

// 模型调用开始：Agent 每轮循环先请求一次模型，要么决策调用工具，要么直接生成
function logLlmStart(callNo) {
  llmCalls = callNo;
  const step = appendStep('running');
  step.innerHTML =
    '<span class="log-node spin"></span>' +
    '<div class="log-head"><span class="log-kind model">模型</span>' +
    '<span class="log-action">第 ' + callNo + ' 次调用 · 推理中</span>' +
    `<span class="log-time">${nowTime()}</span></div>`;
  llmStep = { el: step };
}

// 该次模型调用以"请求调用工具"告终 → 定型为决策步骤
function logLlmDecision() {
  if (!llmStep) return;
  const { el } = llmStep;
  decisionCalls += 1;
  el.classList.remove('running');
  el.classList.add('info');
  el.querySelector('.log-node').classList.remove('spin');
  el.querySelector('.log-action').textContent = `第 ${llmCalls} 次调用 · 决策：调用工具`;
  llmStep = null;
}

function logToolStart(kind, query) {
  toolSeq += 1;
  const step = appendStep('running');
  const label = kind === 'kb' ? '本地检索' : '联网搜索';
  step.innerHTML =
    '<span class="log-node spin"></span>' +
    '<div class="log-head">' +
    `<span class="log-kind ${kind}">#${toolSeq} ${label}</span>` +
    `<span class="log-time">${nowTime()}</span></div>` +
    `<div class="log-query">检索词 <code>${escapeHtml(query || '')}</code></div>` +
    '<div class="log-meta">执行中…</div>';
  activeToolStep = { el: step, start: performance.now(), kind };
}

function _domain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return '';
  }
}

function _resultTitle(doc, kind) {
  if (kind === 'web') return doc.title || doc.source || '(无标题)';
  return doc.question || doc.source || '(无标题)';
}

function logToolFinish(docs, kind) {
  if (!activeToolStep) return;
  const { el, start } = activeToolStep;
  const secs = ((performance.now() - start) / 1000).toFixed(1);
  const list = docs || [];

  el.classList.remove('running');
  el.classList.add('ok');
  el.querySelector('.log-node').classList.remove('spin');
  el.querySelector('.log-meta').textContent = `返回 ${list.length} 条 · 耗时 ${secs}s`;

  if (list.length) {
    const ul = document.createElement('ul');
    ul.className = 'log-results';
    list.forEach(d => {
      const li = document.createElement('li');
      const idx = `<span class="log-result-idx">[资料${d.index}]</span>`;
      if (kind === 'web' && d.source) {
        li.innerHTML = idx;
        const a = document.createElement('a');
        a.href = d.source;
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = _resultTitle(d, kind);
        li.appendChild(a);
        const src = document.createElement('span');
        src.className = 'log-result-src';
        src.textContent = _domain(d.source);
        li.appendChild(src);
      } else {
        li.innerHTML = idx + escapeHtml(_resultTitle(d, kind));
      }
      ul.appendChild(li);
    });

    const toggle = document.createElement('button');
    toggle.className = 'log-results-toggle';
    toggle.textContent = '查看返回结果 ▸';
    toggle.onclick = () => {
      const expanded = el.classList.toggle('expanded');
      toggle.textContent = expanded ? '收起结果 ▾' : '查看返回结果 ▸';
    };
    el.append(toggle, ul);
  }
  activeToolStep = null;
  logListEl.scrollTop = logListEl.scrollHeight;
}

function logToolFail(message) {
  // 防御：没有进行中步骤时补建一个，保证失败一定有日志
  if (!activeToolStep) {
    logToolStart('web', '');
  }
  const { el, start } = activeToolStep;
  const secs = ((performance.now() - start) / 1000).toFixed(1);
  el.classList.remove('running');
  el.classList.add('fail');
  el.querySelector('.log-node').classList.remove('spin');
  el.querySelector('.log-meta').textContent = `失败 · ${secs}s`;
  const detail = document.createElement('div');
  detail.className = 'log-detail error';
  detail.textContent = message || '检索失败';
  el.appendChild(detail);
  activeToolStep = null;
  logListEl.scrollTop = logListEl.scrollHeight;
}

function logAnswerStart(basis) {
  const step = appendStep('info');
  step.innerHTML =
    '<span class="log-node"></span>' +
    '<div class="log-head"><span class="log-kind answer">回答</span>' +
    '<span class="log-action">开始生成回答</span>' +
    `<span class="log-time">${nowTime()}</span></div>` +
    `<div class="log-detail">${escapeHtml(basis)}</div>`;
  answerStep = { el: step, start: performance.now() };
}

function logAnswerDone() {
  if (answerStep) {
    const { el, start } = answerStep;
    const secs = ((performance.now() - start) / 1000).toFixed(1);
    el.classList.add('ok');
    el.querySelector('.log-action').textContent = '回答完成';
    el.querySelector('.log-detail').textContent += ` · 生成耗时 ${secs}s`;
    answerStep = null;
  }
  if (qaStart) {
    const total = ((performance.now() - qaStart) / 1000).toFixed(1);
    const genCalls = Math.max(0, llmCalls - decisionCalls);  // 生成次数 = 总调用 - 决策调用，避免写死 1 导致拆解自相矛盾
    const summary = llmCalls ? `模型调用 ${llmCalls} 次（决策 ${decisionCalls} + 生成 ${genCalls}）` : '';
    const step = appendStep('info');
    step.innerHTML = `<span class="log-node"></span><div class="log-meta">本轮总耗时 ${total}s${summary ? ' · ' + summary : ''}</div>`;
  }
  logListEl.scrollTop = logListEl.scrollHeight;
}

function logError(message) {
  if (llmStep) {
    llmStep.el.querySelector('.log-node').classList.remove('spin');
    llmStep.el.querySelector('.log-action').textContent = `第 ${llmCalls} 次调用 · 中断`;
    llmStep = null;
  }
  if (activeToolStep) logToolFail(message);
  const step = appendStep('fail');
  step.innerHTML =
    '<span class="log-node"></span>' +
    '<div class="log-head"><span class="log-kind answer">错误</span>' +
    `<span class="log-time">${nowTime()}</span></div>` +
    `<div class="log-detail error">${escapeHtml(message)}</div>`;
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

    const isWeb = s.collection === 'web';

    // 卡片头部：来源名 + 类型标签 + 相似度
    const head = document.createElement('div');
    head.className = 'source-card-head';

    const nameWrap = document.createElement('div');
    nameWrap.className = 'source-name-wrap';

    const tag = document.createElement('span');
    tag.className = `source-tag ${isWeb ? 'tag-web' : s.collection === 'faq' ? 'tag-faq' : 'tag-doc'}`;
    tag.textContent = isWeb ? '网页' : s.collection === 'faq' ? 'FAQ' : '文档';
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
    // 网页卡片展示标题（更可读），KB 卡片展示来源路径
    name.textContent = isWeb ? (s.title || s.source) : (s.source || '(未知来源)');
    name.title = s.source;
    nameWrap.appendChild(name);

    head.append(nameWrap);
    card.appendChild(head);

    if (isWeb) {
      // 网页卡片：Tavily 相关度与向量相似度不同义，不展示百分比，改显域名
      const host = document.createElement('div');
      host.className = 'source-host';
      try { host.textContent = new URL(s.source).hostname; } catch (e) { host.textContent = ''; }
      card.appendChild(host);
    } else {
      const score = document.createElement('span');
      score.className = 'source-score';
      const pct = Math.round((s.score || 0) * 100);
      score.textContent = `${pct}%`;
      score.title = `相似度 ${pct}%`;
      if (pct >= 80) score.classList.add('score-high');
      else if (pct >= 60) score.classList.add('score-mid');
      else score.classList.add('score-low');

      head.append(score);

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
    }

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
  pendingImages = [];
  renderImagePreview();
  closeSceneModal();
  renderChatWelcome();
  setLogCollapsed(true);  // 新会话无日志 → 自动收起面板
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
  pendingImages = [];
  renderImagePreview();
  closeSceneModal();
  await loadSessionMessages(id);
  await loadSessions();
}

async function loadSessionMessages(id) {
  const res = await fetch(`/api/sessions/${id}/messages`);
  const data = await res.json();
  messagesEl.innerHTML = '';
  resetLog();  // 先恢复空状态，若该会话有轨迹则随后回放
  setLogCollapsed(true);  // 历史会话自动收起日志面板
  const msgs = data.messages || [];
  if (!msgs.length) {
    renderChatWelcome();
    return;
  }
  // 后端返回正序（旧→新），直接逐条 append 即可；图片附件随消息一并回放
  msgs.forEach(m => addMessage(m.role, m.content, m.images || null));
  // 回放该会话的历史执行轨迹（有则渲染，替代空状态；面板仍保持收起，展开即可见）
  await loadTraceForSession(id);
}

// 拉取并回放历史会话的执行轨迹；失败保持空状态，不影响历史消息展示。
async function loadTraceForSession(id) {
  try {
    const res = await fetch(`/api/sessions/${id}/trace`);
    const data = await res.json();
    if (data.events && data.events.length) renderTrace(data.events);
  } catch (e) {
    // 忽略：轨迹回放失败不阻断历史消息
  }
}

// 静态回放一段执行轨迹（区别于 sendQuestion 内的实时流式渲染，
// 这里直接渲染最终态：决策/生成、检索词、命中数、结果列表、失败信息，不做耗时计时）。
function renderTrace(events) {
  // 实时流进行中不回放：否则轨迹节点会叠加到实时时间线上，造成顺序错乱
  if (liveStreaming) return;
  resetLog();  // 回放前先清空既有节点，防止与上一轮日志叠加
  const empty = logListEl.querySelector('.log-empty');
  if (empty) empty.remove();

  let modelEl = null, modelCall = 0, toolEl = null, visionEl = null;

  const relabelModel = (text) => {
    if (!modelEl) return;
    modelEl.classList.remove('running');
    modelEl.classList.add('info');
    modelEl.querySelector('.log-node').classList.remove('spin');
    modelEl.querySelector('.log-action').textContent = text;
    modelEl = null;
  };

  events.forEach(evt => {
    if (evt.type === 'question') {
      // 新一轮提问边界：先收尾上一个问题悬空的模型步骤，再开"提问"节点
      if (modelEl) relabelModel(`第 ${modelCall} 次调用 · 生成回答`);
      toolEl = null;
      const step = appendStep('info');
      step.innerHTML =
        '<span class="log-node"></span>' +
        '<div class="log-head"><span class="log-kind question">提问</span>' +
        '<span class="log-action">用户问题</span></div>' +
        `<div class="log-detail">${escapeHtml(evt.question || '')}</div>`;
    } else if (evt.type === 'vision_start') {
      // 截图问答：多模态解析节点（先于该轮的"提问/模型/检索"节点出现）
      if (modelEl) relabelModel(`第 ${modelCall} 次调用 · 生成回答`);
      toolEl = null;
      visionEl = appendStep('running');
      visionEl.innerHTML =
        '<span class="log-node spin"></span>' +
        '<div class="log-head"><span class="log-kind model">多模态</span>' +
        `<span class="log-action">图片解析 · ${escapeHtml(evt.label || '')}</span></div>`;
    } else if (evt.type === 'vision_done') {
      if (visionEl) {
        visionEl.classList.remove('running');
        visionEl.classList.add('ok');
        visionEl.querySelector('.log-node').classList.remove('spin');
        visionEl.querySelector('.log-action').textContent = '图片解析完成';
        visionEl = null;
      }
    } else if (evt.type === 'llm_start') {
      if (modelEl) relabelModel(`第 ${modelCall} 次调用 · 生成回答`);
      modelCall = evt.call;
      modelEl = appendStep('running');
      modelEl.innerHTML =
        '<span class="log-node spin"></span>' +
        '<div class="log-head"><span class="log-kind model">模型</span>' +
        `<span class="log-action">第 ${modelCall} 次调用 · 推理中</span></div>`;
    } else if (evt.type === 'kb_searching' || evt.type === 'searching') {
      if (modelEl) relabelModel(`第 ${modelCall} 次调用 · 决策：调用工具`);
      const kind = evt.type === 'kb_searching' ? 'kb' : 'web';
      const label = kind === 'kb' ? '本地检索' : '联网搜索';
      toolEl = appendStep('running');
      toolEl.innerHTML =
        '<span class="log-node spin"></span>' +
        '<div class="log-head">' +
        `<span class="log-kind ${kind}">${label}</span></div>` +
        `<div class="log-query">检索词 <code>${escapeHtml(evt.query || '')}</code></div>` +
        '<div class="log-meta">执行中…</div>';
    } else if (evt.type === 'sources' || evt.type === 'web_sources') {
      fillToolDone(toolEl, evt.type === 'web_sources' ? 'web' : 'kb', evt.documents || []);
      toolEl = null;
    } else if (evt.type === 'search_failed') {
      fillToolFailed(toolEl, evt.message || '检索失败');
      toolEl = null;
    }
  });

  if (modelEl) relabelModel(`第 ${modelCall} 次调用 · 生成回答`);
}

function fillToolDone(step, kind, docs) {
  if (!step) return;
  step.classList.remove('running');
  step.classList.add('ok');
  step.querySelector('.log-node').classList.remove('spin');
  step.querySelector('.log-meta').textContent = `返回 ${docs.length} 条`;
  if (docs.length) {
    const ul = document.createElement('ul');
    ul.className = 'log-results';
    docs.forEach(d => {
      const li = document.createElement('li');
      const idx = `<span class="log-result-idx">[资料${d.index}]</span>`;
      if (kind === 'web' && d.source) {
        li.innerHTML = idx;
        const a = document.createElement('a');
        a.href = d.source; a.target = '_blank'; a.rel = 'noopener';
        a.textContent = _resultTitle(d, kind);
        li.appendChild(a);
        const src = document.createElement('span');
        src.className = 'log-result-src';
        src.textContent = _domain(d.source);
        li.appendChild(src);
      } else {
        li.innerHTML = idx + escapeHtml(_resultTitle(d, kind));
      }
      ul.appendChild(li);
    });
    const toggle = document.createElement('button');
    toggle.className = 'log-results-toggle';
    toggle.textContent = '查看返回结果 ▸';
    toggle.onclick = () => {
      const expanded = step.classList.toggle('expanded');
      toggle.textContent = expanded ? '收起结果 ▾' : '查看返回结果 ▸';
    };
    step.append(toggle, ul);
  }
}

function fillToolFailed(step, message) {
  if (!step) return;
  step.classList.remove('running');
  step.classList.add('fail');
  step.querySelector('.log-node').classList.remove('spin');
  step.querySelector('.log-meta').textContent = '失败';
  const detail = document.createElement('div');
  detail.className = 'log-detail error';
  detail.textContent = message;
  step.appendChild(detail);
}

function renderChatWelcome() {
  messagesEl.innerHTML = '';
  resetLog();
  addMessage('bot', '你好！知识文件已在服务启动时自动入库，直接基于你的资料提问即可。');
}

// ---------- 问答（SSE 流式） ----------
// scene 为空 = 原有文字问答入口（行为不变）；scene 非空 = 截图问答（场景 key）
async function sendQuestion(scene) {
  let question = inputEl.value.trim();
  const images = pendingImages.slice();
  if (!images.length && !question) return;
  inputEl.value = '';
  sendBtn.disabled = true;
  liveStreaming = true;

  // 图片随本轮发出，清空待发送队列与预览
  pendingImages = [];
  renderImagePreview();

  // 用户气泡：图片问答与纯文字问答统一，只含用户文字 + 图片；
  // 图片解析（vision）是模型侧行为，归属模型气泡的 thinking 区域。
  addMessage('user', question, images.length ? images : null);
  if (scene) {
    const label = SCENE_LABELS[scene] || '截图';
    logSceneStart(label);  // 右侧日志：先记"图片解析"节点并清空上一轮日志
  } else {
    logQuestion(question);  // 右侧日志：新一轮问答，先记提问并清空上一轮日志
  }
  // 立即创建助手消息骨架（含"思考中"占位），检索期间页面不再空白
  const built = buildBotMessage();
  let { body, answerEl, thinking } = built;
  let fullText = '';   // 流式累积的回答原文
  let gotToken = false;
  let sourcesCount = 0;    // 本轮实际资料数（KB + 联网），判定 [资料N] 角标是否越界
  let currentSources = []; // 本轮累计资料（KB + 联网），网页资料编号接续其后
  let sourcesBlock = null; // 当前资料面板 DOM（sources/web_sources 到达时整体重渲染）

  const removeThinking = () => { if (thinking) { thinking.remove(); thinking = null; } };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        session_id: currentSessionId,
        web_search_enabled: webSearchCheck.checked,
        images: images,
        scene: scene || '',
      }),
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

        if (evt.type === 'vision_start') {
          // 图片解析开始：模型侧 thinking 更新为"正在解析图片…"
          if (thinking) thinking.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span> 正在解析图片…';
        } else if (evt.type === 'vision_token') {
          // 解析文本是模型中间产物，不再展示到聊天气泡（右侧日志已有记录）
        } else if (evt.type === 'vision_done') {
          // 解析结束：thinking 恢复为"思考中…"，进入 Agent 工具循环
          if (thinking) thinking.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span> 思考中';
          logVisionDone();
        } else if (evt.type === 'llm_start') {
          // 右侧日志：模型第 N 次调用开始（是决策还是生成，由后续事件定型）
          logLlmStart(evt.call);
        } else if (evt.type === 'kb_searching') {
          // 该次模型调用以调用工具告终 → 定型为决策；右侧日志：本地知识库检索开始
          logLlmDecision();
          logToolStart('kb', evt.query);
        } else if (evt.type === 'sources') {
          // 右侧日志：本地检索完成（结果数 + 耗时，可展开看每条标题）
          logToolFinish(evt.documents, 'kb');
          // KB 资料到达：追加进累计列表并整体重渲染资料面板（编号接续）
          currentSources = currentSources.concat(evt.documents || []);
          sourcesCount = currentSources.length;
          if (sourcesBlock) sourcesBlock.remove();
          sourcesBlock = showSources(body, currentSources, answerEl);
        } else if (evt.type === 'searching') {
          // 该次模型调用以调用工具告终 → 定型为决策；右侧日志：联网搜索开始
          logLlmDecision();
          logToolStart('web', evt.query);
        } else if (evt.type === 'web_sources') {
          // 右侧日志：联网搜索完成
          logToolFinish(evt.documents, 'web');
          // 联网资料到达：追加进累计列表并整体重渲染资料面板（编号接续 KB 之后）
          currentSources = currentSources.concat(evt.documents || []);
          sourcesCount = currentSources.length;
          if (sourcesBlock) sourcesBlock.remove();
          sourcesBlock = showSources(body, currentSources, answerEl);
        } else if (evt.type === 'search_failed') {
          // 右侧日志：检索失败/无结果（红色标记，模型会自行声明降级）
          logToolFail(evt.message);
        } else if (evt.type === 'token') {
          if (!gotToken) {
            removeThinking();
            const basis = currentSources.length
              ? `基于 ${currentSources.length} 条检索资料作答`
              : '未检索资料，基于模型自身知识作答';
            if (llmStep) {
              // 进行中的"模型调用"步骤就地定型为生成回答（本次调用 = 生成）
              const { el } = llmStep;
              el.classList.remove('running');
              el.classList.add('info');
              el.querySelector('.log-node').classList.remove('spin');
              el.innerHTML =
                '<span class="log-node"></span>' +
                '<div class="log-head"><span class="log-kind answer">回答</span>' +
                '<span class="log-action">开始生成回答</span>' +
                `<span class="log-time">${nowTime()}</span></div>` +
                `<div class="log-detail">${escapeHtml(basis)}</div>`;
              answerStep = { el, start: performance.now() };
              llmStep = null;
            } else {
              logAnswerStart(basis);
            }
            gotToken = true;
          }
          fullText += evt.content;
          // 每个 token 后重渲染 Markdown + 引用角标（文本量小，性能无压力）
          answerEl.innerHTML = renderAnswerHTML(fullText, sourcesCount, true);
        } else if (evt.type === 'error') {
          removeThinking();
          logError(evt.message);
          const errEl = document.createElement('div');
          errEl.className = 'chat-error';
          errEl.textContent = '[错误] ' + evt.message;
          answerEl.appendChild(errEl);
        } else if (evt.type === 'done') {
          removeThinking();
          logAnswerDone();
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
    logError('请求失败：' + e.message);
    const errEl = document.createElement('div');
    errEl.className = 'chat-error';
    errEl.textContent = '请求失败：' + e.message;
    answerEl.appendChild(errEl);
  } finally {
    sendBtn.disabled = false;
    liveStreaming = false;
  }
}

// ---------- 事件绑定 ----------
// 待发图片存在 → 先弹场景选择（选定后由弹窗发起）；否则走原文字问答入口
function handleSend() {
  if (pendingImages.length) {
    openSceneModal();
    return;
  }
  sendQuestion();
}

sendBtn.onclick = handleSend;

// 图片上传：点按钮 → 触发文件选择（支持多选）
imageBtn.onclick = () => imageInput.click();
imageInput.addEventListener('change', e => {
  addImageFiles(e.target.files);
  imageInput.value = '';  // 允许连续选择同一文件
});

// 粘贴截图：剪贴板含图片时拦截并取图压缩入队；纯文本粘贴放行。
// 挂到 document 层，弹窗打开时补第二张图也能粘贴（不依赖输入框聚焦）
document.addEventListener('paste', e => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  const files = [];
  for (const it of items) {
    if (it.type && it.type.startsWith('image/')) {
      const f = it.getAsFile();
      if (f) files.push(f);
    }
  }
  if (files.length) {
    e.preventDefault();
    addImageFiles(files);
  }
});

// 拖拽截图到聊天区
const chatEl = document.querySelector('.chat');
chatEl.addEventListener('dragover', e => { e.preventDefault(); });
chatEl.addEventListener('drop', e => {
  e.preventDefault();
  addImageFiles(e.dataTransfer && e.dataTransfer.files);
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
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
  await loadSessions();
}

init();
