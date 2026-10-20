'use strict';
const $ = id => document.getElementById(id);
const state = {mode: 'audio', file: null, busy: false, recording: false, recorder: null, stream: null, timer: null, maxSeconds: 600, activeJob: null};
const stages = {queued:'排队中', preparing:'准备中', transcribing:'语音转录中', transcribed:'转录完成', translating:'翻译为中文', refining:'实体校正与中文精修', chunk_complete:'分段处理完成', completed:'处理完成', translation_rejected:'翻译校验未通过', interrupted:'任务已中断'};
const statuses = {queued:'排队中', running:'处理中', completed:'已完成', failed:'处理失败'};
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try { const body = await response.json(); message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail); } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}
function notice(message) { $('notice').textContent = message; $('notice').classList.toggle('hidden', !message); }
function showView(view) {
  document.querySelectorAll('.view').forEach(e => e.classList.toggle('hidden', e.id !== view));
  document.querySelectorAll('[data-view]').forEach(e => e.classList.toggle('active', e.dataset.view === view));
  if (view === 'glossary') loadEntities().catch(e => notice(e.message));
  if (view === 'history') loadHistory().catch(e => notice(e.message));
}
document.querySelectorAll('[data-view]').forEach(e => e.addEventListener('click', () => showView(e.dataset.view)));
function setMode(mode) {
  state.mode = mode;
  $('audio-input').classList.toggle('hidden', mode !== 'audio'); $('text-input').classList.toggle('hidden', mode !== 'text');
  for (const tab of ['audio', 'text']) { $(`${tab}-tab`).classList.toggle('selected', tab === mode); $(`${tab}-tab`).setAttribute('aria-selected', String(tab === mode)); }
}
$('audio-tab').addEventListener('click', () => setMode('audio'));
$('text-tab').addEventListener('click', () => setMode('text'));
function selectFile(file) { state.file = file; $('file-label').textContent = file ? file.name : '点击选择或拖放一段录音'; }
$('audio-file').addEventListener('change', e => selectFile(e.target.files[0] || null));
for (const name of ['dragenter', 'dragover']) $('drop-zone').addEventListener(name, e => { e.preventDefault(); $('drop-zone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('drop-zone').addEventListener(name, e => { e.preventDefault(); $('drop-zone').classList.remove('dragover'); });
$('drop-zone').addEventListener('drop', e => { if (!state.busy && !state.recording) selectFile(e.dataTransfer.files[0]); });
function setBusy(value) {
  state.busy = value;
  for (const id of ['submit','audio-file','source-text','language','domain','record','audio-tab','text-tab']) $(id).disabled = value;
  $('submit').textContent = value ? '本地处理中…' : '开始翻译 →';
}
$('record').addEventListener('click', async () => {
  if (state.recording) { state.recorder.stop(); return; }
  try {
    if (!navigator.mediaDevices || !window.MediaRecorder) throw new Error('此浏览器不支持录音，请在 localhost 上使用新版 Chrome、Edge 或 Firefox。');
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    state.stream = stream;
    const mimeType = ['audio/webm;codecs=opus','audio/ogg;codecs=opus','audio/mp4'].find(type => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(stream, mimeType ? {mimeType} : {});
    state.recorder = recorder;
    const chunks = []; let seconds = 0;
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      clearInterval(state.timer); stream.getTracks().forEach(track => track.stop());
      state.recording = false; $('record').classList.remove('recording'); $('record').textContent = '● 开始录音';
      for (const id of ['submit','audio-file','text-tab']) $(id).disabled = state.busy;
      const type = recorder.mimeType || 'audio/webm'; const ext = type.includes('ogg') ? 'ogg' : type.includes('mp4') ? 'm4a' : 'webm';
      selectFile(new File(chunks, `录音-${new Date().toISOString().replace(/[:.]/g,'-')}.${ext}`, {type}));
      $('record-time').textContent = `已录制 ${seconds} 秒，可开始翻译`;
    };
    recorder.onerror = () => { notice('录音失败，请检查麦克风后重试。'); if (recorder.state !== 'inactive') recorder.stop(); };
    recorder.start(1000); state.recording = true; notice('');
    $('record').classList.add('recording'); $('record').textContent = '■ 停止录音';
    for (const id of ['submit','audio-file','text-tab']) $(id).disabled = true;
    state.timer = setInterval(() => { seconds++; $('record-time').textContent = `${seconds} 秒`; if (seconds >= state.maxSeconds && recorder.state !== 'inactive') recorder.stop(); }, 1000);
  } catch (e) { state.stream?.getTracks().forEach(track => track.stop()); notice(e.name === 'NotAllowedError' ? '麦克风权限未开启，请允许浏览器使用麦克风。' : e.message); }
});
function renderJob(job) {
  const result = job.result || {};
  for (const [id, key, empty] of [['source-result','source_text','等待转录结果…'],['translation-result','translation','等待中文译文…'],['refined-result','refined_text','等待精修结果…']]) {
    $(id).textContent = result[key] || empty; $(id).classList.toggle('muted', !result[key]);
  }
  $('job-status').textContent = (statuses[job.status] || job.status) + (result.needs_review ? ' · 待核对' : '');
  const running = ['queued','running'].includes(job.status);
  $('progress').classList.toggle('hidden', !running);
  $('progress').textContent = (stages[job.stage] || job.stage) + (result.total_chunks ? ` · 分段 ${Math.min(result.chunks?.length || 0, result.total_chunks)} / ${result.total_chunks}` : '') + '。可在处理记录中查看进度。';
  const warnings = [...(result.warnings || []), ...(job.error ? [job.error] : [])];
  $('warnings').classList.toggle('hidden', !warnings.length); $('warnings').textContent = warnings.join('\n');
  $('exports').classList.toggle('hidden', !result.source_text);
  $('export-txt').href = `/api/jobs/${job.id}/export?format=txt`; $('export-json').href = `/api/jobs/${job.id}/export?format=json`;
  $('audit').classList.toggle('hidden', !result.chunks?.length);
  $('audit-content').textContent = JSON.stringify({models:result.models, elapsed_seconds:result.elapsed_seconds, asr:result.asr, chunks:result.chunks}, null, 2);
}
async function watchJob(id) {
  state.activeJob = id; setBusy(true);
  try {
    while (state.activeJob === id) {
      const job = await api(`/api/jobs/${id}`); renderJob(job);
      if (!['queued','running'].includes(job.status)) break;
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  } catch (e) { notice(`${e.message}；任务可能仍在后台运行，可在处理记录中重新打开。`); }
  finally { if (state.activeJob === id) setBusy(false); }
}
$('submit').addEventListener('click', async () => {
  notice(''); setBusy(true);
  try {
    const language = $('language').value, domain = $('domain').value.trim() || 'general'; let job;
    if (state.mode === 'audio') {
      if (!state.file) throw new Error('请先选择音频或录制一段语音。');
      const body = new FormData(); body.append('file', state.file); body.append('language', language); body.append('domain', domain);
      job = await api('/api/jobs/audio', {method:'POST', body});
    } else {
      const text = $('source-text').value.trim(); if (!text) throw new Error('请输入外语原文。');
      job = await api('/api/jobs/text', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text, language, domain})});
    }
    renderJob(job); await watchJob(job.id);
  } catch (e) { notice(e.message); setBusy(false); }
});
async function checkModels() {
  $('model-status').textContent = '正在检查本地推理服务…';
  try {
    const data = await api('/api/readiness');
    const items = [`转录：${data.asr.installed ? data.asr.model + ' / CPU 优先，权重推理时检查' : '尚未安装 ASR 依赖'}`, `翻译：${data.translation.model} ${data.translation.ready ? '已就绪' : '未就绪'}`, `精修：${data.refiner_enabled ? data.refiner.model + (data.refiner.ready ? ' 已就绪' : ' 未就绪') : '已关闭'}`];
    $('model-status').textContent = items.join('　·　');
  } catch (e) { $('model-status').textContent = e.message; }
}
$('check-models').addEventListener('click', checkModels);
const aliases = value => value.split(/[,，\n]/).map(s => s.trim()).filter(Boolean);
$('entity-form').addEventListener('submit', async e => {
  e.preventDefault(); const form = new FormData(e.target);
  const body = Object.fromEntries(form); body.source_aliases = aliases(body.source_aliases); body.zh_aliases = aliases(body.zh_aliases);
  try { await api('/api/entities', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}); notice('实体已保存。下一个任务将使用更新后的实体库。'); await loadEntities(); }
  catch (err) { notice(err.message); }
});
async function loadEntities() {
  const entities = await api('/api/entities'); $('entity-list').replaceChildren();
  if (!entities.length) { const p = document.createElement('p'); p.className = 'empty'; p.textContent = '实体库尚为空。添加产品名、品牌、贸易术语或行业专有名词。'; $('entity-list').append(p); }
  for (const entity of entities) {
    const row = document.createElement('div'); row.className = 'entity-row';
    const content = document.createElement('div'), title = document.createElement('strong'), description = document.createElement('p'), meta = document.createElement('small'), button = document.createElement('button');
    title.textContent = entity.canonical_zh; description.textContent = `外语：${entity.source_aliases.join(' / ') || '—'}\n中文别名：${entity.zh_aliases.join(' / ') || '—'}`;
    meta.textContent = `${entity.domain} · ${entity.language} · ${entity.policy === 'preserve' ? '保留拼写' : '标准译名'}${entity.enabled ? '' : ' · 已停用'}`;
    button.textContent = '删除'; button.className = 'delete-entity'; button.setAttribute('aria-label', `删除 ${entity.canonical_zh}`);
    button.addEventListener('click', async () => { if (!window.confirm(`删除“${entity.canonical_zh}”？`)) return; try { await api(`/api/entities/${entity.id}`, {method:'DELETE'}); await loadEntities(); } catch (e) { notice(e.message); } });
    content.append(title, description, meta); row.append(content, button); $('entity-list').append(row);
  }
}
$('import-entities').addEventListener('change', async e => {
  const file = e.target.files[0]; if (!file) return;
  try { const value = JSON.parse(await file.text()); const body = Array.isArray(value) ? {entities:value} : value; const r = await api('/api/entities/import', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}); notice(`已导入 ${r.ids.length} 个实体。`); await loadEntities(); }
  catch (err) { notice(err.message); } finally { e.target.value = ''; }
});
async function loadHistory() {
  const jobs = await api('/api/jobs'); $('history-list').replaceChildren();
  if (!jobs.length) { const p = document.createElement('p'); p.className = 'empty'; p.textContent = '尚无处理记录。'; $('history-list').append(p); }
  for (const job of jobs) {
    const row = document.createElement('button'); row.className = 'history-row';
    const status = document.createElement('span'), id = document.createElement('span'), time = document.createElement('small');
    status.textContent = statuses[job.status] || job.status; id.textContent = `${job.id.slice(0,10)} · ${stages[job.stage] || job.stage}`; id.className = 'history-id'; time.textContent = new Date(job.created_at).toLocaleString();
    row.append(status, id, time); row.addEventListener('click', () => { showView('workspace'); watchJob(job.id); }); $('history-list').append(row);
  }
}
async function init() {
  try {
    const health = await api('/api/health'); state.maxSeconds = health.max_audio_seconds;
    $('file-hint').textContent = `WAV、MP3、M4A、WEBM 等 · 最长 ${Math.floor(health.max_audio_seconds / 60)} 分钟 · 最大 ${health.max_upload_mb} MB`;
    for (const [code, name] of Object.entries(health.languages)) {
      if (code === 'auto') continue;
      for (const id of ['language','entity-language']) { const option = document.createElement('option'); option.value = code; option.textContent = name; $(id).append(option); }
    }
    await checkModels();
  } catch (e) { notice(e.message); }
}
window.addEventListener('beforeunload', () => state.stream?.getTracks().forEach(track => track.stop()));
init();
