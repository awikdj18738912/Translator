(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const startButton = $('translation-start');
  const stopButton = $('translation-stop');
  const downloadButton = $('translation-download');
  const textTab = $('translation-text-tab');
  const audioTab = $('translation-audio-tab');
  const textView = $('translation-text-view');
  const audioView = $('translation-audio-view');
  const status = $('translation-status');
  const log = $('translation-log');
  const audioFile = $('translation-audio-file');
  let runCounter = 0;
  let activeRun = null;
  let audioReady = false;
  let serviceReady = false;
  let pipelineReady = false;
  let isAudioTab = true;
  let events = [];
  let serializedEvents = [];
  let lastSourceRevision = 0;
  let lastTranslationRevision = 0;
  let lastTranslationPhase = '';
  let lastPreviewSequence = -1;
  let queuedTranslationRender = null;
  let translationRenderFrame = null;
  let translationRenderUsesAnimationFrame = false;
  let lastRefinementRevision = 0;
  let logRenderScheduled = false;
  let initialHealthComplete = false;
  let elapsedStartedAt = null;
  let elapsedFrozenAt = null;
  let elapsedTimer = null;
  let modelCallsMaximum = 0;
  let legacyCallsMaximum = 0;
  let legacyTotalCallsMaximum = 0;
  let hasSessionCallMetrics = false;
  let gateSkipsMaximum = 0;
  let gateMetricsProvided = false;
  let audioDurationReceived = false;
  let asrTotalMsMaximum = null;

  function renderAsrTime(value = asrTotalMsMaximum) {
    $('translation-asr-time').textContent = !isAudioTab
      ? '不适用'
      : value === null ? '—' : `${(value / 1000).toFixed(1)} s`;
  }

  function renderElapsed(now = performance.now()) {
    if (elapsedStartedAt === null) return;
    const end = elapsedFrozenAt === null ? now : elapsedFrozenAt;
    $('translation-total-time').textContent = `${(Math.max(0, end - elapsedStartedAt) / 1000).toFixed(1)} s`;
  }
  function beginElapsed() {
    elapsedStartedAt = performance.now();
    elapsedFrozenAt = null;
    renderElapsed();
    elapsedTimer = window.setInterval(() => renderElapsed(), 150);
  }
  function freezeElapsed() {
    if (elapsedStartedAt === null || elapsedFrozenAt !== null) return;
    elapsedFrozenAt = performance.now();
    if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
    elapsedTimer = null;
    renderElapsed();
  }
  function renderCallMetrics() {
    $('translation-calls').textContent = hasSessionCallMetrics
      ? String(modelCallsMaximum)
      : `${legacyCallsMaximum} / ${legacyTotalCallsMaximum}`;
  }

  log.title = '仅展示最近60条，下载包含全部';
  log.setAttribute('aria-label', '翻译事件日志。仅展示最近60条，下载包含全部。');

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle('error', isError);
  }
  function syncSourceLanguage() {
    $('source-language-label').textContent = '中文';
    syncTargetLanguage();
  }
  function syncTargetLanguage(code) {
    const language = $(isAudioTab ? 'translation-audio-language' : 'translation-language');
    if (code && language && [...language.options].some(option => option.value === code)) language.value = code;
    const label = $('target-language-label');
    if (label) label.textContent = language?.selectedOptions?.[0]?.textContent || '英语';
  }
  function setTab(audio) {
    if (activeRun) return;
    audioDurationReceived = false;
    clearAudioDuration(audio ? '—' : '不适用');
    isAudioTab = audio;
    asrTotalMsMaximum = null;
    renderAsrTime();
    textTab.setAttribute('aria-selected', String(!audio));
    audioTab.setAttribute('aria-selected', String(audio));
    textView.hidden = audio;
    audioView.hidden = !audio;
    startButton.disabled = !serviceReady || !pipelineReady || (audio && !audioReady);
    startButton.textContent = audio ? '开始音频处理' : '开始精修并翻译';
    syncSourceLanguage();
  }
  textTab.addEventListener('click', () => setTab(false));
  audioTab.addEventListener('click', () => setTab(true));
  function clearAudioDuration(value = '—') {
    $('translation-audio-duration').textContent = value;
  }
  audioFile.addEventListener('change', () => {
    if (activeRun) return;
    audioDurationReceived = false;
    clearAudioDuration();
    asrTotalMsMaximum = null;
    renderAsrTime();
  });

  function populateLanguages(languages) {
    // Target languages are intentionally fixed by the UI contract.
    void languages;
  }
  async function refreshHealth() {
    try {
      const response = await fetch('/api/translation/health');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      populateLanguages(data.languages);
      pipelineReady = data.pipeline_direction === 'zh_to_foreign';
      const gateInfo = $('translation-gate-info');
      if (gateInfo && !activeRun) gateInfo.textContent = data.refinement_gate_mode === 'regex'
        ? '正则门控 · 不使用置信度' : '门控关闭';
      if (!activeRun) syncSourceLanguage();
      serviceReady = data.ready === true;
      audioReady = data.audio_ready === true;
      audioTab.disabled = !audioReady;
      $('translation-audio-help').textContent = audioReady
        ? '选择一个音频文件，服务会自动分块回放并推送翻译结果。'
        : '语音转录服务正在准备，可先使用文本回放。';
      if (!activeRun) {
        if (!pipelineReady) {
          setStatus('正在切换中文精修翻译流程，请稍后再试。');
        } else if (!events.length || !serviceReady) {
          setStatus(serviceReady
            ? (audioReady ? '翻译与音频服务已就绪。' : '翻译服务已就绪；音频翻译暂不可用。')
            : '翻译服务尚未就绪。', !serviceReady);
        }
        startButton.disabled = !serviceReady || !pipelineReady || (isAudioTab && !audioReady);
      }
    } catch (error) {
      if (!initialHealthComplete) {
        serviceReady = false;
        audioReady = false;
        pipelineReady = false;
        audioTab.disabled = true;
        $('translation-audio-help').textContent = '暂时无法连接翻译服务，请稍后重试。';
      }
      if (!activeRun) {
        startButton.disabled = !serviceReady || !pipelineReady || (isAudioTab && !audioReady);
        setStatus(`无法检查翻译服务：${error.message}`, true);
      }
    } finally {
      initialHealthComplete = true;
    }
  }
  refreshHealth();
  window.setInterval(refreshHealth, 10000);

  function addLog(event) {
    events.push(event);
    serializedEvents.push(JSON.stringify(event));
    if (!logRenderScheduled) {
      logRenderScheduled = true;
      const render = () => {
        logRenderScheduled = false;
        const recent = serializedEvents.slice(-60);
        log.textContent = `仅展示最近60条，下载包含全部\n${recent.length ? recent.join('\n') : '等待服务端事件…'}`;
      };
      if (typeof window.requestAnimationFrame === 'function') window.requestAnimationFrame(render);
      else window.setTimeout(render, 100);
    }
    downloadButton.disabled = events.length === 0;
  }
  function queueTranslationRender(text, label) {
    queuedTranslationRender = { text, label };
    if (translationRenderFrame !== null) return;
    translationRenderUsesAnimationFrame = typeof window.requestAnimationFrame === 'function';
    const render = () => {
      translationRenderFrame = null;
      const queued = queuedTranslationRender;
      queuedTranslationRender = null;
      if (!queued) return;
      $('translation-original').textContent = queued.text;
      $('translation-phase-label').textContent = queued.label;
    };
    translationRenderFrame = translationRenderUsesAnimationFrame
      ? window.requestAnimationFrame(render)
      : window.setTimeout(render, 16);
  }
  function acceptTranslation(event, phase) {
    const revision = Number(event.translation_revision ?? event.revision);
    if (!Number.isFinite(revision) || typeof event.translation !== 'string') return false;
    const sequence = phase === 'preview' ? Number(event.preview_sequence) : -1;
    if (phase === 'preview' && !Number.isFinite(sequence)) return false;

    let accepted = false;
    if (revision > lastTranslationRevision) {
      accepted = true;
    } else if (revision === lastTranslationRevision) {
      const phaseRank = { '': -1, preview: 0, complete: 1, final: 2 };
      const previousRank = phaseRank[lastTranslationPhase] ?? -1;
      const nextRank = phaseRank[phase] ?? 0;
      accepted = nextRank > previousRank
        || (phase === 'preview' && lastTranslationPhase === 'preview' && sequence > lastPreviewSequence);
    }
    if (!accepted) return false;

    lastTranslationRevision = revision;
    lastTranslationPhase = phase;
    lastPreviewSequence = phase === 'preview' ? sequence : -1;
    const label = phase === 'preview' ? '翻译中' : phase === 'final' ? '最终译文' : '片段完成';
    queueTranslationRender(event.translation, label);
    return true;
  }
  function resetRun() {
    if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
    elapsedTimer = null;
    elapsedStartedAt = null;
    elapsedFrozenAt = null;
    modelCallsMaximum = 0;
    legacyCallsMaximum = 0;
    legacyTotalCallsMaximum = 0;
    hasSessionCallMetrics = false;
    gateSkipsMaximum = 0;
    gateMetricsProvided = false;
    audioDurationReceived = false;
    asrTotalMsMaximum = null;
    renderAsrTime();
    clearAudioDuration(isAudioTab ? '读取中…' : '不适用');
    events = [];
    serializedEvents = [];
    lastSourceRevision = 0;
    lastTranslationRevision = 0;
    lastTranslationPhase = '';
    lastPreviewSequence = -1;
    queuedTranslationRender = null;
    if (translationRenderFrame !== null) {
      if (translationRenderUsesAnimationFrame) window.cancelAnimationFrame(translationRenderFrame);
      else window.clearTimeout(translationRenderFrame);
      translationRenderFrame = null;
    }
    lastRefinementRevision = 0;
    log.textContent = '仅展示最近60条，下载包含全部\n等待服务端事件…';
    $('translation-source').textContent = '';
    $('translation-original').textContent = '';
    $('translation-phase-label').textContent = '等待翻译';
    $('translation-clean').textContent = '';
    $('quality-tag').textContent = '等待精修';
    $('translation-calls').textContent = '0';
    $('translation-total-time').textContent = '0.0 s';
    $('translation-gate-skips').textContent = '0';
    $('translation-guard').textContent = '等待运行';
    $('translation-time').textContent = '—';
    $('refinement-time').textContent = '—';
    downloadButton.disabled = true;
  }
  function wsUrl(path) {
    const url = new URL(path, location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return url.href;
  }
  function setRunning(running, run) {
    if (activeRun !== run) return;
    if (running) {
      startButton.disabled = true;
      stopButton.disabled = false;
    } else {
      if (isAudioTab && !audioDurationReceived) clearAudioDuration();
      activeRun = null;
      stopButton.disabled = true;
      startButton.disabled = !serviceReady || !pipelineReady || (isAudioTab && !audioReady);
    }
  }
  function renderRefinedText(event) {
    const clean = $('translation-clean');
    if (typeof event.display_refined_text === 'string' && typeof event.pending_raw_text === 'string') {
      clean.replaceChildren();
      if (event.display_refined_text) clean.append(document.createTextNode(event.display_refined_text));
      if (event.pending_raw_text) {
        const pending = document.createElement('span');
        pending.className = 'pending-refinement';
        pending.title = '等待后台精修';
        pending.setAttribute('aria-label', '待精修文本');
        pending.textContent = event.pending_raw_text;
        clean.append(pending);
      }
      return;
    }
    if (typeof event.clean_text === 'string') clean.textContent = event.clean_text;
  }
  function updateFromEvent(event) {
    if (event.event === 'ready' && typeof event.target_language === 'string') {
      syncTargetLanguage(event.target_language);
    }
    if (isAudioTab && Number.isFinite(event.asr_total_ms) && event.asr_total_ms >= 0) {
      asrTotalMsMaximum = Math.max(asrTotalMsMaximum ?? 0, event.asr_total_ms);
      renderAsrTime();
    }
    const metrics = event.session_metrics;
    if (metrics && typeof metrics === 'object') {
      if (Number.isFinite(Number(metrics.refiner_model_calls))) {
        hasSessionCallMetrics = true;
        modelCallsMaximum = Math.max(modelCallsMaximum, Number(metrics.refiner_model_calls));
        renderCallMetrics();
      }
      if (Number.isFinite(Number(metrics.gate_skipped_groups))) {
        gateMetricsProvided = true;
        gateSkipsMaximum = Math.max(gateSkipsMaximum, Number(metrics.gate_skipped_groups));
        $('translation-gate-skips').textContent = String(gateSkipsMaximum);
        const info = $('translation-gate-info');
        if (info) info.textContent = `门控累计跳过 ${gateSkipsMaximum} 组`;
      }
    }
    const eventRevision = Number.isFinite(event.revision) ? event.revision : null;
    if (event.event === 'final' && eventRevision !== null && eventRevision < lastSourceRevision) return false;
    if (event.event === 'refining') {
      const version = eventRevision === null ? '—' : eventRevision;
      setStatus(`正在精修第 ${version} 版译文…`);
      if ($('translation-clean').querySelector('.pending-refinement')) {
        $('quality-tag').textContent = '精修中';
      }
      return true;
    }
    if (event.event === 'translation_preview') {
      acceptTranslation(event, 'preview');
      return true;
    }
    if (event.event === 'translation_finalizing') {
      setStatus('正在生成最终译文…');
      if (queuedTranslationRender) queuedTranslationRender.label = '最终翻译中';
      $('translation-phase-label').textContent = '最终翻译中';
      return true;
    }
    if (typeof event.source_text === 'string') {
      const revision = Number(event.source_revision ?? eventRevision ?? lastSourceRevision + 1);
      if (revision >= lastSourceRevision) {
        lastSourceRevision = revision;
        $('translation-source').textContent = event.source_text;
        if (event.event === 'source') $('translation-guard').textContent = '等待更新';
      }
    }
    if (isAudioTab && Number.isFinite(event.audio_total_seconds) && event.audio_total_seconds >= 0) {
      audioDurationReceived = true;
      clearAudioDuration(`${event.audio_total_seconds.toFixed(1)} s`);
    }
    if (Number.isFinite(event.audio_processed_seconds) && Number.isFinite(event.audio_total_seconds)) {
      setStatus(`音频已处理 ${event.audio_processed_seconds.toFixed(1)} / ${event.audio_total_seconds.toFixed(1)} 秒`);
    }
    if (typeof event.translation === 'string'
      && ['translation', 'update', 'final'].includes(event.event)) {
      const translationAccepted = acceptTranslation(event, event.event === 'final' ? 'final' : 'complete');
      if (translationAccepted && event.event === 'translation' && event.translation_stats?.strategy === 'adaptive_context_v1'
        && event.translation_stats.stable_windows < event.translation_stats.windows) {
        if (queuedTranslationRender) queuedTranslationRender.label = '待确认译文';
        $('translation-phase-label').textContent = '待确认译文';
      }
    }
    const refinementRevision = Number(event.refinement_revision ?? event.revision);
    const hasRefinedDisplay = (typeof event.display_refined_text === 'string' && typeof event.pending_raw_text === 'string')
      || typeof event.clean_text === 'string';
    let displayAccepted = false;
    if (hasRefinedDisplay && Number.isFinite(refinementRevision)) {
      const displayIsCurrent = refinementRevision >= lastRefinementRevision;
      if (displayIsCurrent) {
        displayAccepted = true;
        renderRefinedText(event);
        lastRefinementRevision = Math.max(lastRefinementRevision, refinementRevision);
        if (event.has_pending_refinement === true || event.pending_raw_text) {
          $('quality-tag').textContent = event.display_refined_text ? '部分待精修' : '待精修';
        } else if (event.refiner_executed === true) {
          $('quality-tag').textContent = event.refiner_accepted === true ? '已接受' : event.refiner_accepted === false ? '已回退' : '精修完成';
        } else if (event.refinement_state === 'refining') {
          $('quality-tag').textContent = '精修中';
        } else if (event.refinement_state === 'rejected') {
          $('quality-tag').textContent = '已回退';
        } else if (event.refiner_executed === false) {
          $('quality-tag').textContent = event.refinement_gate_mode === 'regex' && event.has_pending_refinement === false ? '规则保留' : '待精修';
        }
      }
    }
    if (!metrics && (event.model_calls !== undefined || event.total_model_calls !== undefined)) {
      const legacy = Number(event.model_calls ?? 0);
      const legacyTotal = Number(event.total_model_calls ?? event.model_calls ?? 0);
      if (Number.isFinite(legacy)) legacyCallsMaximum = Math.max(legacyCallsMaximum, legacy);
      if (Number.isFinite(legacyTotal)) legacyTotalCallsMaximum = Math.max(legacyTotalCallsMaximum, legacyTotal);
      renderCallMetrics();
    }
    if (displayAccepted && event.refinement_state !== undefined) {
      const stateLabels = {
        waiting_for_context: '等待后文', waiting_for_group: '等待完整片段',
        window_refined: '精修完成', waiting_for_complete_segment: '等待完整片段',
        waiting_for_translation: '等待翻译', refining: '精修中', completed: '精修完成',
        complete: '精修完成', accepted: '精修完成', rejected: '守卫拒绝精修', skipped: '跳过精修',
      };
      const reasonLabels = {
        numeric_value_mismatch: '数量或单位校验未通过', numeric_mismatch: '数字校验未通过',
        entity_mismatch: '实体校验未通过', meaning_drift: '语义校验未通过',
        empty_refinement: '精修结果为空', invalid_refinement: '精修结果无效',
      };
      const rawReasons = Array.isArray(event.refiner_reject_reasons) ? event.refiner_reject_reasons : [];
      const reasons = rawReasons.length ? `：${rawReasons.map(item => {
        const key = typeof item === 'string' ? item : (item && (item.code || item.reason || item.key)) || String(item);
        return reasonLabels[key] || key;
      }).join('、')}` : '';
      const accepted = event.refiner_accepted === true ? '已接受' : event.refiner_accepted === false ? '已回退' : '';
      const state = stateLabels[event.refinement_state] || event.refinement_state;
      $('translation-guard').textContent = `${state}${accepted ? ` · ${accepted}` : ''}${reasons}`;
    } else if (displayAccepted && event.refiner_executed !== undefined) {
      $('translation-guard').textContent = event.refiner_executed ? (event.refiner_accepted ? '已执行并接受' : '已执行，结果未接受') : '未执行';
    }
    if (displayAccepted && (event.has_pending_refinement === true || event.pending_raw_text)) {
      $('translation-guard').textContent = event.display_refined_text ? '已有确认前缀，仍有文本待精修' : '等待精修';
    }
    if (displayAccepted && event.refinement_gate_mode === 'regex') {
      const gateInfo = $('translation-gate-info');
      if (gateInfo && !gateMetricsProvided) gateInfo.textContent = '正则门控 · 跳过次数未提供';
      if (event.refiner_executed === false && event.has_pending_refinement === false) $('translation-guard').textContent = '规则未命中，保留译文';
    }
    if (event.translation_ms !== undefined) $('translation-time').textContent = `${formatMs(event.translation_ms)} ms`;
    if (event.refinement_ms !== undefined) $('refinement-time').textContent = `${formatMs(event.refinement_ms)} ms`;
    return true;
  }
  function formatMs(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(1) : String(value);
  }
  function connect(path, textPayload, run) {
    const socket = new WebSocket(wsUrl(path));
    run.socket = socket;
    socket.addEventListener('open', () => {
      if (activeRun !== run || run.id !== runCounter) return socket.close();
      if (textPayload) socket.send(JSON.stringify(textPayload));
      setStatus(textPayload ? '已连接，正在回放文本…' : '已连接，正在处理音频…');
    });
    socket.addEventListener('message', message => {
      if (activeRun !== run || run.id !== runCounter) return;
      let event;
      try { event = JSON.parse(message.data); }
      catch (_) { addLog({ event: 'parse_error', detail: String(message.data) }); return; }
      addLog(event);
      const mayFinish = updateFromEvent(event);
      if (!mayFinish) return;
      if (event.event === 'ready') setStatus('服务已就绪，正在处理…');
      else if (event.event === 'error') {
        freezeElapsed();
        setStatus(event.detail || '翻译过程中发生错误。', true);
        $('quality-tag').textContent = '处理失败';
        run.terminal = true;
      } else if (event.event === 'final') {
        freezeElapsed();
        setStatus('中文精修与外语翻译完成。');
        run.terminal = true;
      } else if (event.event === 'refinement_progress') setStatus('已完成一组精修，正在处理后续内容…');
      else if (event.event === 'update') setStatus('收到新的翻译与精修结果…');
    });
    function markInterrupted() {
      if (activeRun !== run || run.id !== runCounter || run.terminal || run.cancelled) return;
      freezeElapsed();
      $('quality-tag').textContent = '连接中断';
      $('translation-guard').textContent = '服务连接中断，当前任务未完成';
      setStatus('WebSocket 连接中断，当前任务未完成。', true);
    }
    socket.addEventListener('error', markInterrupted);
    socket.addEventListener('close', () => {
      if (activeRun !== run || run.id !== runCounter) return;
      if (!run.terminal && !run.cancelled) markInterrupted();
      setRunning(false, run);
    });
  }
  async function startText(run) {
    const text = $('translation-input').value.trim();
    if (!text) throw new Error('请先输入待精修的中文文本。');
    const targetLanguage = $('translation-language').value;
    const domain = $('translation-domain').value;
    connect('/ws/translation', { event: 'start', source_language: 'zh', target_language: targetLanguage, domain, text, chunk_chars: 40, interval_ms: 500 }, run);
  }
  async function startAudio(run) {
    const file = audioFile.files && audioFile.files[0];
    if (!file) throw new Error('请先选择音频文件。');
    if (!audioReady) throw new Error('音频翻译服务未就绪。');
    const form = new FormData();
    form.append('file', file);
    form.append('target_language', $('translation-audio-language').value);
    form.append('domain', $('translation-audio-domain').value);
    form.append('chunk_seconds', '2');
    form.append('interval_ms', '0');
    const controller = new AbortController();
    run.uploadController = controller;
    const response = await fetch('/api/translation/audio', { method: 'POST', body: form, signal: controller.signal });
    run.uploadResponseSeen = true;
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const result = await response.json();
    if (!result.id || !result.websocket) throw new Error('音频服务未返回任务 ID 或 WebSocket 地址。');
    if (activeRun !== run || run.id !== runCounter || run.cancelled) {
      fetch(`/api/translation/audio/${encodeURIComponent(result.id)}`, { method: 'DELETE' }).catch(() => {});
      return;
    }
    connect(result.websocket, null, run);
  }
  startButton.addEventListener('click', async () => {
    if (activeRun) return;
    const run = {
      id: ++runCounter, socket: null, terminal: false, cancelled: false,
    };
    activeRun = run;
    syncSourceLanguage();
    resetRun();
    beginElapsed();
    setRunning(true, run);
    setStatus(isAudioTab ? '正在上传音频…' : '正在连接中文精修与翻译服务…');
    try {
      if (isAudioTab) await startAudio(run);
      else await startText(run);
    } catch (error) {
      if (activeRun !== run) return;
      addLog({ event: 'error', detail: error.message });
      run.terminal = true;
      freezeElapsed();
      setStatus(error.message, true);
      setRunning(false, run);
      if (run.socket && run.socket.readyState < WebSocket.CLOSING) run.socket.close();
      if (run.uploadController) run.uploadController.abort();
    }
  });
  $('translation-audio-language').addEventListener('change', () => { if (isAudioTab && !activeRun) syncTargetLanguage(); });
  $('translation-language').addEventListener('change', () => { if (!isAudioTab && !activeRun) syncTargetLanguage(); });
  syncSourceLanguage();
  stopButton.addEventListener('click', () => {
    const run = activeRun;
    if (!run) return;
    run.cancelled = true;
    run.terminal = true;
    freezeElapsed();
    setStatus('已取消。');
    if (run.uploadController && !run.uploadResponseSeen) run.uploadController.abort();
    if (run.socket) run.socket.close();
    else setRunning(false, run);
  });
  downloadButton.addEventListener('click', () => {
    if (!events.length) return;
    const blob = new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' });
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `translation-log-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  });
})();
