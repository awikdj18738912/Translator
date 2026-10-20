(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const panel = $('entity-panel');
  let entities = [];
  let searchTimer;
  const escapeHtml = value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const notice = (message, error = false) => {
    $('entity-notice').textContent = message;
    $('entity-notice').classList.toggle('error', error);
  };
  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `请求失败（${response.status}）`);
    return payload;
  }
  function render() {
    const rows = $('entity-rows');
    if (!entities.length) {
      rows.innerHTML = '<tr><td colspan="7">没有符合条件的术语。</td></tr>';
      return;
    }
    rows.innerHTML = entities.map(item => `<tr class="${item.enabled ? '' : 'muted-row'}">
      <td><span class="chip ${item.enabled ? '' : 'disabled'}">${item.enabled ? '启用' : '停用'}</span></td>
      <td>${escapeHtml(item.canonical_text)}</td><td><div class="chips">${item.aliases.map(alias => `<span class="chip">${escapeHtml(alias)}</span>`).join('') || '—'}</div></td>
      <td>${escapeHtml(item.entity_type)}<br><span class="muted">${escapeHtml(item.domain)}</span></td>
      <td>${item.normalization_policy === 'normalize' ? '统一标准名' : '保留原写法'}</td><td>${Number(item.priority) || 0}</td>
      <td><div class="row-actions"><button class="button quiet compact" data-action="edit" data-id="${item.entity_id}" type="button">编辑</button><button class="button quiet compact" data-action="toggle" data-id="${item.entity_id}" type="button">${item.enabled ? '停用' : '启用'}</button></div></td>
    </tr>`).join('');
  }
  async function load() {
    notice('正在加载术语库…');
    const params = new URLSearchParams({include_disabled:'true'});
    if ($('entity-search').value.trim()) params.set('query', $('entity-search').value.trim());
    if ($('entity-type-filter').value) params.set('entity_type', $('entity-type-filter').value);
    try {
      const payload = await api(`/api/entities?${params}`);
      const list = Array.isArray(payload.entities) ? payload.entities : [];
      entities = list.filter(item => $('entity-status-filter').value === 'all' || ( $('entity-status-filter').value === 'enabled' ? item.enabled : !item.enabled));
      render(); notice(`已加载 ${entities.length} 条术语。`);
    } catch (error) { entities = []; render(); notice(error.message, true); }
  }
  function edit(item = null) {
    const form = $('entity-form');
    const ensureOption = (select, value) => {
      if (!value || [...select.options].some(option => option.value === value)) return;
      const option = document.createElement('option');
      option.value = value;
      option.textContent = `${value}（现有值）`;
      select.append(option);
    };
    form.dataset.entityId = item ? String(item.entity_id) : '';
    $('entity-dialog-title').textContent = item ? '编辑术语' : '新增术语';
    $('entity-canonical').value = item?.canonical_text || '';
    ensureOption($('entity-type'), item?.entity_type);
    ensureOption($('entity-domain'), item?.domain);
    $('entity-type').value = item?.entity_type || 'TERM';
    $('entity-domain').value = item?.domain || 'trade';
    $('entity-policy').value = item?.normalization_policy || 'normalize';
    $('entity-priority').value = String(item?.priority ?? 0);
    $('entity-aliases').value = item?.aliases?.join('\n') || '';
    $('entity-enabled').checked = item?.enabled ?? true;
    $('entity-form-error').textContent = '';
    $('entity-dialog').showModal(); $('entity-canonical').focus();
  }
  $('manage-entities').addEventListener('click', () => {
    panel.hidden = false; load(); panel.scrollIntoView({behavior:'smooth', block:'start'});
  });
  $('close-entities').addEventListener('click', () => { panel.hidden = true; });
  $('refresh-entities').addEventListener('click', load);
  $('add-entity').addEventListener('click', () => edit());
  $('cancel-entity').addEventListener('click', () => $('entity-dialog').close());
  $('entity-search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(load, 180); });
  $('entity-type-filter').addEventListener('change', load);
  $('entity-status-filter').addEventListener('change', load);
  $('entity-rows').addEventListener('click', async event => {
    const button = event.target.closest('button[data-action]'); if (!button) return;
    const item = entities.find(entry => entry.entity_id === Number(button.dataset.id)); if (!item) return;
    if (button.dataset.action === 'edit') return edit(item);
    button.disabled = true;
    try {
      await api(`/api/entities/${item.entity_id}/enabled`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({enabled:!item.enabled})});
      notice(item.enabled ? '术语已停用。' : '术语已启用。'); await load();
    } catch (error) { notice(error.message, true); button.disabled = false; }
  });
  $('entity-form').addEventListener('submit', async event => {
    event.preventDefault();
    const payload = {
      canonical_text:$('entity-canonical').value.trim(), entity_type:$('entity-type').value,
      domain:$('entity-domain').value, normalization_policy:$('entity-policy').value,
      priority:Number($('entity-priority').value), aliases:$('entity-aliases').value.split('\n').map(v => v.trim()).filter(Boolean),
    };
    if (!payload.canonical_text || !Number.isInteger(payload.priority)) { $('entity-form-error').textContent = '请填写标准名称和整数优先级。'; return; }
    const form = $('entity-form'), id = form.dataset.entityId;
    const submit = form.querySelector('[type="submit"]'); submit.disabled = true;
    try {
      const saved = await api(id ? `/api/entities/${id}` : '/api/entities', {method:id ? 'PUT':'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      if (saved.enabled !== $('entity-enabled').checked) await api(`/api/entities/${saved.entity_id}/enabled`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({enabled:$('entity-enabled').checked})});
      $('entity-dialog').close(); notice('术语已保存，后续任务将使用更新后的词库。'); await load();
    } catch (error) { $('entity-form-error').textContent = error.message; }
    finally { submit.disabled = false; }
  });
  $('translation-audio-file').addEventListener('change', () => {
    const file = $('translation-audio-file').files?.[0];
    document.querySelector('.file-name').textContent = file ? file.name : '选择本地音频…';
  });
})();
