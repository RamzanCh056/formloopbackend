// Shared helpers for the customizable GIF library (Region -> Equipment ->
// GIFs). Used by both dashboard.html (assign at save time) and gifs.html
// (browse/filter, manage categories, edit-after-save). Talks only to the
// Phase-1 backend endpoints -- no backend behavior lives here.
//
// Category shape from GET /api/v1/library/categories:
//   { regions: [ { id, name, order, equipment: [ {id, name, order}, ... ] }, ... ] }

window.FormLoopLibrary = (function () {
  let cache = null;
  let inflight = null;

  async function fetchCategories(force) {
    if (cache && !force) return cache;
    // BUG (found via reproduction): this used to be `if (inflight) return
    // inflight;` with no `force` check -- so calling fetchCategories(true)
    // right after creating/renaming/deleting a category could still hand
    // back a STALE promise that was already in flight from BEFORE that
    // mutation (e.g. the page's initial load fetch), missing the just-created
    // category entirely. That stale result then got cached as if it were
    // current, so the newly created category silently never became
    // selectable until something else invalidated the cache again.
    if (inflight && !force) return inflight;
    inflight = fetch('/api/v1/library/categories', { credentials: 'same-origin' })
      .then((r) => (r.ok ? r.json() : { regions: [] }))
      .then((d) => {
        cache = d.regions || [];
        inflight = null;
        return cache;
      })
      .catch(() => {
        inflight = null;
        return cache || [];
      });
    return inflight;
  }

  function invalidate() {
    cache = null;
  }

  async function _send(url, method, body) {
    const opts = { method, credentials: 'same-origin' };
    if (body !== undefined) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const r = await fetch(url, opts);
    if (!r.ok) {
      let detail = `${method} ${url} failed (${r.status})`;
      try {
        const j = await r.json();
        if (j && j.detail) detail = j.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return r.json();
  }

  async function createRegion(name) {
    const out = await _send('/api/v1/library/categories', 'POST', { name });
    invalidate();
    return out.region;
  }

  async function renameRegion(regionId, name) {
    await _send(`/api/v1/library/categories/${regionId}`, 'PATCH', { name });
    invalidate();
  }

  async function reorderRegions(orderedIds) {
    await _send('/api/v1/library/categories/reorder', 'POST', { ordered_ids: orderedIds });
    invalidate();
  }

  async function deleteRegion(regionId) {
    const out = await _send(`/api/v1/library/categories/${regionId}`, 'DELETE');
    invalidate();
    return out;
  }

  async function createEquipment(regionId, name) {
    const out = await _send(`/api/v1/library/categories/${regionId}/equipment`, 'POST', { name });
    invalidate();
    return out.equipment;
  }

  async function renameEquipment(regionId, subId, name) {
    await _send(`/api/v1/library/categories/${regionId}/equipment/${subId}`, 'PATCH', { name });
    invalidate();
  }

  async function reorderEquipment(regionId, orderedIds) {
    await _send(`/api/v1/library/categories/${regionId}/equipment/reorder`, 'POST', { ordered_ids: orderedIds });
    invalidate();
  }

  async function deleteEquipment(regionId, subId) {
    const out = await _send(`/api/v1/library/categories/${regionId}/equipment/${subId}`, 'DELETE');
    invalidate();
    return out;
  }

  async function editExport(jobId, fields) {
    return _send(`/api/v1/matte/export/${jobId}`, 'PATCH', fields);
  }

  // Populates a <select> with: blank "Uncategorized" option, one option per
  // region/equipment item, and a trailing "+ New ..." option. `selectedId`
  // (may be '' or null) is preselected if present among the options.
  function populateSelect(selectEl, items, opts) {
    const o = opts || {};
    const blankLabel = o.blankLabel || 'Uncategorized';
    const newLabel = o.newLabel || '+ New…';
    const selectedId = o.selectedId || '';
    selectEl.innerHTML = '';
    const blankOpt = document.createElement('option');
    blankOpt.value = '';
    blankOpt.textContent = blankLabel;
    selectEl.appendChild(blankOpt);
    (items || []).forEach((it) => {
      const opt = document.createElement('option');
      opt.value = it.id;
      opt.textContent = it.name;
      selectEl.appendChild(opt);
    });
    const newOpt = document.createElement('option');
    newOpt.value = '__new__';
    newOpt.textContent = newLabel;
    selectEl.appendChild(newOpt);
    selectEl.value = selectedId && [...selectEl.options].some((o2) => o2.value === selectedId) ? selectedId : '';
  }

  return {
    fetchCategories,
    invalidate,
    createRegion,
    renameRegion,
    reorderRegions,
    deleteRegion,
    createEquipment,
    renameEquipment,
    reorderEquipment,
    deleteEquipment,
    editExport,
    populateSelect,
  };
})();
