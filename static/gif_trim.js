// Over-50MB GIF dialog (Google Slides & Docs limit).
//
// Offers a trimmed copy that fits, WITHOUT quality loss: the server uses
// gifsicle frame selection, which only drops whole frames. The full-quality
// original is never modified; the trimmed GIF is an extra downloadable file.
//
// Usage: window.FormloopGifTrim.open({ jobId, gifUrl, sizeBytes })
(function () {
  // Decimal 50 MB (stricter than 50 MiB) so "fits" holds under either reading
  // of Google's limit. Must match gif_trim.SLIDES_SIZE_LIMIT_BYTES.
  const LIMIT = 50000000;
  const MB = 1000 * 1000;
  const fmtMB = (b, digits = 1) => `${(b / MB).toFixed(digits)} MB`;
  const fmtSec = (s) => `${s.toFixed(1)}s`;

  let root = null;

  const close = () => {
    if (root) root.remove();
    root = null;
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (e) => { if (e.key === 'Escape') close(); };

  const btn = (label, kind) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.className = kind === 'primary'
      ? 'rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2.5 text-sm font-semibold text-white transition disabled:opacity-50'
      : kind === 'ghost'
        ? 'rounded-xl px-4 py-2.5 text-sm font-semibold text-zinc-400 hover:text-white transition'
        : 'rounded-xl border border-white/15 bg-white/[0.04] hover:bg-white/[0.08] px-4 py-2.5 text-sm font-semibold text-zinc-100 transition disabled:opacity-50';
    return b;
  };

  const shell = () => {
    close();
    root = document.createElement('div');
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    root.setAttribute('data-gif-trim-dialog', '');
    root.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4';
    root.innerHTML = '<div class="w-full max-w-lg rounded-2xl border border-white/10 bg-zinc-950 p-5 sm:p-6 text-zinc-200 shadow-2xl" data-body></div>';
    root.addEventListener('click', (e) => { if (e.target === root) close(); });
    document.addEventListener('keydown', onKey);
    document.body.appendChild(root);
    return root.querySelector('[data-body]');
  };

  const api = async (url, opts) => {
    const r = await fetch(url, { credentials: 'same-origin', ...opts });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body?.detail ? String(body.detail) : `Request failed (${r.status})`);
    return body;
  };

  const showError = (body, ctx, msg) => {
    body.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'text-sm text-rose-300';
    p.textContent = `Couldn't trim the GIF: ${msg}`;
    const row = document.createElement('div');
    row.className = 'mt-5 flex justify-end gap-2';
    const back = btn('Back', 'secondary');
    back.onclick = () => renderPrompt(ctx);
    const done = btn('Close', 'ghost');
    done.onclick = close;
    row.append(done, back);
    body.append(p, row);
  };

  const renderResult = (ctx, res) => {
    const body = shell();
    const fits = !res.oversize_for_slides;
    body.innerHTML = `
      <h2 class="text-base font-semibold text-white">${fits ? 'Your trimmed GIF fits' : 'Still over 50 MB'}</h2>
      <p class="mt-2 text-sm text-zinc-400">
        <span class="font-semibold text-zinc-100">${fmtMB(res.original_size_bytes)}</span> →
        <span class="font-semibold ${fits ? 'text-emerald-300' : 'text-rose-300'}">${fmtMB(res.size_bytes)}</span>
        · kept frames ${res.start + 1}–${res.end + 1} of ${res.frame_count} (${res.frames_kept} frames).
      </p>
      <p class="mt-1 text-xs text-zinc-500">Every kept frame is untouched — frames were removed, nothing was recompressed. Your full-quality original is unchanged.</p>
      <p class="mt-1 text-xs text-zinc-500">Once saved, your library's “Copy GIF URL” link uses this trimmed version.</p>
      <div class="mt-4 overflow-hidden rounded-xl border border-white/10 bg-black/40">
        <img alt="Trimmed GIF preview" class="mx-auto max-h-64 object-contain" />
      </div>
      <div class="mt-5 flex flex-wrap justify-end gap-2" data-actions></div>`;
    body.querySelector('img').src = res.gif_url;
    const actions = body.querySelector('[data-actions]');
    const again = btn('Trim it myself', 'secondary');
    again.onclick = () => renderManual(ctx);
    const dl = document.createElement('a');
    dl.href = res.gif_url;
    dl.download = res.filename || 'matte_trimmed.gif';
    dl.textContent = `Download trimmed GIF (${fmtMB(res.size_bytes)})`;
    dl.className = 'rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2.5 text-sm font-semibold text-white transition';
    const done = btn('Done', 'ghost');
    done.onclick = close;
    actions.append(done, again, dl);
  };

  const runTrim = async (ctx, payload, workingLabel) => {
    const body = shell();
    body.innerHTML = `<p class="text-sm text-zinc-300 flex items-center gap-2">
      <span class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-indigo-400 border-t-transparent"></span>
      ${workingLabel}</p>`;
    try {
      const res = await api(`/api/v1/matte/trim/${ctx.jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, gif_url: ctx.gifUrl || '' }),
      });
      renderResult(ctx, res);
    } catch (e) {
      showError(body, ctx, e.message);
    }
  };

  const renderManual = async (ctx) => {
    let body = shell();
    body.innerHTML = '<p class="text-sm text-zinc-300">Reading frames…</p>';
    let info;
    try {
      info = ctx.info || (ctx.info = await api(`/api/v1/matte/gif-info/${ctx.jobId}?gif_url=${encodeURIComponent(ctx.gifUrl || '')}`));
    } catch (e) {
      showError(body, ctx, e.message);
      return;
    }
    const total = info.frame_count;
    const size = info.size_bytes;
    // Cumulative start time of each frame, for human-readable handle labels.
    const t = [0];
    info.frame_delays.forEach((d, i) => { t[i + 1] = t[i] + (d || 0); });
    const duration = t[total];

    body = shell();
    body.innerHTML = `
      <h2 class="text-base font-semibold text-white">Trim it yourself</h2>
      <p class="mt-1 text-xs text-zinc-500">Drag the start and end handles. Frames outside the range are removed; kept frames stay full quality.</p>
      <div class="mt-5 space-y-4">
        <label class="block">
          <span class="flex justify-between text-xs text-zinc-400"><span>Start</span><span data-start-label></span></span>
          <input type="range" min="0" max="${total - 1}" step="1" value="0" data-start class="mt-1 w-full accent-indigo-500" />
        </label>
        <label class="block">
          <span class="flex justify-between text-xs text-zinc-400"><span>End</span><span data-end-label></span></span>
          <input type="range" min="0" max="${total - 1}" step="1" value="${total - 1}" data-end class="mt-1 w-full accent-indigo-500" />
        </label>
      </div>
      <div class="mt-5 rounded-xl border border-white/10 bg-white/[0.03] p-3">
        <div class="h-2 w-full overflow-hidden rounded-full bg-white/10">
          <div class="h-full rounded-full transition-all" data-bar></div>
        </div>
        <p class="mt-2 text-sm"><span class="text-zinc-400">Estimated size:</span>
          <span class="font-semibold" data-estimate></span>
          <span class="text-zinc-500" data-frames></span></p>
      </div>
      <div class="mt-5 flex justify-end gap-2" data-actions></div>`;
    const start = body.querySelector('[data-start]');
    const end = body.querySelector('[data-end]');
    const est = body.querySelector('[data-estimate]');
    const frames = body.querySelector('[data-frames]');
    const bar = body.querySelector('[data-bar]');
    const save = btn('Save trimmed GIF', 'primary');
    const back = btn('Back', 'ghost');
    back.onclick = () => renderPrompt(ctx);
    body.querySelector('[data-actions]').append(back, save);

    const update = (moved) => {
      let a = Number(start.value);
      let b = Number(end.value);
      if (a > b) {
        if (moved === start) { end.value = a; b = a; } else { start.value = b; a = b; }
      }
      const kept = b - a + 1;
      const estimate = size * (kept / total);
      const ok = estimate <= LIMIT;
      body.querySelector('[data-start-label]').textContent = `frame ${a + 1} · ${fmtSec(t[a])}`;
      body.querySelector('[data-end-label]').textContent = `frame ${b + 1} · ${fmtSec(t[b + 1])}`;
      est.textContent = `~${fmtMB(estimate)}`;
      est.className = `font-semibold ${ok ? 'text-emerald-300' : 'text-rose-300'}`;
      frames.textContent = ` · ${kept} of ${total} frames · ${fmtSec(t[b + 1] - t[a])} of ${fmtSec(duration)}`;
      bar.style.width = `${Math.min(100, (estimate / size) * 100)}%`;
      bar.className = `h-full rounded-full transition-all ${ok ? 'bg-emerald-400' : 'bg-rose-400'}`;
      save.disabled = !ok;
      save.title = ok ? '' : 'Estimated size is still over 50 MB';
    };
    start.addEventListener('input', () => update(start));
    end.addEventListener('input', () => update(end));
    update(null);
    save.onclick = () => runTrim(
      ctx,
      { mode: 'range', start: Number(start.value), end: Number(end.value) },
      'Cutting the selected frames…',
    );
  };

  const renderPrompt = (ctx) => {
    const body = shell();
    body.innerHTML = `
      <h2 class="text-base font-semibold text-white">GIF too large for Slides &amp; Docs</h2>
      <p class="mt-2 text-sm text-zinc-300" data-msg></p>
      <div class="mt-5 flex flex-col-reverse sm:flex-row sm:justify-end gap-2" data-actions></div>`;
    body.querySelector('[data-msg]').textContent =
      `This GIF is ${Math.round(ctx.sizeBytes / MB)} MB — over the 50 MB limit for Google Slides & Docs. ` +
      'Your full-quality GIF is saved. Want a version that fits?';
    const later = btn('Not now', 'ghost');
    later.onclick = close;
    const manual = btn('Trim it myself', 'secondary');
    manual.onclick = () => renderManual(ctx);
    const auto = btn('Auto-fit under 50 MB', 'primary');
    auto.onclick = () => runTrim(ctx, { mode: 'auto' }, 'Trimming frames from the end until it fits…');
    body.querySelector('[data-actions]').append(later, manual, auto);
  };

  window.FormloopGifTrim = {
    LIMIT,
    open(opts) {
      if (!opts || !opts.jobId || !opts.sizeBytes) return;
      renderPrompt({ jobId: opts.jobId, gifUrl: opts.gifUrl || '', sizeBytes: Number(opts.sizeBytes) });
    },
    close,
  };
})();
