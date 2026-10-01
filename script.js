(function () {
  "use strict";
  const AV = "?v=20261001a";   // cache-buster for media replaced on 2026-10-01 (selfhist 4-step results)

  function rowOf(el) { return el.closest(".grow"); }
  function videosOf(row) { return [...row.querySelectorAll(".gcell video")]; }

  function setPlayIcon(row, playing) {
    const b = row.querySelector(".rbtns .play");
    if (b) { b.textContent = playing ? "❚❚" : "▶"; b.classList.toggle("active", playing); }
  }

  function clearMarks(row) {
    row.querySelectorAll(".rbtns button:not(.play)").forEach((b) => b.classList.remove("active"));
  }

  // Resolves with the video once it can seek, or null on error / after 3s,
  // so one stuck clip never holds up the rest of the row.
  function whenReady(v) {
    if (v.readyState >= 1) return Promise.resolve(v);
    return new Promise((res) => {
      v.addEventListener("loadedmetadata", () => res(v), { once: true });
      v.addEventListener("error", () => res(null), { once: true });
      setTimeout(() => res(v.readyState >= 1 ? v : null), 3000);
    });
  }

  function makeVideo(cell, src, poster) {
    const v = document.createElement("video");
    v.muted = true;
    v.loop = false;
    // the row stops once every clip in it has reached its end
    v.addEventListener("ended", () => {
      const row = rowOf(v);
      if (row && videosOf(row).every((x) => x.paused)) { row.dataset.playing = "0"; setPlayIcon(row, false); }
    });
    v.playsInline = true;
    v.preload = "metadata";
    if (poster) v.poster = poster;
    v.src = src;
    v.addEventListener("click", () => {
      const row = rowOf(v);
      if (!row) { v.paused ? v.play() : v.pause(); return; }
      rowPlaying(row) ? pauseRow(row) : playRow(row, v.currentTime);
    });
    cell.firstElementChild ? cell.firstElementChild.replaceWith(v) : cell.appendChild(v);
    return v;
  }

  // Make sure a cell holds a working clip: upgrade a still, or reload a clip
  // that failed (e.g. the file was still being written when the page loaded).
  function ensureVideo(cell) {
    const cur = cell.firstElementChild;
    if (!cur) return Promise.resolve(null);
    if (cur.tagName === "VIDEO" && !cur.error) return whenReady(cur);
    const src = (cur.getAttribute("data-video") || cur.dataset.src || "").replace(/[?&]t=\d+/, "");   // keep ?v= cache-busters
    if (!src) return Promise.resolve(null);
    const poster = cur.tagName === "IMG" ? (cur.currentSrc || cur.src) : (cur.poster || "");
    return fetch(src, { method: "HEAD", cache: "no-store" })
      .then((res) => {
        if (!res.ok) return null;
        const bust = cur.tagName === "VIDEO" ? `${src.includes("?") ? "&" : "?"}t=${Date.now()}` : "";
        const v = makeVideo(cell, src + bust, poster);
        v.dataset.src = src;
        return whenReady(v);
      })
      .catch(() => null);
  }

  function rowPlaying(row) { return row.dataset.playing === "1"; }

  // play() is called synchronously inside the click so browsers that only
  // allow playback from a user gesture (Safari, strict autoplay settings) accept it.
  // Start every clip of a row together, only once all of them can play:
  // starting whichever is ready first made the late ones drag the row back.
  // Each clip is started once inside the click (so strict autoplay rules
  // accept later play() calls), held, and released when the row is ready.
  function playRow(row, t) {
    clearMarks(row);
    endStop(row);
    row.dataset.playing = "1";
    const token = (row._playToken = (row._playToken || 0) + 1);
    const btn = row.querySelector(".rbtns .play");
    if (btn) { btn.textContent = "…"; btn.classList.add("active"); }
    unparkRow(row);
    const now = videosOf(row).filter((v) => !v.error);
    let start = t != null ? t : (now[0] ? now[0].currentTime : 0);
    if (now.some((v) => v.ended || (v.duration && start >= v.duration - 0.25))) start = 0;
    now.forEach((v) => {
      v.preload = "auto";
      const p = v.play();
      if (p) p.then(() => { if (row._playToken === token && !row._released) v.pause(); }).catch(() => {});
    });
    row._released = false;
    // wait until the browser expects to play through AND ~6 s are buffered
    const canPlay = (v) => new Promise((res) => {
      const t0 = Date.now();
      const check = () => {
        if (v.error) return res(null);
        const need = Math.min(6, (v.duration || 60) - start - 0.1);
        if ((v.readyState >= 4 && ahead(v, start) >= need) || Date.now() - t0 > 15000) return res(v);
        setTimeout(check, 250);
      };
      check();
    });
    const cells = [...row.querySelectorAll(".gcell")];
    Promise.all(cells.map(ensureVideo))
      .then((vs) => Promise.all(vs.filter(Boolean).map((v) => { v.preload = "auto"; return canPlay(v); })))
      .then((vs) => {
        if (row._playToken !== token || !rowPlaying(row)) return;
        vs = vs.filter(Boolean);
        row._released = true;
        row._holding = false;
        vs.forEach((v) => { v.currentTime = start; v.play().catch(() => {}); });
        setPlayIcon(row, true);
      });
  }

  function pauseRow(row) {
    endStop(row);
    row.dataset.playing = "0";
    row._playToken = (row._playToken || 0) + 1;   // cancels a start still waiting for data
    videosOf(row).forEach((v) => v.pause());
    setPlayIcon(row, false);
  }

  // seconds buffered ahead of time t (or of the current time)
  function ahead(v, t) {
    t = t == null ? v.currentTime : t;
    for (let i = 0; i < v.buffered.length; i++)
      if (v.buffered.start(i) <= t + 0.05 && v.buffered.end(i) >= t) return v.buffered.end(i) - t;
    return 0;
  }

  // Keep a released row in step. If a clip is about to run dry the whole row
  // holds, and resumes only once every clip has ~3 s in hand (or reached its
  // end) -- hysteresis, so a marginal connection gives a few longer pauses
  // instead of constant stop-start. A clip starved for over 6 s is dropped
  // from the wait. Every clip was started once inside the click, so resuming
  // here is allowed under strict autoplay rules.
  setInterval(() => {
    const now = Date.now();
    document.querySelectorAll(".grow[data-playing='1']").forEach((row) => {
      if (!row._released || row._stopped) return;
      const vs = videosOf(row).filter((v) => !v.error && !v.ended && v.getAttribute("src"));
      const live = vs.filter((v) => {
        const left = (v.duration || 0) - v.currentTime;
        if (v.readyState >= 3 || ahead(v) > 0.5 || left < 0.3) { v._starvedSince = 0; return true; }
        v._starvedSince = v._starvedSince || now;
        return now - v._starvedSince < 6000;
      });
      if (!live.length) return;
      const dry = (v) => ahead(v) < 0.3 && (v.duration || 0) - v.currentTime > 0.3;
      const ready = (v) => ahead(v) >= Math.min(3, (v.duration || 0) - v.currentTime - 0.1);
      if (!row._holding && live.some(dry)) row._holding = true;
      if (row._holding) {
        if (live.every(ready)) row._holding = false;
        else { live.forEach((v) => { if (!v.paused) v.pause(); }); return; }
      }
      // Keep clips together by holding the ones that ran ahead (pause, no
      // seek) until the slowest catches up. Seeking them back instead makes
      // them re-buffer and decode from a keyframe, which on a slow machine
      // leaves the whole row stuck re-seeking; only a large gap is seeked.
      if (live.some((v) => v.seeking)) return;
      const t = Math.min(...live.map((v) => v.currentTime));
      live.forEach((v) => {
        const lead = v.currentTime - t;
        if (lead > 3) { v.currentTime = t; return; }
        if (lead > 0.25) { if (!v.paused) v.pause(); v._ahead = true; return; }
        if (v._ahead && lead > 0.05) return;
        v._ahead = false;
        if (v.paused) v.play().catch(() => {});
      });
    });
  }, 250);

  // Revisit stops: when a playing row reaches a moment where the camera is
  // back somewhere it has been, every clip is held on that exact frame for
  // STOP_MS so the methods can be compared, then the row plays on by itself.
  const STOP_MS = 6500;
  function checkStops(row) {
    if (!row._stops || !row._stops.length || !row._released || row._stopped || !rowPlaying(row)) return;
    const t = rowTime(row);
    row._stops.forEach((st) => { if (t < st.t - 1) st.done = false; });   // rewound: stop again next time
    const st = row._stops.find((x) => !x.done && t >= x.t - 0.08 && t < x.t + 1);
    if (!st) return;
    st.done = true;
    row._stopped = true;
    const vs = videosOf(row).filter((v) => !v.error && v.getAttribute("src"));
    vs.forEach((v) => { v.pause(); v.currentTime = st.t; });
    placeStopBox(row);
    row.classList.add("at-stop"); row.classList.toggle("at-start", !!st.start);
    row.querySelectorAll(".stopnote").forEach((n) => { n.textContent = st.label; });
    // 3 x 3 rows, back again: hold on the first-pass / back-again stills until the viewer plays on
    if (row.classList.contains("cell-stops") && st.kind === "return" && showCompare(row)) return;
    row._stopTimer = setTimeout(() => {
      endStop(row);
      if (rowPlaying(row)) vs.forEach((v) => v.play().catch(() => {}));
    }, STOP_MS);
  }
  // one frame around all the method clips (not the input or the path)
  function placeStopBox(row) {
    let box = row.querySelector(".stopbox");
    if (!box) { box = document.createElement("div"); box.className = "stopbox"; row.appendChild(box); }
    const items = [...row.querySelectorAll(".gitem")];
    if (!items.length) return;
    const pad = 7;
    const l = Math.min(...items.map((e) => e.offsetLeft)), t = Math.min(...items.map((e) => e.offsetTop));
    const r = Math.max(...items.map((e) => e.offsetLeft + e.offsetWidth)), b = Math.max(...items.map((e) => e.offsetTop + e.offsetHeight));
    const right = Math.min(r + pad, row.clientWidth);   // the scroller would clip anything past the row
    Object.assign(box.style, { left: `${l - pad}px`, top: `${t - pad}px`, width: `${right - (l - pad)}px`, height: `${b - t + 2 * pad}px` });
  }
  // 3 x 3 rows, compare mode: every clip's first-pass and back-again stills (pre-extracted jpgs, no
  // video needed) with a divider; one slider moves every divider. Play leaves it (playRow -> endStop).
  function compareNow(row) {
    if (row.classList.contains("comparing")) return;
    pauseRow(row);
    const r = (row._stops || []).find((x) => x.kind === "return");
    if (!r || !showCompare(row)) return;
    row._stopped = true;
    row.classList.add("at-stop");
    row.querySelectorAll(".stopnote").forEach((n) => { n.textContent = r.label; });
  }
  function showCompare(row) {
    if (row.classList.contains("comparing")) return true;
    const f = (row._stops || []).find((x) => x.kind === "first"), r = (row._stops || []).find((x) => x.kind === "return");
    const cells = [...row.querySelectorAll(".gcell")].filter((c) => c.dataset.still);
    if (!f || !r || !cells.length) return false;
    const setX = (x) => { x = Math.max(0, Math.min(100, x)); row.style.setProperty("--fp", `${x}%`); range.value = x; };
    cells.forEach((cell) => {
      const o = document.createElement("div");
      o.className = "fpcmp";
      o.innerHTML = `<img class="fp-back" src="${cell.dataset.still}_back.jpg${AV}" alt=""><img class="fp-first" src="${cell.dataset.still}_first.jpg${AV}" alt="">`
        + '<i class="fp-line"></i><b class="fp-l">first pass</b><b class="fp-r">back again</b>';
      cell.appendChild(o);
      o.addEventListener("pointerdown", (e) => {
        const go = (ev) => { const b = o.getBoundingClientRect(); setX(100 * (ev.clientX - b.left) / b.width); };
        go(e); o.setPointerCapture(e.pointerId);
        o.onpointermove = go; o.onpointerup = () => { o.onpointermove = null; };
        e.stopPropagation();
      });
    });
    const bar = document.createElement("div");
    bar.className = "fpbar";
    bar.innerHTML = `<span>first pass \u00b7 ${f.t.toFixed(1)} s</span><input type="range" min="0" max="100" step="0.5" aria-label="first pass / back again">`
      + `<span>back again \u00b7 ${r.t.toFixed(1)} s</span>`;
    const range = bar.querySelector("input");
    range.addEventListener("input", () => setX(+range.value));
    row.appendChild(bar);
    setX(50);
    row.classList.add("comparing");
    return true;
  }
  function endStop(row) {
    row.querySelectorAll(".fpcmp, .fpbar").forEach((b) => b.remove());
    row.classList.remove("comparing");
    clearTimeout(row._stopTimer);
    row._stopped = false;
    row.classList.remove("at-stop", "at-start");
  }

  // Snap every clip in the row to the slowest one's time; play state is kept.
  function syncRow(row) {
    const vs = videosOf(row).filter((v) => v.readyState >= 1 && !v.error);
    if (vs.length < 2) return;
    const t = Math.min(...vs.map((v) => v.currentTime));
    vs.forEach((v) => { v.currentTime = t; });
    if (rowPlaying(row)) vs.forEach((v) => v.play().catch(() => {}));
  }

  function freezeRow(row, t) {
    pauseRow(row);
    const cells = [...row.querySelectorAll(".gcell")];
    Promise.all(cells.map(ensureVideo)).then((vs) =>
      vs.filter(Boolean).forEach((v) => { v.pause(); v.currentTime = t; }));
  }

  // ------------------------------------------------------------------
  // Page content: every section is a carousel of the same six web walks.
  // ------------------------------------------------------------------
  const pad2 = (k) => "c" + String(k).padStart(2, "0");

  // 3 x 3 with ours in the centre: [input | CaR | HY-World] [camera path | ours | LingBot] [DreamX | Matrix-Game | Infinite-World]
  const SOTA_COLS = [
    ["car", "CaR"], ["hyworld", "HY-World 1.5"],
    ["ours", "Memorizon (4-step)", true], ["lingbot", "LingBot-World-v2"],
    ["dreamx", "DreamX-World 1.0"], ["matrix", "Matrix-Game 3.0"], ["infworld", "Infinite-World"],
  ];

  // Ablation clips go to assets/videos/ablation/<case>_<key>.mp4 and appear
  // on their own once uploaded. The 100 s per-chunk run *is* the Memorizon
  // column above, so it reuses those clips.
  const ABLATIONS = {
    abl1: [
      { name: "10 s · no retrieval", video: (c) => `assets/videos/ablation/${c}_a1-sw.mp4` },
      { name: "10 s · per-chunk retrieval", ours: true, video: (c) => `assets/videos/ablation/${c}_a2-v3.mp4` },
    ],
    abl2: [
      { name: "100 s · per-segment retrieval", video: (c) => `assets/videos/ablation/${c}_shared6.mp4` },
      { name: "100 s · per-chunk retrieval", ours: true, video: (c) => `assets/videos/v5/${c}_ours.mp4`,
        poster: (c) => `assets/img/v5/${c}_ours.jpg` },
    ],
  };

  const HIDE = { sota: [12] };   // cases left out of a section (by web case id)

  // Where a row holds: the first sighting of a mid-path spot (so it can be
  // remembered), the return to it, and the return to the starting view
  // (compared with the input photo).
  function stopsFor(r) {
    const f = (t) => `${t.toFixed(1)} s`;
    return (r.times || []).map((t, i) => [t, (r.labels || [])[i] || ""]).map(([t, l]) =>
      /start/i.test(l) && /return/i.test(l) ? { t, start: true, label: `back at the start \u00b7 ${f(t)}` }
      : /first sighting/i.test(l) ? { t, kind: "first", label: `first pass \u00b7 ${f(t)} \u00b7 remember this view` }
      : /return/i.test(l) ? { t, kind: "return", label: `back again \u00b7 ${f(t)}` } : null).filter(Boolean).sort((a, b) => a.t - b.t);
  }

  function rowsFor(kind, list, set) {
    set = set || "v5";
    return list.filter((r) => !(set === "v5" && (HIDE[kind] || []).includes(r.case))).map((r) => {
      const c = pad2(r.case);
      // the "input view (t = 0)" tag is the input photo itself, not a moment in the videos
      const keep = (r.times || []).map((t, i) => [t, (r.labels || [])[i] || ""])
        .filter(([t, l]) => !(t === 0 && /input view/i.test(l)));
      const base = { input: `assets/img/${set}/${c}_input.jpg`, stops: set === "v7" ? stopsFor(r) : [],
                     traj: set === "v5" ? `assets/traj/${c}.json` : `assets/traj/${set}_${c}.json`,
                     times: keep.map((x) => x[0]), labels: keep.map((x) => x[1]) };
      if (kind === "sota") {
        base.layout = "lay-grid";
        base.label = r.label;
        base.cols = 3;   // input and camera path are the first two cells of the grid
        base.cells = SOTA_COLS.map(([m, name, ours]) => ({
          name, ours, poster: `assets/img/${set}/${c}_${m}.jpg${ours ? AV : ""}`, video: `assets/videos/${set}/${c}_${m}.mp4${ours ? AV : ""}`,
          still: `assets/img/${set}cmp/${c}_${m}` }));
      } else {
        base.layout = "lay-row";
        base.cells = ABLATIONS[kind].map((a) => ({
          name: a.name, ours: a.ours, video: a.video(c), poster: a.poster ? a.poster(c) : null }));
      }
      return base;
    });
  }

  function button(cls, text, title) {
    const b = document.createElement("button");
    if (cls) b.className = cls;
    b.textContent = text;
    if (title) b.title = title;
    return b;
  }

  function tag(text, ours) {
    const d = document.createElement("div");
    d.className = "gtag" + (ours ? " ours" : "");
    d.textContent = text;
    return d;
  }
  function caption(text, ours) {
    const d = document.createElement("div");
    d.className = "cap" + (ours ? " ours" : "");
    d.textContent = text;
    return d;
  }

  // WASD = translation, arrows = rotation, lit from the trajectory at the current time
  const KEYS = [["W", 1, 2], ["A", 2, 1], ["S", 2, 2], ["D", 2, 3]];
  // arrows are drawn, not typed: font arrow glyphs sit off-centre in the keycap
  const ARROWS = [["u", 1, 2, 0], ["l", 2, 1, -90], ["d", 2, 2, 180], ["r", 2, 3, 90]];
  const arrowSvg = (deg) =>
    `<svg viewBox="0 0 10 10" width="9" height="9" style="transform:rotate(${deg}deg)">` +
    `<path d="M5 1.2 L5 8.8 M1.8 4.4 L5 1.2 L8.2 4.4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  function keyOverlay() {
    const o = document.createElement("div");
    o.className = "keys";
    [KEYS, ARROWS].forEach((set) => {
      const g = document.createElement("div");
      g.className = "cluster";
      set.forEach(([k, row, col, label]) => {
        const e = document.createElement("span");
        e.className = "k";
        e.dataset.k = k;
        if (typeof label === "number") e.innerHTML = arrowSvg(label); else e.textContent = k;
        e.style.gridArea = `${row} / ${col}`;
        g.appendChild(e);
      });
      o.appendChild(g);
    });
    return o;
  }

  function updateKeys(row) {
    const tr = row._traj;
    if (!tr || !tr.keys) return;
    const i = Math.max(0, Math.min(tr.keys.length - 1, Math.floor(rowTime(row) / tr.dt)));
    if (row._keysAt === i) return;
    row._keysAt = i;
    const on = tr.keys[i];
    row.querySelectorAll(".keys .k").forEach((e) => e.classList.toggle("on", on.includes(e.dataset.k)));
  }

  function buildRow(r) {
    const row = document.createElement("div");
    row.className = "grow " + r.layout;
    // input + path in column 1, the method clips in two rows beside it
    if (r.layout === "lay-grid") row.style.gridTemplateColumns = `repeat(${r.cols || 1 + Math.ceil(r.cells.length / 2)}, minmax(0, 1fr))`;
    // one row: input | every clip | path
    if (r.layout === "lay-row") row.style.gridTemplateColumns = `repeat(${2 + r.cells.filter((c) => !c.row2).length}, minmax(0, 1fr))`;
    const label = document.createElement("div");
    label.className = "rowlabel";
    const inp = document.createElement("img");
    inp.dataset.src = r.input;
    inp.alt = "Input photograph";

    const rbtns = document.createElement("div");
    rbtns.className = "rbtns";
    rbtns.style.gridTemplateColumns = "repeat(5, 1fr)";   // same button size as before, left-aligned
    rbtns.append(button("play", "▶", "Play / pause this row"),
                 button("sync", "Sync", "Align all videos to the same moment"));
    const map = document.createElement("div");
    map.className = "trajitem";
    map.append(document.createElement("canvas"), caption("Camera path · top view"));
    if (r.cols) {   // input and camera path carry the same top-left tags as the clips
      const wrap = document.createElement("div");
      wrap.className = "tagwrap";
      wrap.append(inp, tag("Input"));
      label.append(rbtns, wrap, caption("Input"));
      map.appendChild(tag("Camera path"));
    } else {
      label.append(rbtns, inp, caption("Input"));
    }
    row.appendChild(label);
    if (r.layout === "lay-row") map.style.gridColumn = String(2 + r.cells.length);
    if (r.cols) {   // camera path under the input; at a stop each clip gets its own frame
      map.style.gridColumn = "1"; map.style.gridRow = "2";
      row.classList.add("cell-stops");
    }
    if (r.mapFirst) {   // path | input | clips, with the row buttons over the path
      map.style.gridColumn = "1"; map.style.gridRow = "1"; map.style.position = "relative";
      label.style.gridColumn = "2";
      map.appendChild(rbtns);
    }
    row.appendChild(map);
    row._trajUrl = r.traj;
    row._times = [];   // no moment markers: the page compares videos only
    row._stops = (r.stops || []).map((x) => ({ ...x, done: false }));
    if (row._stops.length) {
      row.style.setProperty("--stop-ms", STOP_MS + "ms");
      const tag = document.createElement("div"); tag.className = "stoptag"; tag.textContent = "compare with this view";
      label.appendChild(tag);
      const note = document.createElement("div"); note.className = "stopnote"; row.appendChild(note);   // one label per row
    }

    r.cells.forEach((c) => {
      const item = document.createElement("div");
      item.className = "gitem" + (c.ours ? " ours" : "");
      const cell = document.createElement("div");
      cell.className = "gcell";
      let media;
      if (c.poster) {
        media = document.createElement("img");
        media.onerror = () => { media.onerror = null; media.src = r.input; };
        media.dataset.src = c.poster;
        media.alt = "";
      } else {
        media = document.createElement("div");
        media.className = "pending";
        media.textContent = "Video coming soon";
      }
      media.setAttribute("data-video", c.video);
      cell.appendChild(media);
      cell.appendChild(keyOverlay());
      if (r.cols) cell.appendChild(tag(c.name, c.ours));   // 3 x 3: method name on the clip, top-left
      if (c.still) cell.dataset.still = c.still;
      item.append(cell, caption(c.name, c.ours));
      if (c.row2) { item.style.gridRow = "2"; item.style.gridColumn = "2"; }
      row.appendChild(item);
    });

    // second row: what that clip's bank held, chunk by chunk (frames from another episode)
    if (r.bankStrip) {
      const bs = document.createElement("div");
      bs.className = "bankstrip";
      bs.style.gridRow = "2"; bs.style.gridColumn = "3 / -1";
      bs.innerHTML = `<div class="bs-label">Its bank for the current chunk: <b>K = 6 frames from another episode</b> (${r.bankStrip.donor}) <span class="bs-t"></span></div><div class="bs-thumbs"></div>`;
      const th = bs.querySelector(".bs-thumbs");
      for (let i = 0; i < 6; i++) { const im = document.createElement("img"); im.alt = ""; th.appendChild(im); }
      row.appendChild(bs);
      row._bankStrip = { ...r.bankStrip, el: bs, imgs: [...th.children], tEl: bs.querySelector(".bs-t"), chunk: -2 };
    }

    rbtns.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      if (btn.classList.contains("play")) { rowPlaying(row) ? pauseRow(row) : playRow(row); return; }
      if (btn.classList.contains("sync")) { syncRow(row); return; }
      clearMarks(row);
      btn.classList.add("active");
      freezeRow(row, parseFloat(btn.dataset.t));
    });
    return row;
  }

  // Load a row's pictures, clips and path only when it is (about to be) shown.
  function activate(row) {
    if (!row || row.dataset.loaded) return;
    row.dataset.loaded = "1";
    row.querySelectorAll("img[data-src]").forEach((im) => { im.src = im.dataset.src; });
    row.querySelectorAll(".gcell").forEach((c) => ensureVideo(c));
    fetch(row._trajUrl).then((r) => r.json()).then((t) => { row._traj = t; row._drawnAt = null; }).catch(() => {});
  }

  let focused = null;   // the carousel the arrow keys drive

  // A row that is not on screen should not keep downloading: detach its
  // clips (aborting the transfer) and put them back, at the same moment,
  // when the row is shown again.
  function parkRow(row) {
    videosOf(row).forEach((v) => {
      const src = v.getAttribute("src");
      if (!src) return;
      v.pause();
      v.dataset.parkedAt = v.currentTime;
      v.dataset.parkedSrc = src;
      v.removeAttribute("src"); v.load();
    });
  }
  function unparkRow(row) {
    videosOf(row).forEach((v) => {
      if (!v.dataset.parkedSrc) return;
      const t = parseFloat(v.dataset.parkedAt || "0");
      v.src = v.dataset.parkedSrc; delete v.dataset.parkedSrc;
      if (t > 0) v.addEventListener("loadedmetadata", () => { v.currentTime = t; }, { once: true });
    });
  }

  function carousel(root, rows) {
    const table = root.querySelector(".gtable");
    const count = root.querySelector(".gnav .count");
    rows.forEach((r) => table.appendChild(buildRow(r)));
    const els = [...table.querySelectorAll(".grow")];
    let cur = 0;
    // Prev/Next/swipe start the new example playing (from the gesture, so autoplay rules allow it).
    const show = (i, autoplay, dir) => {
      cur = (i + els.length) % els.length;
      els.forEach((g, k) => {
        if (k !== cur) { if (rowPlaying(g)) pauseRow(g); parkRow(g); }
        g.classList.toggle("active", k === cur);
      });
      const row = els[cur];
      row.classList.remove("enter-next", "enter-prev");
      if (dir) { void row.offsetWidth; row.classList.add(dir > 0 ? "enter-next" : "enter-prev"); }
      count.textContent = `${cur + 1} / ${els.length}`;
      if (c && c.picks) c.picks.forEach((p, k) => p.classList.toggle("active", k === cur));
      activate(row);
      unparkRow(row);
      if (autoplay) playRow(row, 0);
      // warm the next example: posters plus clip headers only (preload=metadata)
      const nxt = els[(cur + 1) % els.length];
      setTimeout(() => { if (!nxt.classList.contains("active")) activate(nxt); }, 1500);
    };
    const c = { show, step: (d) => show(cur + d, true, d) };
    if (els.length < 2) root.querySelector(".gnav").style.visibility = "hidden";
    // SOTA: a picker strip of the examples (input photographs) instead of the prev / next arrows
    if (root.dataset.kind === "sota") {
      const picker = document.createElement("div");
      picker.className = "rv-picker";
      const picks = rows.map((r, k) => {
        const p = document.createElement("button");
        p.type = "button"; p.className = "rv-pick"; p.title = r.label || "";
        p.innerHTML = `<img src="${r.input}" alt="" loading="lazy"><span class="rv-pick-label">${r.label || k + 1}</span>`;
        p.addEventListener("click", () => { if (k !== cur) { show(k, false, k > cur ? 1 : -1); compareNow(els[cur]); } });
        picker.appendChild(p);
        return p;
      });
      root.querySelector(".gallery-scroll").after(picker);
      root.classList.add("has-picker");
      c.picks = picks;
      // one fixed toolbar that drives whichever example is shown (the per-row buttons slid with the rows)
      const bar = document.createElement("div");
      bar.className = "sota-ctl";
      const play = button("play", "\u25B6", "Play / pause"), sync = button("sync", "Sync", "Align all videos to the same moment");
      const cmp = button("cmp", "\u21C6 Compare first pass / back again", "Jump to the return and compare it with the first pass");
      bar.append(play, sync, cmp);
      root.querySelector(".gallery-scroll").before(bar);
      play.addEventListener("click", () => els[cur].querySelector(".rbtns .play").click());
      sync.addEventListener("click", () => els[cur].querySelector(".rbtns .sync").click());
      cmp.addEventListener("click", () => compareNow(els[cur]));
      setTimeout(() => compareNow(els[cur]), 0);   // open on the comparison; Play starts the clips
      setInterval(() => {   // mirror the current row's play state
        const b = els[cur].querySelector(".rbtns .play");
        if (b && play.textContent !== b.textContent) play.textContent = b.textContent;
        play.classList.toggle("active", !!b && b.classList.contains("active"));
      }, 250);
    }
    root.querySelector(".gnav .prev").addEventListener("click", () => c.step(-1));
    root.querySelector(".gnav .next").addEventListener("click", () => c.step(1));
    root.addEventListener("pointerdown", () => { focused = c; });

    // Swipe: drag left/right with the mouse or a finger over the clips.
    let sx = null, sy = 0, swiped = false;
    table.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button, input, .fpbar, .fpcmp")) return;
      sx = e.clientX; sy = e.clientY; swiped = false;
    });
    window.addEventListener("pointerup", (e) => {
      if (sx === null) return;
      const dx = e.clientX - sx, dy = e.clientY - sy;
      sx = null;
      if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(dy)) { swiped = true; c.step(dx < 0 ? 1 : -1); }
    });
    // a drag that switched examples must not also count as a click on a video
    table.addEventListener("click", (e) => { if (swiped) { e.stopPropagation(); e.preventDefault(); swiped = false; } }, true);
    table.addEventListener("dragstart", (e) => e.preventDefault());
    // Two-finger horizontal scroll on a trackpad.
    let acc = 0, lock = 0;
    table.addEventListener("wheel", (e) => {
      if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
      e.preventDefault();
      if (Date.now() < lock) return;
      acc += e.deltaX;
      if (Math.abs(acc) > 80) { c.step(acc > 0 ? 1 : -1); acc = 0; lock = Date.now() + 700; }
    }, { passive: false });

    // a carousel scrolled out of view stops downloading too
    new IntersectionObserver(([e]) => {
      const row = els[cur];
      if (e.isIntersecting) unparkRow(row);
      else { if (rowPlaying(row)) pauseRow(row); parkRow(row); }
    }, { threshold: 0.05 }).observe(root);

    show(0, false);
    return c;
  }

  document.addEventListener("keydown", (e) => {
    if (!focused || e.target.closest("input, textarea")) return;
    if (e.key === "ArrowRight") focused.step(1);
    if (e.key === "ArrowLeft") focused.step(-1);
  });

  // Clips can land after the page is open (still uploading): keep probing
  // the loaded cells that are still showing a picture or a placeholder.
  setInterval(() => {
    document.querySelectorAll(".grow[data-loaded] .gcell > :first-child:not(video)").forEach((m) => ensureVideo(m.parentElement));
  }, 15000);

  // ---- top-view camera path, driven by the row's video time ----
  const C = { bg: "#FBFAF6", grid: "#EFEBE1", path: "#D6D0C3", done: "#0E7387", start: "#14181E", mark: "#94241C" };

  // The resync above pulls fast clips back to the slowest one, so the slowest
  // playing clip is the one steady clock; reading any other clip would make
  // the marker jump backwards every time a resync lands.
  function rowTime(row) {
    const vs = videosOf(row).filter((x) => x.readyState >= 1 && !x.error);
    if (!vs.length) return 0;
    const playing = vs.filter((x) => !x.paused);
    return Math.min(...(playing.length ? playing : vs).map((x) => x.currentTime));
  }

  // Shared top-view renderer: light grid, smoothed path coloured by time
  // (pale → deep teal as the clip plays), dashed path still ahead, soft view
  // cone and a haloed camera dot. marks: [{t, label, on}]
  function drawTopView(cv, tr, t, marks, bare) {   // bare: no text labels (small insets)
    const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.fillStyle = bare ? "rgba(251,250,246,0.55)" : C.bg; g.fillRect(0, 0, w, h);

    if (!tr._sx) {                                   // light smoothing, once per trajectory
      const sm = (a) => a.map((_, i) => { let s = 0, c = 0; for (let j = -2; j <= 2; j++) { const v = a[i + j]; if (v !== undefined) { s += v; c++; } } return s / c; });
      tr._sx = sm(tr.x); tr._sz = sm(tr.z);
    }
    const xs = tr._sx, zs = tr._sz, n = xs.length;
    const x0 = Math.min(...xs), x1 = Math.max(...xs), z0 = Math.min(...zs), z1 = Math.max(...zs);
    const L = Math.min(w, h) * (bare ? 0.4 : 0.2), pad = Math.min(w, h) * 0.16 + 6;   // insets: longer cone so the heading reads
    const span = Math.max(x1 - x0, z1 - z0, 0.5);
    const s = Math.min((w - 2 * pad) / Math.max(x1 - x0, span * 0.35), (h - 2 * pad) / Math.max(z1 - z0, span * 0.35));
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    const P = (i) => [w / 2 + (xs[i] - cx) * s, h / 2 - (zs[i] - cz) * s];

    // grid at a round metric step, anchored to the start
    const raw = 42 / s, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw) * s;
    const [ox, oy] = P(0);
    g.strokeStyle = bare ? "rgba(20,24,30,0.08)" : C.grid; g.lineWidth = 1;
    g.beginPath();
    for (let x = ((ox % step) + step) % step; x < w; x += step) { g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, h); }
    for (let y = ((oy % step) + step) % step; y < h; y += step) { g.moveTo(0, Math.round(y) + 0.5); g.lineTo(w, Math.round(y) + 0.5); }
    g.stroke();

    const k = Math.max(0, Math.min(n - 1, t / tr.dt)), ki = Math.floor(k);
    const f = k - ki, i2 = Math.min(n - 1, ki + 1);
    const [ax, ay] = P(ki), [bx, by] = P(i2);
    const px = ax + (bx - ax) * f, py = ay + (by - ay) * f;
    g.lineJoin = g.lineCap = "round";

    // path still ahead
    g.beginPath(); g.strokeStyle = C.path; g.lineWidth = 2;
    g.moveTo(px, py);
    for (let i = ki + 1; i < n; i++) { const [qx, qy] = P(i); g.lineTo(qx, qy); }
    g.stroke();

    // path travelled: white casing, then time-coloured stroke
    const trail = (col, wd) => { g.beginPath(); g.lineWidth = wd; g.strokeStyle = col;
      for (let i = 0; i <= ki; i++) { const [qx, qy] = P(i); i ? g.lineTo(qx, qy) : g.moveTo(qx, qy); } g.lineTo(px, py); g.stroke(); };
    if (k > 0) {
      trail("rgba(255,255,255,0.9)", bare ? 3 : 6);
      // colour follows time, not screen position, so a path that doubles back still reads old → new
      const c0 = [167, 211, 218], c1 = [14, 115, 135], lerp = (u) => `rgb(${c0.map((v, j) => Math.round(v + (c1[j] - v) * u)).join(",")})`;
      g.lineWidth = bare ? 1.5 : 3;
      for (let i = 0; i <= ki; i++) {
        const [qx, qy] = P(i), [rx, ry] = i < ki ? P(i + 1) : [px, py];
        g.strokeStyle = lerp(Math.min(1, (i + 1) / Math.max(k, 1)));
        g.beginPath(); g.moveTo(qx, qy); g.lineTo(rx, ry); g.stroke();
      }
    }

    // start
    const [sx, sy] = P(0);
    g.fillStyle = "#fff"; g.strokeStyle = C.start; g.lineWidth = bare ? 1.2 : 2;
    g.beginPath(); g.arc(sx, sy, bare ? 2.5 : 4.5, 0, 2 * Math.PI); if (!bare) g.fill(); g.stroke();

    // label pills: try four spots around the point, keep the first that clears earlier labels
    const placed = [];
    const pill = (text, x0, y0, col, bold) => {
      g.font = `${bold ? 700 : 600} 11px Arsenal, sans-serif`; g.textBaseline = "middle";
      const tw = g.measureText(text).width + 10, th = 17;
      const cands = [[x0 + 8, y0 - th / 2], [x0 + 8, y0 - th - 6], [x0 + 8, y0 + 6], [x0 - tw - 8, y0 - th / 2], [x0 - tw - 8, y0 - th - 6], [x0 - tw - 8, y0 + 6]]
        .map(([x, y]) => [Math.max(2, Math.min(w - tw - 2, x)), Math.max(2, Math.min(h - th - 2, y))]);
      const hit = ([x, y]) => placed.some((r) => x < r[0] + r[2] + 3 && r[0] < x + tw + 3 && y < r[1] + r[3] + 2 && r[1] < y + th + 2);
      const [lx, ly] = cands.find((c) => !hit(c)) || cands[0];
      placed.push([lx, ly, tw, th]);
      g.fillStyle = "rgba(255,255,255,0.88)"; g.beginPath();
      g.roundRect ? g.roundRect(lx, ly, tw, th, 8) : g.rect(lx, ly, tw, th); g.fill();
      g.fillStyle = col; g.fillText(text, lx + 5, ly + th / 2 + 0.5);
    };
    if (!bare) pill("start", sx, sy, C.start);

    const now = performance.now();
    (marks || []).forEach((m, j) => {
      const i = Math.min(n - 1, Math.round(m.t / tr.dt)), [mx, my] = P(i);
      if (m.on) {
        const r = 9 + 5 * (0.5 + 0.5 * Math.sin(now / 180));
        g.fillStyle = "rgba(148,36,28,0.16)"; g.beginPath(); g.arc(mx, my, r, 0, 2 * Math.PI); g.fill();
      }
      g.fillStyle = C.mark; g.strokeStyle = "#fff"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(mx, my, m.on ? 5.5 : 4, 0, 2 * Math.PI); g.fill(); g.stroke();
      pill(m.label, mx, my, C.mark, m.on);
    });

    // current camera: soft view cone + haloed dot
    const yaw = tr.yaw[ki] + (((tr.yaw[i2] - tr.yaw[ki] + 3 * Math.PI) % (2 * Math.PI)) - Math.PI) * f;
    const half = (tr.hfov / 2) * Math.PI / 180;
    const cone = g.createRadialGradient(px, py, 0, px, py, L);
    cone.addColorStop(0, `rgba(14,115,135,${bare ? 0.5 : 0.32})`); cone.addColorStop(1, "rgba(14,115,135,0)");
    g.fillStyle = cone; g.beginPath(); g.moveTo(px, py);
    g.arc(px, py, L, yaw - half - Math.PI / 2, yaw + half - Math.PI / 2); g.closePath(); g.fill();
    if (bare) {   // small insets: a thin hollow ring, no halo
      g.strokeStyle = C.done; g.lineWidth = 1.5;
      g.beginPath(); g.arc(px, py, 3, 0, 2 * Math.PI); g.stroke();
    } else {
      g.fillStyle = "rgba(14,115,135,0.18)"; g.beginPath(); g.arc(px, py, 10, 0, 2 * Math.PI); g.fill();
      g.fillStyle = C.done; g.strokeStyle = "#fff"; g.lineWidth = 2;
      g.beginPath(); g.arc(px, py, 5.5, 0, 2 * Math.PI); g.fill(); g.stroke();
    }
  }

  function updateBankStrip(row) {
    const b = row._bankStrip;
    if (!b) return;
    const v = row.querySelector(`video[src*="${b.clip}"]`) || row.querySelector(`[data-video*="${b.clip}"]`);
    const t = v && v.tagName === "VIDEO" ? v.currentTime : 0;
    const lat = Math.floor(t / 0.25), c = Math.max(0, Math.min(b.chunks.length - 1, Math.floor(Math.max(0, lat - 1) / (b.chunk_latents || 4))));
    if (c === b.chunk) return;
    b.chunk = c;
    const ids = b.chunks[c] || [];
    b.imgs.forEach((im, i) => {
      const j = ids[i];
      im.style.visibility = j == null ? "hidden" : "visible";
      if (j != null) im.src = `${b.dir}/k${j}.jpg`;
    });
    const c0 = (1 + 4 * c) * 0.25;
    b.tEl.textContent = ids.length ? `\u00b7 generating ${c0.toFixed(1)}\u2013${(c0 + 0.75).toFixed(1)} s` : "\u00b7 bank still empty";
  }

  function drawMap(row) {
    const tr = row._traj, cv = row.querySelector(".trajitem canvas");
    if (!tr || !cv) return;
    const t = rowTime(row);
    if (row._drawnAt === t && cv.width === Math.round(cv.clientWidth * (window.devicePixelRatio || 1))) return;
    row._drawnAt = t;
    drawTopView(cv, tr, t, (row._times || []).map((tm, j) => ({ t: tm, label: "R" + (j + 1) })));
  }

  (function loop() {
    document.querySelectorAll(".grow.active").forEach((g) => { checkStops(g); drawMap(g); updateKeys(g); updateBankStrip(g); });
    requestAnimationFrame(loop);
  })();

  // More Results: a strip that rests on a card for a few seconds, then steps one to the left; it can be moved with
  // the arrows, a mouse drag, a finger or a trackpad. The set of cards is
  // duplicated and the offset wraps by one set width, so it loops seamlessly.
  // Only cards actually on screen download and play; a card that leaves is
  // detached (aborting its transfer) and starts again from t = 0 when it returns.
  // Teaser lightbox: one clip at its native 864x480 over a grey backdrop, with arrows to step
  // through the strip; Esc, the close button or a click on the backdrop closes it.
  function teaserLightbox(clips) {
    const arrow = (d) => `<svg viewBox="0 0 24 24" width="22" height="22"><path d="${d}" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    const el = document.createElement("div");
    el.className = "lb"; el.hidden = true;
    el.innerHTML = `<button class="lb-close" aria-label="Close">&times;</button>
      <button class="mq-arrow lb-prev" aria-label="Previous clip">${arrow("M15 5l-7 7 7 7")}</button>
      <figure class="lb-stage"><div class="lb-media"><video controls loop muted playsinline></video><div class="tmap lb-map"><canvas></canvas></div></div><figcaption></figcaption></figure>
      <button class="mq-arrow lb-next" aria-label="Next clip">${arrow("M9 5l7 7-7 7")}</button>`;
    document.body.appendChild(el);
    const v = el.querySelector("video"), cap = el.querySelector("figcaption"), cv = el.querySelector(".lb-map canvas");
    const st = { open: false, i: 0 }, trajs = {};
    const traj = (i) => {
      if (!(i in trajs)) {
        trajs[i] = null;
        fetch((clips[i].traj ? clips[i].traj + AV + "planned" : `assets/traj/teaser59/t${String(i).padStart(2, "0")}.json${AV}`)).then((r) => r.json()).then((t) => { trajs[i] = t; }).catch(() => {});
      }
      return trajs[i];
    };
    (function draw() {
      if (st.open) { const t = traj(st.i); if (t) drawTopView(cv, t, v.currentTime || 0, [], true); }
      requestAnimationFrame(draw);
    })();
    const load = (i) => {
      st.i = (i + clips.length) % clips.length;
      v.poster = clips[st.i].poster; v.src = clips[st.i].src; v.play().catch(() => {});
      cap.textContent = `${st.i + 1} / ${clips.length}`;
    };
    st.show = (i) => {
      st.open = true; el.hidden = false; document.body.classList.add("lb-lock");
      requestAnimationFrame(() => el.classList.add("on"));
      load(i);
    };
    const hide = () => {
      st.open = false; el.classList.remove("on"); document.body.classList.remove("lb-lock");
      v.pause(); v.removeAttribute("src"); v.load();
      setTimeout(() => { if (!st.open) el.hidden = true; }, 250);
    };
    el.querySelector(".lb-prev").addEventListener("click", (e) => { e.stopPropagation(); load(st.i - 1); });
    el.querySelector(".lb-next").addEventListener("click", (e) => { e.stopPropagation(); load(st.i + 1); });
    el.querySelector(".lb-close").addEventListener("click", hide);
    el.addEventListener("click", (e) => { if (e.target === el) hide(); });
    document.addEventListener("keydown", (e) => {
      if (!st.open) return;
      if (e.key === "Escape") hide();
      else if (e.key === "ArrowLeft") load(st.i - 1);
      else if (e.key === "ArrowRight") load(st.i + 1);
    });
    return st;
  }

  // More Results: the 4-step model on the other web photographs. Hover plays a clip, click opens it
  // full size in the same lightbox as the teaser (with its own camera path).
  function moreResults(list) {
    const grid = document.querySelector("#more .mgrid");
    if (!grid || !list.length) return;
    const lb = teaserLightbox(list.map((x) => ({ src: x.src, poster: x.poster, traj: x.traj })));
    list.forEach((x, i) => {
      const t = document.createElement("div");
      t.className = "mtile";
      t.innerHTML = `<img src="${x.poster}${AV}p" alt="" loading="lazy"><video muted loop playsinline preload="none"></video><span>${x.label}</span>`;
      const v = t.querySelector("video");
      t.addEventListener("mouseenter", () => { if (!v.getAttribute("src")) v.src = x.src; v.play().then(() => t.classList.add("on")).catch(() => {}); });
      t.addEventListener("mouseleave", () => { v.pause(); t.classList.remove("on"); });
      t.addEventListener("click", () => { v.pause(); t.classList.remove("on"); lb.show(i); });
      grid.appendChild(t);
    });
    document.getElementById("more").hidden = false;
  }

  const marquee = document.querySelector(".marquee");
  if (marquee) {
    const wrap = marquee.parentElement;
    const track = marquee.querySelector(".track");
    [...track.children].forEach((t) => track.appendChild(t.cloneNode(true)));
    const tiles = [...track.querySelectorAll(".ttile")];
    // camera path in the top-left corner of the centre card, drawn at the clip's own time
    const trajs = {};
    tiles.forEach((t) => {
      const m = document.createElement("div"); m.className = "tmap";
      m.appendChild(document.createElement("canvas")); t.appendChild(m);
    });
    let lit = null;
    function drawTeaserMap(t) {
      if (t !== lit) { if (lit) lit.classList.remove("centre"); if (t) t.classList.add("centre"); lit = t; }
      if (!t) return;
      const i = +t.dataset.idx;
      if (!(i in trajs)) {
        trajs[i] = null;
        fetch(`assets/traj/teaser59/t${String(i).padStart(2, "0")}.json${AV}`).then((r) => r.json()).then((tr) => { trajs[i] = tr; }).catch(() => {});
      }
      if (trajs[i]) drawTopView(t.querySelector(".tmap canvas"), trajs[i], t.querySelector("video").currentTime || 0, [], true);
    }
    const lb = teaserLightbox(tiles.slice(0, tiles.length / 2).map((t) => t.querySelector("video")).map((v) => ({ src: v.dataset.src, poster: v.poster })));

    const HOLD_MS = 4000;                               // rest on each card, then step one
    let x = 0, setW = track.scrollWidth / 2, last = performance.now();
    let hover = false, drag = null, tween = null, inView = true;
    const stepW = () => tiles[1].offsetLeft - tiles[0].offsetLeft;
    // the offset at which some card sits exactly in the middle, nearest to v
    const snap = (v) => {
      const x0 = marquee.clientWidth / 2 - tiles[0].offsetLeft - tiles[0].offsetWidth / 2, st = stepW();
      return x0 + Math.round((v - x0) / st) * st;
    };
    const wrapX = () => { while (x <= -setW) x += setW; while (x > 0) x -= setW; };
    window.addEventListener("resize", () => { setW = track.scrollWidth / 2; });

    (function frame(now) {
      const dt = Math.min(now - last, 64) / 1000; last = now;
      if (tween) {
        const k = Math.min(1, (now - tween.t0) / 700), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        x = tween.from + (tween.to - tween.from) * e;
        if (k === 1) tween = null;
      }
      wrapX();
      track.style.transform = `translate3d(${x}px,0,0)`;
      // the card nearest the centre is full size with a deeper shadow, cards
      // toward the edges are a little smaller; colours are never washed out
      const vw = marquee.clientWidth, half = vw / 2;
      let centre = null, best = Infinity;
      tiles.forEach((t) => {
        const c = x + t.offsetLeft + t.offsetWidth / 2;
        const sd = Math.max(-1, Math.min(1, (c - half) / half)), d = Math.abs(sd);
        if (c < -t.offsetWidth || c > vw + t.offsetWidth) return;
        // size by distance in cards: the centre one 1.18x, its neighbours ~0.87x, then down to 0.8x.
        // Two neighbours' scales always sum to <= 2.07, below 2 * step / width, so they never overlap.
        const dn = Math.abs(c - half) / stepW();
        if (dn < best) { best = dn; centre = t; }
        const sc = dn < 1 ? 1.18 - 0.31 * dn : Math.max(0.8, 0.87 - 0.07 * (dn - 1));
        t.style.transform = `scale(${sc.toFixed(4)})`;
        t.style.setProperty("--lift", (1 - d).toFixed(3));
      });
      drawTeaserMap(centre);
      requestAnimationFrame(frame);
    })(last);

    const slide = (dir) => { const from = x; tween = { from, to: snap(from) - dir * stepW(), t0: performance.now() }; };
    x = snap(0); wrapX();
    let lastMove = performance.now();
    setInterval(() => {
      const now = performance.now();
      if (drag || hover || !inView || tween || lb.open || now - lastMove < HOLD_MS) return;
      slide(1); lastMove = now;
    }, 250);
    let wheelT = null;
    const settle = () => { tween = { from: x, to: snap(x), t0: performance.now() }; lastMove = performance.now(); };
    wrap.querySelector(".mq-next").addEventListener("click", () => { slide(1); lastMove = performance.now(); });
    wrap.querySelector(".mq-prev").addEventListener("click", () => { slide(-1); lastMove = performance.now(); });
    marquee.addEventListener("mouseenter", () => { hover = true; });
    marquee.addEventListener("mouseleave", () => { hover = false; lastMove = performance.now(); });
    marquee.addEventListener("pointerdown", (e) => {
      tween = null; drag = { x0: e.clientX, from: x };
      marquee.classList.add("dragging"); marquee.setPointerCapture(e.pointerId);
    });
    marquee.addEventListener("pointermove", (e) => { if (drag) { x = drag.from + (e.clientX - drag.x0); } });
    const endDrag = (e) => {
      // a press that barely moved is a click: open that clip full size
      const click = drag && e.type === "pointerup" && Math.abs(e.clientX - drag.x0) < 6;
      if (drag) settle(); drag = null; marquee.classList.remove("dragging");
      if (click) {
        if (marquee.hasPointerCapture(e.pointerId)) marquee.releasePointerCapture(e.pointerId);
        const t = document.elementFromPoint(e.clientX, e.clientY)?.closest(".ttile");
        if (t) lb.show(+t.dataset.idx);
      }
    };
    marquee.addEventListener("pointerup", endDrag);
    marquee.addEventListener("pointercancel", endDrag);
    marquee.addEventListener("wheel", (e) => {
      if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
      e.preventDefault(); tween = null; x -= e.deltaX;
      clearTimeout(wheelT); wheelT = setTimeout(settle, 180);
    }, { passive: false });

    new IntersectionObserver(([e]) => { inView = e.isIntersecting; }).observe(marquee);
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      const v = e.target.querySelector("video");
      // every appearance starts the clip from t = 0; off screen it neither plays nor downloads
      if (e.isIntersecting) {
        if (!v.getAttribute("src")) v.src = v.dataset.src;
        else v.currentTime = 0;
        v.play().catch(() => {});
      } else if (v.getAttribute("src")) {
        v.pause();
        v.removeAttribute("src"); v.load();
      }
    }), { threshold: 0.2 });
    tiles.forEach((t) => io.observe(t));

    // HUD: each card lit from its own clip's trajectory at its own time
    fetch("assets/videos/teaser59/keys.json").then((r) => r.json()).then((hud) => {
      const huds = tiles.map((t) => {
        const o = keyOverlay();
        t.appendChild(o);
        return { v: t.querySelector("video"), ks: [...o.querySelectorAll(".k")], keys: hud.tiles[+t.dataset.idx].keys, at: -1 };
      });
      (function tick() {
        huds.forEach((h) => {
          if (h.v.paused && h.at >= 0) return;
          const i = Math.max(0, Math.min(h.keys.length - 1, Math.floor(h.v.currentTime / hud.dt)));
          if (i === h.at) return;
          h.at = i;
          h.ks.forEach((e) => e.classList.toggle("on", h.keys[i].includes(e.dataset.k)));
        });
        requestAnimationFrame(tick);
      })();
    }).catch(() => {});
  }

  // Fig. 1(a) step animation: episode -> ordinary training -> long span ->
  // per-chunk top-K -> model input, then A, bank, recent, target in turn.
  const fig = document.getElementById("fig1a");
  if (fig) {
    const N = 8, HOLD = [1800, 2600, 2600, 2800, 1500, 1500, 1500, 3200];
    const groups = [...fig.querySelectorAll("[data-s]")];
    const inputParts = [...fig.querySelectorAll('g[data-s="5"] [data-part]')];
    const loss = fig.querySelector(".loss");
    const spanT = fig.querySelector(".tgHL");
    const retr = fig.querySelector('g[data-s="4"]');
    const items = [...document.querySelectorAll(".insight-text li")];
    const dots = document.querySelector(".fig-dots");
    const playBtn = document.querySelector(".fig-play");
    for (let i = 1; i <= N; i++) { const b = document.createElement("button"); b.setAttribute("aria-label", `Step ${i}`); b.dataset.i = i; dots.appendChild(b); }
    let step = 1, timer = null, playing = true;
    const render = () => {
      groups.forEach((g) => g.classList.toggle("hide", +g.dataset.s > Math.min(step, 5)));
      const focus = { 5: "A", 6: "B", 7: "R", 8: "T" }[step];
      inputParts.forEach((g) => g.classList.toggle("dim", !!focus && g.dataset.part !== focus));
      inputParts.forEach((g) => g.classList.toggle("glow", g.dataset.part === focus));
      loss.classList.toggle("hide", step < 8);
      spanT.classList.toggle("glow", step === 3);
      retr.classList.toggle("flow", step === 4);
      items.forEach((li) => li.classList.toggle("on", li.dataset.for.split(" ").includes(String(step))));
      [...dots.children].forEach((d) => d.classList.toggle("on", +d.dataset.i === step));
    };
    const tick = () => { render(); timer = setTimeout(() => { step = step % N + 1; tick(); }, HOLD[step - 1]); };
    const stop = () => { clearTimeout(timer); timer = null; };
    const setPlaying = (on) => { playing = on; playBtn.textContent = on ? "❚❚" : "▶"; on ? tick() : stop(); };
    playBtn.addEventListener("click", () => setPlaying(!playing));
    dots.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; step = +b.dataset.i; stop(); setPlaying(false); render(); });
    // only animate while on screen
    new IntersectionObserver(([e]) => { if (!playing) return; e.isIntersecting ? (timer || tick()) : stop(); }).observe(fig);
    render();
  }

  // "Back to the same place": one clip, its top-view path with the return
  // highlighted, the two matching frames, and what each chunk retrieved.
  function revisitSection(list) {
    const root = document.querySelector(".rv");
    if (!root || !list.length) return;
    document.getElementById("revisit").hidden = false;
    const stage = root.querySelector(".rv-stage");
    const items = list.map((r) => {
      const el = document.createElement("div");
      el.className = "rv-item";
      el.innerHTML = `
        <div class="rv-main"><button class="rv-play">▶</button><div class="rv-now">now \u00b7 0.0 s</div><div class="rv-stopnote"></div><video muted playsinline preload="none" poster="${r.poster}${AV}" data-src="${r.clip}${AV}"></video></div>
        <div class="rv-side">
          <div class="rv-pair">
            <figure class="rv-start"><img src="${r.poster}${AV}" alt=""><figcaption>first frame \u00b7 0.0 s</figcaption></figure>
            ${r.near_start != null ? `<figure class="rv-near rv-slide"><img src="${r.poster.replace("_poster", "_near")}${AV}" alt=""><figcaption>back near the start \u00b7 ${r.near_start.toFixed(1)} s</figcaption></figure>` : '<figure class="rv-near"></figure>'}
          </div>
          <div class="rv-pair">
            <figure class="rv-first rv-slide"><img src="${r.first_img}${AV}" alt=""><figcaption>first visit \u00b7 ${r.first.toFixed(1)} s</figcaption></figure>
            <figure class="rv-ret rv-slide"><img src="${r.return_img}${AV}" alt=""><figcaption>back again \u00b7 ${r.return.toFixed(1)} s</figcaption></figure>
          </div>
          <div class="rv-pair">
            <figure class="rv-top1 empty"><img alt=""><figcaption><span>top-1 memory</span></figcaption></figure>
            <figure class="rv-mapfig"><canvas class="rv-map"></canvas><figcaption>camera path \u00b7 top view</figcaption></figure>
          </div>
        </div>
`;
      stage.appendChild(el);
      const v = el.querySelector("video");
      el.querySelector(".rv-main").appendChild(keyOverlay());
      const o = { r, el, v, hud: el.querySelector(".rv-top1"), now: el.querySelector(".rv-now"),
                  first: el.querySelector(".rv-first"), ret: el.querySelector(".rv-ret"), near: el.querySelector(".rv-near"), btn: el.querySelector(".rv-play"),
                  cv: el.querySelector(".rv-map"), traj: null, chunk: -1, keysAt: -1, ks: [...el.querySelectorAll(".keys .k")] };
      fetch(r.traj).then((x) => x.json()).then((t) => { o.traj = t; });
      o.part = 0;
      const toggle = () => {
        endRvStop(o); if (!v.getAttribute("src")) v.src = v.dataset.src;
        if (v.ended) { setPart(o, 0); v.currentTime = 0; }
        v.paused ? v.play().catch(() => {}) : v.pause();
      };
      v.addEventListener("click", toggle); o.btn.addEventListener("click", toggle);
      v.addEventListener("play", () => { o.btn.textContent = "❚❚"; });
      v.addEventListener("pause", () => { o.btn.textContent = "▶"; });
      return o;
    });

    let cur = 0, inView = false;
    const count = root.querySelector(".count");
    // picker strip under the stage: every example as a thumbnail, the current one lit; click to switch
    const picker = document.createElement("div");
    picker.className = "rv-picker";
    const picks = list.map((r, k) => {
      const p = document.createElement("button");
      p.type = "button"; p.className = "rv-pick"; p.title = r.label || r.id;
      p.innerHTML = `<img src="${r.poster}${AV}" alt="" loading="lazy"><span class="rv-pick-label">${r.label || r.id}</span>`;
      p.addEventListener("click", () => { if (k !== cur) show(k, k > cur ? 1 : -1); });
      picker.appendChild(p);
      return p;
    });
    stage.after(picker);
    // clips split into parts: time on the whole walk = part * part_dur + time in the part
    function setPart(o, i) {
      if (!o.r.parts) return;
      o.part = i; o.v.dataset.src = o.r.parts[i]; o.v.src = o.r.parts[i];
      const nx = o.r.parts[i + 1];
      if (nx) { const w = document.createElement("link"); w.rel = "prefetch"; w.href = nx; document.head.appendChild(w); }
    }
    const clipTime = (o) => (o.r.parts ? o.part * o.r.part_dur : 0) + (o.v.currentTime || 0);
    function seekClip(o, t) {
      if (o.r.parts) { const i = Math.min(o.r.parts.length - 1, Math.floor(t / o.r.part_dur)); if (i !== o.part) setPart(o, i); o.v.currentTime = t - i * o.r.part_dur; }
      else o.v.currentTime = t;
    }
    const show = (i, dir) => {
      items.forEach((o) => { if (!o.v.paused) o.v.pause(); if (o.v.getAttribute("src")) { o.v.removeAttribute("src"); o.v.load(); } });
      cur = (i + items.length) % items.length;
      items.forEach((o, k) => o.el.classList.toggle("active", k === cur));
      const ne = items[cur].el;
      ne.classList.remove("enter-next", "enter-prev");
      if (dir) { void ne.offsetWidth; ne.classList.add(dir > 0 ? "enter-next" : "enter-prev"); }
      count.textContent = `${cur + 1} / ${items.length}`;
      picks.forEach((p, k) => p.classList.toggle("active", k === cur));
      if (picker.scrollWidth > picker.clientWidth) picks[cur].scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
      const o = items[cur];
      items.forEach(endRvStop);
      if (o.r.parts) setPart(o, 0); else o.v.src = o.v.dataset.src;
      o.chunk = -1; (o.stops || []).forEach((x) => { x.done = false; });
      o.pre = o.pre || new Set();   // top-1 frames are fetched a few chunks ahead, not all at once
      if (inView) o.v.play().catch(() => {});
    };
    // when an example finishes, slide on to the next one by itself
    items.forEach((o, k) => o.v.addEventListener("ended", () => {
      if (k !== cur || !inView) return;
      if (o.r.parts && o.part < o.r.parts.length - 1) { setPart(o, o.part + 1); o.v.play().catch(() => {}); return; }
      show(cur + 1, 1);
    }));
    root.querySelector(".prev").addEventListener("click", () => show(cur - 1, -1));
    root.querySelector(".next").addEventListener("click", () => show(cur + 1, 1));
    new IntersectionObserver(([e]) => {
      inView = e.isIntersecting;
      const o = items[cur];
      if (inView) { if (!o.v.getAttribute("src")) o.v.src = o.v.dataset.src; o.v.play().catch(() => {}); } else o.v.pause();
    }, { threshold: 0.35 }).observe(root);
    show(0);

    // hold on the first visit and on the return, like the comparison rows do
    function endRvStop(o) { clearTimeout(o.stopT); o.stopT = null; o.el.classList.remove("at-stop"); }
    function checkRvStop(o, t) {
      if (!o.stops) o.stops = [
        { t: o.r.first, label: `first visit \u00b7 ${o.r.first.toFixed(1)} s \u00b7 remember this view`, fig: o.first },
        { t: o.r.return, label: `back again \u00b7 ${o.r.return.toFixed(1)} s`, fig: o.ret },
        ...(o.r.near_start != null ? [{ t: o.r.near_start, label: `back near the start \u00b7 ${o.r.near_start.toFixed(1)} s \u00b7 compare with the first frame`, fig: o.near }] : [])];
      o.stops.forEach((st) => { if (t < st.t - 1) st.done = false; });
      if (o.stopT || o.v.paused) return;
      const st = o.stops.find((x) => !x.done && t >= x.t - 0.08 && t < x.t + 1);
      if (!st) return;
      st.done = true;
      o.v.pause(); seekClip(o, st.t);
      o.el.querySelector(".rv-stopnote").textContent = st.label;
      o.el.classList.add("at-stop");
      o.stopT = setTimeout(() => { endRvStop(o); if (items[cur] === o && inView) o.v.play().catch(() => {}); }, STOP_MS);
    }

    const drawRv = (o, t) => {
      if (!o.traj) return;
      drawTopView(o.cv, o.traj, t, [
        { t: o.r.first, label: `first visit \u00b7 ${o.r.first.toFixed(0)} s`, on: Math.abs(t - o.r.first) < 1.5 },
        { t: o.r.return, label: `back again \u00b7 ${o.r.return.toFixed(0)} s`, on: Math.abs(t - o.r.return) < 1.5 },
      ]);
    };

    (function loop() {
      const o = items[cur];
      if (o) {
        const t = clipTime(o);
        drawRv(o, t);
        o.first.classList.toggle("hit", Math.abs(t - o.r.first) < 1.5);
        o.ret.classList.toggle("hit", Math.abs(t - o.r.return) < 1.5);
        o.first.classList.toggle("shown", t >= o.r.first - 0.1);
        o.ret.classList.toggle("shown", t >= o.r.return - 0.1);
        if (o.r.near_start != null) {
          o.near.classList.toggle("shown", t >= o.r.near_start - 0.1);
          o.near.classList.toggle("hit", Math.abs(t - o.r.near_start) < 1.5);
        }
        checkRvStop(o, t);
        const ts = `now \u00b7 ${t.toFixed(1)} s`; if (o.now.textContent !== ts) o.now.textContent = ts;
        if (o.traj && o.traj.keys) {
          const i = Math.max(0, Math.min(o.traj.keys.length - 1, Math.floor(t / o.traj.dt)));
          if (i !== o.keysAt) { o.keysAt = i; o.ks.forEach((e) => e.classList.toggle("on", o.traj.keys[i].includes(e.dataset.k))); }
        }
        // chunk c generates latents; show what it retrieved (sprite thumbnails, 16 per row)
        const lat = Math.floor(t / 0.25), c = Math.max(0, Math.min(o.r.retrieval.length - 1, Math.floor(Math.max(0, lat - 1) / (o.r.chunk_latents || 4))));
        if (c !== o.chunk) {
          o.chunk = c;
          // HUD: the frame this chunk scored highest among everything generated before it
          for (let a = c + 1; a <= c + 6; a++) {
            const kk = (o.r.retrieval[a] || [])[0];
            if (kk != null && o.r.top1_dir && !o.pre.has(kk)) { o.pre.add(kk); new Image().src = `${o.r.top1_dir}/k${kk}.jpg${AV}`; }
          }
          const k = (o.r.retrieval[c] || [])[0];
          o.hud.classList.toggle("empty", k == null);
          if (k != null) {
            o.hud.querySelector("img").src = `${o.r.top1_dir}/k${k}.jpg${AV}`;
            const fromFirst = Math.abs(k - o.r.first_latent) <= 8;
            o.hud.classList.toggle("first", fromFirst);
            o.hud.querySelector("span").textContent = `top-1 memory \u00b7 ${(k * 0.25).toFixed(1)} s` + (fromFirst ? " \u00b7 first visit" : "");
          }
        }
      }
      requestAnimationFrame(loop);
    })();
  }


  // Fig. 1(b): a cursor sweeps the training span; the pool follows it up while the bank stays flat
  (function fig1b() {
    const svg = document.getElementById("fig1b");
    if (!svg) return;
    const X0 = 70, X1 = 650, Y0 = 24, Y1 = 236, xmin = 28, xmax = 2400, ymin = 4.6, ymax = 3400;
    const lx = (v) => X0 + (Math.log(v) - Math.log(xmin)) / (Math.log(xmax) - Math.log(xmin)) * (X1 - X0);
    const ly = (v) => Y1 - (Math.log(v) - Math.log(ymin)) / (Math.log(ymax) - Math.log(ymin)) * (Y1 - Y0);
    const xs = [81, 101, 201, 401, 801, 1601], pool = [36, 56, 156, 356, 756, 1556], bank6 = [14.2, 17.5, 22.9, 27.9, 30.7, 31.5];
    const interp = (x, ys) => {   // log-log interpolation between measured spans
      let i = 0; while (i < xs.length - 2 && x > xs[i + 1]) i++;
      const u = (Math.log(x) - Math.log(xs[i])) / (Math.log(xs[i + 1]) - Math.log(xs[i]));
      return Math.exp(Math.log(ys[i]) + u * (Math.log(ys[i + 1]) - Math.log(ys[i])));
    };
    const reveal = svg.querySelector(".f1b-reveal"), cur = svg.querySelector(".f1b-cur"), line = cur.querySelector("line");
    const cp = cur.querySelector(".cp"), cb = cur.querySelector(".cb"), rd = svg.querySelector(".f1b-read");
    const [r1, r2, r3] = ["r1", "r2", "r3"].map((c) => rd.querySelector("." + c));
    const SWEEP = 5200, HOLD = 2600;
    let t0 = null, vis = false;
    const draw = (x) => {
      const px = lx(x), p = interp(x, pool), b = interp(x, bank6);
      reveal.setAttribute("width", Math.max(0, px - X0));
      line.setAttribute("x1", px); line.setAttribute("x2", px);
      cp.setAttribute("cx", px); cp.setAttribute("cy", ly(p));
      cb.setAttribute("cx", px); cb.setAttribute("cy", ly(b));
      const rx = Math.min(px + 12, X1 - 156), ry = Math.max(Y0 + 30, ly(p) - 10);
      rd.setAttribute("transform", `translate(${rx} ${ry})`);
      r1.textContent = `span ${Math.round(x)} latents \u00b7 ${(x / 4).toFixed(0)} s`;
      r2.textContent = `pool ${Math.round(p)}`;
      r3.textContent = `bank (K = 6) ${b.toFixed(1)}`;
    };
    const frame = (now) => {
      if (!vis) { t0 = null; return; }
      if (t0 == null) t0 = now;
      const e = (now - t0) % (SWEEP + HOLD), u = Math.min(1, e / SWEEP);
      const k = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      draw(Math.exp(Math.log(81) + k * (Math.log(1601) - Math.log(81))));
      requestAnimationFrame(frame);
    };
    draw(1601);
    new IntersectionObserver(([en]) => { const was = vis; vis = en.isIntersecting; if (vis && !was) requestAnimationFrame(frame); }, { threshold: 0.3 }).observe(svg);
  })();

  // Prev / Next of sections 2-4 look and sit like the strip's arrows: round, at the sides
  const CHEV = { prev: "M15 5l-7 7 7 7", next: "M9 5l7 7-7 7" };
  document.querySelectorAll(".gnav .prev, .gnav .next").forEach((b) => {
    const k = b.classList.contains("prev") ? "prev" : "next";
    b.classList.add("side-arrow");
    b.setAttribute("aria-label", k === "prev" ? "Previous example" : "Next example");
    b.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20"><path d="${CHEV[k]}" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  });

  const getJSON = (u) => fetch(u, { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(u)));
  getJSON(`assets/videos/revisit/manifest.json${AV}f160`).then(revisitSection).catch(() => {});

  // unseen-scene ablation: all eight Table 2 settings per example
  // dir = asset folder name, tp = trajectory file prefix
  function ablationRows(m, dir, tp) {
    return (m.items || m.rows).map((r) => ({
      input: `assets/img/${dir}/${r.id}_input.jpg`, traj: `assets/traj/${tp}${r.id}.json`,
      times: r.times, labels: r.labels || [], layout: "lay-grid",
      stops: stopsFor(r),
      cells: m.settings.map(([key, label]) => ({
        name: label, ours: key === "m399",
        poster: `assets/img/${dir}/${r.id}_${key}.jpg`, video: `assets/videos/${dir}/${r.id}_${key}.mp4` })),
    }));
  }

  function build(kind, rows) {
    const root = document.querySelector(`.carousel[data-kind="${kind}"]`);
    if (!root || !rows.length) return;
    const c = carousel(root, rows);
    if (kind === "sota") focused = c;
  }

  // Comparison with SOTA: ours vs the five baselines
  getJSON(`assets/videos/gallery_manifest.json${AV}k24s`).then(moreResults).catch(() => {});
  getJSON(`assets/videos/v7/manifest.json${AV}p`).then((list) => build("sota", rowsFor("sota", list, "v7"))).catch(() => {});
  // Ablation Study: eight training settings on unseen scenes
  // Bank contents (paper Table 7): one walk, the bank's frames swapped at inference
  getJSON("assets/videos/bank/manifest.json").then((m) => {
    const c = "c" + String(m.case).padStart(2, "0");
    const sb = m.swap_bank;
    build("bank", [{
      input: `assets/img/bank/${c}_input.jpg`, traj: `assets/traj/bank_${c}.json`, times: [], labels: [], layout: "lay-row", mapFirst: true,
      cells: m.arms.map(([k, name]) => ({ name, ours: k === "topk", row2: !!sb && k === "other",
        poster: `assets/img/bank/${c}_${k}.jpg`, video: `assets/videos/bank/${c}_${k}.mp4` })),
      bankStrip: sb ? { clip: `${c}_other.mp4`, dir: sb.dir, chunks: sb.chunks, chunk_latents: sb.chunk_latents, donor: "ancient ruins on grassy hills" } : null,
    }]);
  }).catch(() => {});
  // web-photo examples first, then the first two unseen-scene examples
  Promise.all([getJSON("assets/videos/abl_unseen2/manifest.json"), getJSON("assets/videos/abl_web/manifest.json").catch(() => null)])
    .then(([u, w]) => build("ablu", w ? ablationRows(w, "abl_web", "ablw_").concat(ablationRows(u, "abl_unseen2", "abl2_").slice(0, 2))
                                        : ablationRows(u, "abl_unseen2", "abl2_")))
    .catch(() => {});
})();
