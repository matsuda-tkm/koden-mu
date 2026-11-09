// IQ データビューアの画面処理。
// サーバーから処理結果を取得して Plotly で描画し、自動更新が有効な間は
// 5 秒ごとに新しく処理されたファイルを取り込む。

const JET = [[0, "#000080"], [0.125, "#0000ff"], [0.25, "#0080ff"], [0.375, "#16ffe1"],
  [0.5, "#7dff7a"], [0.625, "#e4ff13"], [0.75, "#ff9400"], [0.875, "#ff1e00"], [1, "#800000"]];
const TURBO = [[0, "#30123b"], [0.1, "#4559cb"], [0.2, "#3e9bfe"], [0.3, "#19d5cd"],
  [0.4, "#46f884"], [0.5, "#a4fc3c"], [0.6, "#e1dd37"], [0.7, "#fea431"], [0.8, "#f05b12"],
  [0.9, "#c32503"], [1, "#7a0403"]];
// 既存の PPI 図（scripts/process_iq.py）と同じ色
const RADAR = ["#FFFFFF", "#A0D2FF", "#218CFF", "#0041FF", "#FAF500", "#FF9900", "#FF2800", "#B40068"]
  .map((c, i, a) => [i / (a.length - 1), c]);
const RNG_STEP = 0.02;
const AZI_STEP = 1.0;
const POLL_MS = 5000;
const CONFIG = { responsive: true, displaylogo: false };
const FONT = { family: getComputedStyle(document.body).fontFamily, size: 12 };

const el = (id) => document.getElementById(id);
const state = {
  folder: null,
  tag: "",
  data: null,      // 時間・高度図のデータ
  selected: null,  // 下段に表示中のファイル名
  fileReq: 0,
  busy: false,
};

function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (e) { /* 保存できない環境では何もしない */ }
  return null;
}

async function getJSON(url) {
  const res = await fetch(url);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

function query(params) {
  return new URLSearchParams({
    folder: state.folder, max_range: el("maxRange").value, axis: el("axis").value, ...params,
  });
}

// 横軸の時刻の意味と、欠測の表し方の説明
const AXIS_NOTE = {
  obs: "横軸は推定観測時刻です。記録ソフトの書き込みは取り込みに追いつかず、ファイルの更新時刻は観測時刻より少しずつ遅れます。"
    + "この遅れを、記録が途切れるたびに 0 に戻り、経過時間の 30% ずつ増えるとみなして補正しています。"
    + "約 17 分ごとにバッファが一杯になり、約 5 分の観測データが失われます。斜線の範囲が欠測です。",
  mtime: "横軸はファイルの更新時刻で、観測時刻より最大で約 5 分遅れます。"
    + "破線は記録が途中で途切れたファイルの位置で、その直前の約 5 分の観測データが失われていますが、この横軸では空白になりません。"
    + "斜線の範囲は、ファイルが書き込まれていない時間（欠測）です。",
};

function percentile(values, p) {
  const v = values.filter((x) => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return 0;
  return v[Math.min(v.length - 1, Math.floor((p / 100) * v.length))];
}

// ---------------------------------------------------------------- フォルダ一覧

async function loadFolders() {
  const res = await getJSON("/api/folders");
  const sel = el("folder");
  sel.innerHTML = "";
  for (const f of res.folders) {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f === "." ? `${res.root}（親フォルダ自身）` : f;
    sel.appendChild(opt);
  }
  if (!res.folders.length) {
    el("status").textContent = `${res.root} に .bin ファイルを含むフォルダがありません`;
    return;
  }
  const last = storage("folder");
  sel.value = res.folders.includes(last) ? last : res.folders[res.folders.length - 1];
  selectFolder(sel.value);
}

function selectFolder(folder) {
  state.folder = folder;
  state.tag = "";
  state.data = null;
  state.selected = null;
  storage("folder", folder);
  Plotly.purge("timeHeight");
  el("timeHeight").innerHTML = '<div class="empty"><span class="spinner"></span>処理結果を読み込んでいます…</div>';
  clearFile();
  refresh();
}

// ---------------------------------------------------------------- 時間・高度図

async function refresh() {
  if (!state.folder) return;
  if (state.busy) {
    state.again = true;  // 取得中に設定が変わった場合は、取得が終わってからもう一度取得する
    return;
  }
  state.busy = true;
  const folder = state.folder;
  try {
    const res = await getJSON(`/api/timeheight?${query({ tag: state.tag })}`);
    if (folder !== state.folder) return;
    state.lastStatus = res.status;
    if (res.data) {
      state.tag = res.tag;
      state.data = res.data;
      drawTimeHeight();
      updateFileSelect();
    } else if (state.data && !state.data.time.length) {
      drawTimeHeight();  // データがない間は、状態に合わせて説明を更新する
    }
    showStatus(res.status);
    const names = (state.data?.name || []).filter((n) => n);
    if (names.length && (el("follow").checked || !state.selected)) {
      const latest = names[names.length - 1];
      if (latest !== state.selected) selectFile(latest);
    }
  } catch (e) {
    el("status").innerHTML = `<span class="error">取得に失敗しました: ${e.message}</span>`;
  } finally {
    state.busy = false;
    if (state.again) {
      state.again = false;
      refresh();
    }
  }
}

function showStatus(s) {
  const d = state.data;
  const parts = [];
  if (s.queued) {
    // 処理待ちのファイルがある間は、サーバーが解析中であることを進捗とともに示す
    const done = s.processed + s.skipped;
    const total = done + s.queued;
    parts.push(`<span class="busy"><span class="spinner"></span>解析中 ${done} / ${total} ファイル`
      + `<progress max="${total}" value="${done}"></progress></span>`);
  }
  parts.push(`処理済み ${s.processed} ファイル`);
  if (d?.first_time) parts.push(`${d.first_time} 〜 ${d.last_time.slice(11)} JST`);
  if (s.skipped) parts.push(`除外 ${s.skipped} ファイル（_0000 と 1 方位分に満たないファイル）`);
  if (d?.gaps.length) {
    const min = d.gaps.reduce((a, g) => a + g.seconds, 0) / 60;
    parts.push(`欠測 ${d.gaps.length} か所（合計 ${min.toFixed(1)} 分）`);
  }
  if (d?.short_time.length && d.axis === "mtime") parts.push(`記録の途切れ ${d.short_time.length} か所`);
  if (d?.n_fixed_time) parts.push(`時刻を補間 ${d.n_fixed_time} ファイル`);
  if (d?.decimated) parts.push("列数が多いため、時間方向に間引いて表示しています");
  let html = parts.map((p) => `<span>${p}</span>`).join("");
  if (s.writing.length) {
    const shown = s.writing.slice(-3).join(", ");
    html += `<span class="writing">取り込み中のため未処理: ${shown}${s.writing.length > 3 ? ` ほか ${s.writing.length - 3} 件` : ""}</span>`;
  }
  el("status").innerHTML = html;
}

function selectedShape() {
  const d = state.data;
  const i = d ? d.name.indexOf(state.selected) : -1;
  if (i < 0) return [];
  return [{
    type: "line", xref: "x", yref: "paper", x0: d.time[i], x1: d.time[i], y0: 0, y1: 1,
    line: { color: "#e0007a", width: 2 },
  }];
}

function drawTimeHeight() {
  const d = state.data;
  const div = el("timeHeight");
  if (!d.time.length) {
    Plotly.purge(div);
    const s = state.lastStatus;
    let msg = "取り込みが完了したファイルがまだありません";
    if (!s?.scanned) msg = '<span class="spinner"></span>フォルダ内のファイルを確認しています…';
    else if (s.queued) msg = '<span class="spinner"></span>ファイルを解析しています。最初のファイルの解析が終わると表示します…';
    div.innerHTML = `<div class="empty">${msg}</div>`;
    return;
  }
  if (div.querySelector(".empty")) div.innerHTML = "";
  const hover = "%{x}<br>距離 %{y:.2f} km<br>%{z}<extra></extra>";
  const traces = [
    {
      type: "heatmap", x: d.time, y: d.range, z: d.moving, colorscale: JET,
      zmin: d.power_range[0], zmax: d.power_range[1], xaxis: "x", yaxis: "y",
      colorbar: { title: { text: "dBm", side: "right" }, y: 0.77, len: 0.46 },
      hovertemplate: hover.replace("%{z}", "受信電力 %{z:.1f} dBm"), hoverongaps: false,
    },
    {
      type: "heatmap", x: d.time, y: d.range, z: d.mean_velocity, colorscale: TURBO,
      zmin: -2, zmax: 10, xaxis: "x", yaxis: "y2",
      colorbar: { title: { text: "m/s", side: "right" }, y: 0.23, len: 0.46 },
      hovertemplate: hover.replace("%{z}", "平均ドップラー速度 %{z:.2f} m/s"), hoverongaps: false,
    },
  ];
  traces.push(...gapTraces(d));
  // 推定観測時刻では途切れの位置が欠測区間として表れるため、破線は更新時刻の軸でだけ引く
  const shortLines = (d.axis === "mtime" ? d.short_time : []).map((t) => ({
    type: "line", xref: "x", yref: "paper", x0: t, x1: t, y0: 0, y1: 1,
    line: { color: "#000", width: 1, dash: "dash" },
  }));
  const layout = {
    font: FONT,
    margin: { l: 60, r: 20, t: 30, b: 40 },
    uirevision: `${state.folder}-${el("maxRange").value}-${d.axis}`,
    plot_bgcolor: "#d9d9d9",
    xaxis: {
      type: "date", anchor: "y2",
      title: { text: d.axis === "obs" ? "推定観測時刻（JST）" : "ファイルの更新時刻（JST）" },
    },
    yaxis: { domain: [0.54, 1], title: { text: "距離 [km]" }, range: [0, d.range[d.range.length - 1] + RNG_STEP] },
    yaxis2: { domain: [0, 0.46], title: { text: "距離 [km]" }, matches: "y" },
    shapes: [...shortLines, ...selectedShape()],
    annotations: [
      { text: "受信電力（ゼロ速度付近を除く）", xref: "paper", yref: "paper", x: 0, y: 1.0, yanchor: "bottom", showarrow: false, xanchor: "left" },
      { text: `平均ドップラー速度（正: 下向き、灰色: 信号が弱い）　斜線: 欠測${d.axis === "mtime" ? "　破線: 記録の途切れ" : ""}`, xref: "paper", yref: "paper", x: 0, y: 0.46, yanchor: "bottom", showarrow: false, xanchor: "left" },
    ],
  };
  Plotly.react(div, traces, layout, CONFIG);
  if (!div.dataset.bound) {
    div.dataset.bound = "1";
    div.on("plotly_click", (ev) => {
      const p = ev.points.find((q) => q.data.type === "heatmap");  // 欠測の四角形は対象外
      if (!p) return;
      const j = Array.isArray(p.pointIndex) ? p.pointIndex[1] : p.pointIndex;
      const name = state.data.name[j];
      if (name) userSelect(name);
    });
  }
}

// 欠測の時間帯を、上下の図それぞれに斜線の四角形で描く
function gapTraces(d) {
  const top = d.range[d.range.length - 1] + RNG_STEP;
  const out = [];
  for (const g of d.gaps.slice(0, 300)) {
    const why = g.kind === "truncated"
      ? "記録ソフトのバッファが一杯になり失われた時間（推定）"
      : "ファイルが書き込まれていない時間";
    const text = `欠測: ${why}<br>${g.start.slice(5)} 〜 ${g.end.slice(11)}（${(g.seconds / 60).toFixed(1)} 分）`;
    for (const yaxis of ["y", "y2"]) {
      out.push({
        type: "scatter", mode: "lines", xaxis: "x", yaxis,
        x: [g.start, g.end, g.end, g.start, g.start], y: [0, 0, top, top, 0],
        fill: "toself", fillcolor: "#ffffff",
        fillpattern: { shape: "/", fgcolor: "#8a8f99", bgcolor: "#ffffff", size: 10, solidity: 0.15 },
        line: { width: 1, color: "#8a8f99" },
        hoveron: "fills", hoverinfo: "text", text, showlegend: false,
      });
    }
  }
  return out;
}

function updateSelectedLine() {
  const div = el("timeHeight");
  if (!div.layout || !state.data) return;
  const keep = (div.layout.shapes || []).filter((s) => s.line.color !== "#e0007a");
  Plotly.relayout(div, { shapes: [...keep, ...selectedShape()] });
}

// ---------------------------------------------------------------- 選択したファイル

function clearFile() {
  for (const id of ["profile", "spectrum", "ppi"]) Plotly.purge(id);
  el("fileSelect").innerHTML = "";
  el("fileTitle").textContent = "";
  updateFileButtons();
}

// 時間・高度図に含まれるファイルの一覧（時刻順）
function fileList() {
  const d = state.data;
  if (!d) return [];
  return d.name.map((n, i) => [n, d.time[i]]).filter(([n]) => n);
}

function updateFileSelect() {
  const sel = el("fileSelect");
  const files = fileList();
  // ファイルが増えたときだけ選択肢を作り直す
  const key = `${state.folder}-${state.data.axis}`;
  if (sel.options.length !== files.length || sel.dataset.key !== key) {
    sel.dataset.key = key;
    sel.innerHTML = "";
    for (const [name, time] of files) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = `${time.slice(5)} JST  ${name}`;
      sel.appendChild(opt);
    }
  }
  sel.value = state.selected || "";
  updateFileButtons();
}

function updateFileButtons() {
  const names = fileList().map(([n]) => n);
  const i = names.indexOf(state.selected);
  el("prevFile").disabled = i <= 0;
  el("nextFile").disabled = i < 0 || i >= names.length - 1;
}

// 利用者がファイルを選んだときは、最新のファイルへの自動切り替えを止める
function userSelect(name) {
  el("follow").checked = false;
  selectFile(name);
}

function stepFile(delta) {
  const names = fileList().map(([n]) => n);
  const i = names.indexOf(state.selected);
  const next = names[i + delta];
  if (i >= 0 && next) userSelect(next);
}

async function selectFile(name) {
  state.selected = name;
  el("fileSelect").value = name;
  updateFileButtons();
  updateSelectedLine();
  const req = ++state.fileReq;
  el("fileTitle").textContent = "";
  el("fileLoadingText").textContent = `${name} を読み込んで計算しています…`;
  el("fileArea").classList.add("loading");
  try {
    const f = await getJSON(`/api/file?${query({ name })}`);
    if (req !== state.fileReq) return;
    const i = state.data?.name.indexOf(f.name) ?? -1;
    el("fileTitle").textContent = i >= 0
      ? `推定観測時刻 ${state.data.obs_time[i].slice(11)}、更新時刻 ${state.data.update_time[i].slice(11)}`
      : "";
    drawProfile(f);
    drawSpectrum(f);
    drawPPI(f);
  } catch (e) {
    if (req === state.fileReq) el("fileTitle").innerHTML = `<span class="error">読み込めませんでした: ${e.message}</span>`;
  } finally {
    if (req === state.fileReq) el("fileArea").classList.remove("loading");
  }
}

function drawProfile(f) {
  const p = f.profile;
  const y = f.range;
  const traces = [
    { x: p.moving, y, name: "ゼロ速度付近を除く", xaxis: "x", yaxis: "y", line: { color: "#1f5fbf" } },
    { x: p.total, y, name: "全速度", xaxis: "x", yaxis: "y", line: { color: "#9aa3b2" } },
    { x: p.mean_velocity, y, name: "平均ドップラー速度", xaxis: "x2", yaxis: "y", line: { color: "#c0392b" }, showlegend: false },
    { x: p.snr, y, name: "SNR", xaxis: "x3", yaxis: "y", line: { color: "#2e7d32" }, showlegend: false },
  ].map((t) => ({ type: "scatter", mode: "lines", hovertemplate: "%{x:.2f}<br>距離 %{y:.2f} km", ...t }));
  const layout = {
    font: FONT,
    title: { text: "高度プロファイル", font: { size: 13 } },
    margin: { l: 55, r: 10, t: 40, b: 70 },
    uirevision: `${state.folder}-${el("maxRange").value}`,
    legend: { orientation: "h", x: 0, y: -0.14 },
    yaxis: { title: { text: "距離 [km]" } },
    xaxis: { domain: [0, 0.38], title: { text: "受信電力 [dBm]" } },
    xaxis2: { domain: [0.42, 0.71], title: { text: "平均速度 [m/s]" }, zeroline: true },
    xaxis3: { domain: [0.75, 1], title: { text: "SNR [dB]" } },
  };
  Plotly.react("profile", traces, layout, CONFIG);
}

function drawSpectrum(f) {
  const flat = f.spectrum.flat();
  const trace = {
    type: "heatmap", x: f.velocity, y: f.range, z: f.spectrum, colorscale: JET,
    zmin: percentile(flat, 5), zmax: percentile(flat, 99.8),
    colorbar: { title: { text: "dBm", side: "right" } },
    hovertemplate: "速度 %{x:.2f} m/s<br>距離 %{y:.2f} km<br>%{z:.1f} dBm<extra></extra>",
  };
  const layout = {
    font: FONT,
    title: { text: "ドップラースペクトル（全方位の平均）", font: { size: 13 } },
    margin: { l: 55, r: 10, t: 40, b: 50 },
    uirevision: `${state.folder}-${el("maxRange").value}`,
    xaxis: { title: { text: "ドップラー速度 [m/s]（正: 下向き）" } },
    yaxis: { title: { text: "距離 [km]" } },
  };
  Plotly.react("spectrum", [trace], layout, CONFIG);
}

function drawPPI(f) {
  // 1 方位 × 1 レンジゲートを 1 本の扇形の棒として描く
  const theta = [], base = [], color = [], custom = [];
  f.angle.forEach((a, i) => {
    f.ppi[i].forEach((v, j) => {
      theta.push(a);
      base.push(f.range[j]);
      color.push(v);
      custom.push([f.range[j] + RNG_STEP / 2, v]);
    });
  });
  const maxRange = f.range[f.range.length - 1] + RNG_STEP;
  const halfSpan = Math.max(6, ...f.angle.map((a) => Math.abs(a) + AZI_STEP));
  const trace = {
    type: "barpolar", theta, base, r: theta.map(() => RNG_STEP), width: AZI_STEP,
    customdata: custom,
    marker: { color, colorscale: RADAR, cmin: -130, cmax: -50, line: { width: 0 },
      colorbar: { title: { text: "dBm", side: "right" } } },
    hovertemplate: "傾き %{theta:.1f}°<br>距離 %{customdata[0]:.2f} km<br>%{customdata[1]:.1f} dBm<extra></extra>",
  };
  const layout = {
    font: FONT,
    title: { text: "PPI 電力マップ", font: { size: 13 } },
    margin: { l: 20, r: 20, t: 40, b: 20 },
    polar: {
      sector: [90 - halfSpan, 90 + halfSpan],
      bargap: 0,
      angularaxis: { rotation: 90, direction: "clockwise", dtick: 5 },
      radialaxis: { range: [0, maxRange], title: { text: "km" } },
    },
  };
  Plotly.react("ppi", [trace], layout, CONFIG);
}

// ---------------------------------------------------------------- 操作

el("folder").addEventListener("change", (e) => selectFolder(e.target.value));
el("fileSelect").addEventListener("change", (e) => userSelect(e.target.value));
el("prevFile").addEventListener("click", () => stepFile(-1));
el("nextFile").addEventListener("click", () => stepFile(1));
el("axis").value = storage("axis") || "obs";
el("axisNote").textContent = AXIS_NOTE[el("axis").value];
el("axis").addEventListener("change", () => {
  storage("axis", el("axis").value);
  el("axisNote").textContent = AXIS_NOTE[el("axis").value];
  state.tag = "";
  refresh();
});
el("maxRange").addEventListener("change", () => {
  state.tag = "";
  refresh().then(() => state.selected && selectFile(state.selected));
});
el("follow").addEventListener("change", () => el("follow").checked && refresh());
setInterval(() => el("auto").checked && refresh(), POLL_MS);
loadFolders().catch((e) => {
  el("status").innerHTML = `<span class="error">フォルダ一覧を取得できませんでした: ${e.message}</span>`;
});
