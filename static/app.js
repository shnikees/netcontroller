/*
 * netcontroller -- live speech-to-text and callsign matching for ham radio nets
 * Copyright (C) 2026 Michelle Michaels
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */

/* Dashboard: connects to /ws, appends a row per transmission.
   No build step -- plain DOM, served straight off the app. */

const log = document.getElementById("log");
const rosterEl = document.getElementById("roster");
const emptyEl = document.getElementById("empty");
const dot = document.getElementById("dot");
const conn = document.getElementById("conn");
const stats = document.getElementById("stats");
const scrollBtn = document.getElementById("scrollBtn");
const clearFilterBtn = document.getElementById("clearFilter");
const exportBtn = document.getElementById("exportBtn");

const toast = document.getElementById("toast");

let autoScroll = true;
let filter = null;
let roster = [];
let sources = [];
/* The active tab is a source name, or ALL for the combined view. The first
   configured source wins by default -- that is the repeater, the frequency
   being monitored, and it should be what is on screen when nobody has touched
   anything. */
const ALL = "\u0000all";
let activeTab = null;
const unread = new Map();
const counts = new Map();
const entries = [];
const rows = new Map(); // entry id -> <tr>, so a correction can update in place

function fmtTime(iso) {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleTimeString([], { hour12: false });
}

function renderRoster() {
  rosterEl.innerHTML = "";
  for (const station of roster) {
    const count = counts.get(station.callsign) || 0;
    const item = document.createElement("div");
    item.className = "roster-item";
    if (count > 0) item.classList.add("checked-in");
    if (filter === station.callsign) item.classList.add("selected");
    item.innerHTML =
      `<span><span class="call">${station.callsign}</span>` +
      (station.name ? ` <span class="name">${station.name}</span>` : "") +
      `</span><span class="count">${count || ""}</span>`;
    item.onclick = () => setFilter(filter === station.callsign ? null : station.callsign);
    rosterEl.appendChild(item);
  }
}

function setFilter(callsign) {
  filter = callsign;
  applyFilters();
  renderRoster();
}

/* The tab decides which rows exist; the callsign filter dims within them, so
   "everything KJ6TUV said on the repeater" still works. */
function applyFilters() {
  clearFilterBtn.hidden = !filter;
  // On a per-source tab the badge just repeats the tab name on every line;
  // it earns its place only in the combined view.
  const scoped = sources.length > 1 && activeTab && activeTab !== ALL;
  log.classList.toggle("scoped", !!scoped);
  for (const row of log.children) {
    row.classList.toggle("hidden-source", !rowInActiveTab(row));
    row.classList.toggle(
      "dim",
      !!filter && row.dataset.callsign !== filter
    );
  }
  updateStats();
}

function rowInActiveTab(row) {
  if (sources.length < 2 || activeTab === null || activeTab === ALL) return true;
  return (row.dataset.source || "") === activeTab;
}

/* One tab per receiver, plus a combined view.
   Each tab carries its own health dot, so a dead receiver is visible even
   while you are looking at a different frequency, and an unread count, so a
   check-in on the other tab does not go unnoticed. */
function renderTabs() {
  const bar = document.getElementById("tabs");
  if (!bar) return;
  if (sources.length < 2) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  if (activeTab === null) activeTab = sources[0];

  const tabs = [...sources, ALL];
  bar.innerHTML = tabs
    .map((name) => {
      const isAll = name === ALL;
      const state = isAll ? null : (sourceHealth[name] || {}).state || "unknown";
      const count = unread.get(name) || 0;
      const dot = isAll
        ? ""
        : `<span class="dot ${state === "ok" ? "live" : state}"></span>`;
      return (
        `<div class="tab${activeTab === name ? " active" : ""}" data-tab="${name}">` +
        dot +
        `<span>${isAll ? "All" : escapeHTML(name)}</span>` +
        `<span class="unread"${count ? "" : " hidden"}>${count}</span></div>`
      );
    })
    .join("");

  for (const tab of bar.querySelectorAll(".tab")) {
    tab.onclick = () => selectTab(tab.dataset.tab);
  }
}

function selectTab(name) {
  activeTab = name;
  unread.set(name, 0);
  applyFilters();
  renderTabs();
  if (autoScroll) scrollToBottom();
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function callsignCellHTML(entry) {
  if (!entry.matched) {
    const why = entry.candidate
      ? `heard “${entry.candidate}”`
      : (entry.unmatched_reason || "").replace(/_/g, " ");
    return `<span class="call-text">UNMATCHED</span><span class="reason">${why}</span>`;
  }
  const name = entry.operator_name ? `<span class="name">${entry.operator_name}</span>` : "";
  const mark = entry.corrected
    ? `<span class="corrected-mark">✓ corrected${entry.original_callsign ? ` from ${entry.original_callsign}` : ""}</span>`
    : entry.via_alias
      ? `<span class="corrected-mark">✓ learned</span>`
      : "";
  return `<span class="call-text">${entry.matched_callsign}</span>${name}${mark}`;
}

function paintRow(row, entry) {
  row.dataset.callsign = entry.matched_callsign || "";
  row.dataset.timestamp = entry.timestamp;
  row.classList.toggle("unmatched", !entry.matched);
  if (filter) row.classList.toggle("dim", row.dataset.callsign !== filter);

  const pct = Math.round((entry.confidence || 0) * 100);
  const late = entry.late
    ? `<span class="late-mark" title="Transcribed from the backlog after the transmission had passed">late</span>`
    : "";
  const src = entry.source
    ? `<span class="src-mark" data-source="${entry.source}">${entry.source}</span>`
    : "";
  row.dataset.source = entry.source || "";
  row.innerHTML =
    `<td class="time">${fmtTime(entry.timestamp)}${src}${late}</td>` +
    `<td class="call" title="Click to set the callsign">${callsignCellHTML(entry)}</td>` +
    `<td class="text"></td>` +
    `<td class="conf"><span class="bar${pct < 60 ? " low" : ""}"><span style="width:${pct}%"></span></span>${pct}%</td>`;
  // Transcript text is model output, so set it as text rather than markup.
  row.querySelector(".text").textContent = entry.raw_text;
  row.querySelector(".call").onclick = () => openCorrection(row, entry);
  const badge = row.querySelector(".src-mark");
  if (badge) {
    badge.onclick = (event) => {
      event.stopPropagation();
      selectTab(badge.dataset.source);
    };
  }
}

function addEntry(entry, isNew) {
  entries.push(entry);
  if (entry.matched) {
    counts.set(entry.matched_callsign, (counts.get(entry.matched_callsign) || 0) + 1);
  }

  const row = document.createElement("tr");
  if (isNew) row.classList.add("new");
  paintRow(row, entry);
  rows.set(entry.id, row);

  // A clip recovered from the disk backlog arrives after later ones but was
  // spoken earlier, so it goes where it belongs rather than at the bottom.
  const after = [...log.children].find(
    (r) => (r.dataset.timestamp || "") > entry.timestamp
  );
  row.dataset.timestamp = entry.timestamp;
  if (after) log.insertBefore(row, after);
  else log.appendChild(row);

  if (isNew && entry.source && sources.length > 1 && activeTab !== entry.source
      && activeTab !== ALL) {
    unread.set(entry.source, (unread.get(entry.source) || 0) + 1);
    renderTabs();
  }
  row.classList.toggle("hidden-source", !rowInActiveTab(row));

  emptyEl.hidden = true;
  updateStats();
  renderRoster();
  // Only chase the bottom for a row the operator can actually see.
  if (autoScroll && rowInActiveTab(row)) scrollToBottom();
}

/* Corrections -------------------------------------------------------------
   Click a callsign cell, pick the right station. The server fixes the log
   line, records the correction, and teaches the matcher the alias so the next
   transmission from that station matches on its own. */

function openCorrection(row, entry) {
  const cell = row.querySelector(".call");
  if (cell.querySelector("select")) return; // already open

  const select = document.createElement("select");
  select.innerHTML =
    `<option value="">— pick station —</option>` +
    roster
      .map(
        (s) =>
          `<option value="${s.callsign}"${s.callsign === entry.matched_callsign ? " selected" : ""}>` +
          `${s.callsign}${s.name ? ` — ${s.name}` : ""}</option>`
      )
      .join("");

  cell.innerHTML = "";
  cell.appendChild(select);
  select.focus();

  const close = () => paintRow(row, findEntry(entry.id) || entry);
  select.onchange = () => (select.value ? submitCorrection(entry.id, select.value) : close());
  select.onblur = () => setTimeout(close, 150);
  select.onkeydown = (e) => {
    if (e.key === "Escape") close();
  };
}

function findEntry(id) {
  return entries.find((e) => e.id === id);
}

async function submitCorrection(entryId, callsign) {
  try {
    const res = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entry_id: entryId, callsign }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "correction failed");
    applyCorrection(data.entry);
    showToast(
      data.learned
        ? `${callsign} set — learned “${data.alias}”, future transmissions will match automatically`
        : `${callsign} set`
    );
  } catch (err) {
    showToast(`Could not save: ${err.message}`);
    const entry = findEntry(entryId);
    if (entry) paintRow(rows.get(entryId), entry);
  }
}

/* Applied both to our own corrections and to ones broadcast from another
   dashboard, so two operators never see different logs. */
function applyCorrection(updated) {
  const index = entries.findIndex((e) => e.id === updated.id);
  const previous = index >= 0 ? entries[index] : null;
  if (previous && previous.matched) {
    counts.set(previous.matched_callsign, Math.max(0, (counts.get(previous.matched_callsign) || 1) - 1));
  }
  if (index >= 0) entries[index] = updated;
  counts.set(updated.matched_callsign, (counts.get(updated.matched_callsign) || 0) + 1);

  const row = rows.get(updated.id);
  if (row) paintRow(row, updated);
  updateStats();
  renderRoster();
}

/* Health banner ------------------------------------------------------------
   The pipeline can be up and producing nothing — SDR app closed, squelch shut,
   sink repointed. The banner says so, and beeps, because the operator is
   usually looking at the radio rather than the screen. */

const banner = document.getElementById("banner");
const alertBtn = document.getElementById("alertBtn");
let alertsOn = localStorage.getItem("netstt.alerts") !== "off";
let lastHealthState = "ok";

alertBtn.classList.toggle("active", alertsOn);
alertBtn.textContent = `Alerts: ${alertsOn ? "on" : "off"}`;
alertBtn.onclick = () => {
  alertsOn = !alertsOn;
  localStorage.setItem("netstt.alerts", alertsOn ? "on" : "off");
  alertBtn.classList.toggle("active", alertsOn);
  alertBtn.textContent = `Alerts: ${alertsOn ? "on" : "off"}`;
  if (alertsOn) beep(660, 0.08); // confirm the browser will actually make noise
};

const sourceHealth = {};

function renderHealth(health) {
  if (!health) return;
  const state = health.state || "ok";
  if (health.sources) {
    Object.assign(sourceHealth, health.sources);
    if (sources.length === 0) sources = Object.keys(health.sources);
    renderTabs();
  }

  dot.classList.remove("warning", "error");
  if (state !== "ok") dot.classList.add(state);

  if (state === "ok" || !health.issues || health.issues.length === 0) {
    banner.hidden = true;
    banner.className = "";
  } else {
    banner.hidden = false;
    banner.className = state;
    const label = state === "error" ? "Not logging" : "Check audio";
    const backlog =
      health.spill_pending > 0
        ? `<span class="backlog">${health.spill_pending} clip(s) catching up</span>`
        : "";
    banner.innerHTML =
      `<span class="label">${label}</span>` +
      `<ul>${health.issues.map((i) => `<li>${escapeHTML(i)}</li>`).join("")}</ul>` +
      backlog;
  }

  // Beep only on the transition into a worse state, not every poll.
  const rank = { ok: 0, warning: 1, error: 2 };
  if (alertsOn && rank[state] > rank[lastHealthState]) {
    beep(state === "error" ? 300 : 520, 0.25);
    if (state === "error") setTimeout(() => beep(300, 0.25), 350);
  }
  lastHealthState = state;
}

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/* A short tone via Web Audio, so there is no asset to ship or path to get
   wrong on a machine that has never been online. */
let audioCtx = null;
function beep(frequency, seconds) {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = frequency;
    osc.type = "sine";
    gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.25, audioCtx.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + seconds);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + seconds);
  } catch (err) {
    /* Autoplay policy blocks audio until the page is clicked; the banner is
       still the primary signal, so this is not worth surfacing. */
  }
}

let toastTimer = null;
function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toast.hidden = true), 4000);
}

function updateStats() {
  const shown =
    sources.length > 1 && activeTab && activeTab !== ALL
      ? entries.filter((e) => e.source === activeTab)
      : entries;
  const stations = new Set(shown.filter((e) => e.matched).map((e) => e.matched_callsign));
  const scope =
    sources.length > 1 && activeTab && activeTab !== ALL ? ` on ${activeTab}` : "";
  stats.textContent =
    `${shown.length} transmission${shown.length === 1 ? "" : "s"} · ` +
    `${stations.size} station${stations.size === 1 ? "" : "s"}${scope}`;
}

scrollBtn.onclick = () => {
  autoScroll = !autoScroll;
  scrollBtn.classList.toggle("active", autoScroll);
  scrollBtn.textContent = `Auto-scroll: ${autoScroll ? "on" : "off"}`;
};
clearFilterBtn.onclick = () => setFilter(null);
exportBtn.onclick = async () => {
  exportBtn.disabled = true;
  try {
    const res = await fetch("/api/export", { method: "POST" });
    const data = await res.json();
    exportBtn.textContent = `Saved ${data.files.length} files`;
  } catch (err) {
    exportBtn.textContent = "Export failed";
  }
  setTimeout(() => {
    exportBtn.textContent = "Export log";
    exportBtn.disabled = false;
  }, 3000);
};

async function loadHistory() {
  const res = await fetch("/api/history");
  const data = await res.json();
  roster = data.roster;
  sources = data.sources || [];
  if (entries.length === 0) {
    for (const entry of data.entries) addEntry(entry, false);
  }
  renderRoster();
  renderTabs();
  applyFilters();
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    dot.classList.add("live");
    conn.textContent = "live";
    // Keeps the socket from idling out behind a proxy.
    ws.pingTimer = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000);
  };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "entry") {
      addEntry(msg.entry, true);
    } else if (msg.type === "correction") {
      applyCorrection(msg.entry);
    } else if (msg.type === "health") {
      renderHealth(msg.health);
    } else if (msg.type === "history") {
      if (entries.length === 0) for (const entry of msg.entries) addEntry(entry, false);
      renderHealth(msg.health);
    }
  };
  ws.onclose = () => {
    clearInterval(ws.pingTimer);
    dot.classList.remove("live");
    conn.textContent = "reconnecting…";
    renderHealth({
      state: "error",
      issues: ["Lost the connection to the app — this log is not updating"],
    });
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

loadHistory().then(connect);
