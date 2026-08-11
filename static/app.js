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

let autoScroll = true;
let filter = null;
let roster = [];
const counts = new Map();
const entries = [];

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
  clearFilterBtn.hidden = !filter;
  for (const row of log.children) {
    row.classList.toggle("dim", !!filter && row.dataset.callsign !== filter);
  }
  renderRoster();
}

function addEntry(entry, isNew) {
  entries.push(entry);
  if (entry.matched) {
    counts.set(entry.matched_callsign, (counts.get(entry.matched_callsign) || 0) + 1);
  }

  const row = document.createElement("tr");
  row.dataset.callsign = entry.matched_callsign || "";
  if (!entry.matched) row.classList.add("unmatched");
  if (isNew) row.classList.add("new");
  if (filter && row.dataset.callsign !== filter) row.classList.add("dim");

  const who = entry.matched
    ? `${entry.matched_callsign}${entry.operator_name ? `<span class="name">${entry.operator_name}</span>` : ""}`
    : `UNMATCHED<span class="reason">${entry.candidate ? `heard “${entry.candidate}”` : entry.unmatched_reason.replace(/_/g, " ")}</span>`;

  const pct = Math.round((entry.confidence || 0) * 100);
  row.innerHTML =
    `<td class="time">${fmtTime(entry.timestamp)}</td>` +
    `<td class="call">${who}</td>` +
    `<td class="text"></td>` +
    `<td class="conf"><span class="bar${pct < 60 ? " low" : ""}"><span style="width:${pct}%"></span></span>${pct}%</td>`;
  // Transcript text is model output, so set it as text rather than markup.
  row.querySelector(".text").textContent = entry.raw_text;

  log.appendChild(row);
  emptyEl.hidden = true;
  updateStats();
  renderRoster();
  if (autoScroll) window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function updateStats() {
  const stations = new Set(entries.filter((e) => e.matched).map((e) => e.matched_callsign));
  stats.textContent = `${entries.length} transmission${entries.length === 1 ? "" : "s"} · ${stations.size} station${stations.size === 1 ? "" : "s"}`;
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
  if (entries.length === 0) {
    for (const entry of data.entries) addEntry(entry, false);
  }
  renderRoster();
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
    } else if (msg.type === "history" && entries.length === 0) {
      for (const entry of msg.entries) addEntry(entry, false);
    }
  };
  ws.onclose = () => {
    clearInterval(ws.pingTimer);
    dot.classList.remove("live");
    conn.textContent = "reconnecting…";
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

loadHistory().then(connect);
