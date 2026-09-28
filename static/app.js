"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (t) => { t = Math.floor(t || 0); return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`; };
let ME = "Me";

// Minimal, safe markdown -> HTML (escapes first).
function md(src) {
  const lines = esc(src || "").split("\n");
  let html = "", inList = false;
  for (let ln of lines) {
    ln = ln.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
    const li = ln.match(/^\s*[-*] (?:\[( |x)\] )?(.*)/);
    if (li) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${li[1] === "x" ? "✅ " : li[1] === " " ? "☐ " : ""}${li[2]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    const h = ln.match(/^(#{1,3}) (.*)/);
    if (h) html += `<h${h[1].length}>${h[2]}</h${h[1].length}>`;
    else if (ln.trim()) html += `<p>${ln}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

// ---------- tabs ----------
document.querySelectorAll("nav button").forEach((b) => b.onclick = () => {
  document.querySelectorAll("nav button, .tab").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  $(`#tab-${b.dataset.tab}`).classList.add("active");
  ({ sessions: loadSessions, memory: loadFiles, actions: loadActions }[b.dataset.tab] || (() => {}))();
});

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

api("/api/health").then((h) => {
  ME = h.name;
  $("#health").innerHTML = `STT: ${esc(h.stt)} (${esc(h.stt_model)}) · LLM: ${esc(h.llm)} ${esc(h.model)} · ` +
    (h.private ? `<span class="priv">fully local</span>` : "transcripts sent to LLM API") +
    (h.keep_audio ? " · audio kept" : " · audio discarded");
  if (h.stt === "none") $("#health").innerHTML += " · <b>no STT installed</b>";
});

// ---------- live capture ----------
let ws = null, audioCtx = null, streams = [], recording = false;

async function listInputs() {
  const sel = $("#themSource"), keep = sel.value;
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput" && d.label);
  sel.querySelectorAll("option[data-dev]").forEach((o) => o.remove());
  for (const d of devs) {
    const o = document.createElement("option");
    o.value = "dev:" + d.deviceId; o.dataset.dev = 1;
    o.textContent = "Device: " + d.label;
    if (/blackhole|loopback|soundflower|aggregate/i.test(d.label)) o.textContent += "  ← call audio";
    sel.appendChild(o);
  }
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : sel.value;
}
navigator.mediaDevices?.enumerateDevices && listInputs();

async function addSource(stream, id, meter) {
  streams.push(stream);
  const src = audioCtx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(audioCtx, "pcm-worklet");
  node.port.onmessage = ({ data }) => {
    $(meter).style.height = Math.min(22, 2 + data.rms * 200) + "px";
    if (ws?.readyState !== 1) return;
    const pcm = new Uint8Array(data.pcm), pkt = new Uint8Array(pcm.length + 1);
    pkt[0] = id; pkt.set(pcm, 1);
    ws.send(pkt);
  };
  src.connect(node);
  // keep the graph pulling audio without playing it back
  const mute = audioCtx.createGain(); mute.gain.value = 0;
  node.connect(mute).connect(audioCtx.destination);
}

async function start() {
  const them = $("#themSource").value, useMic = $("#useMic").checked;
  if (!useMic && them === "none") return alert("Pick at least one audio source.");
  try {
    audioCtx = new AudioContext({ sampleRate: 16000 });
    await audioCtx.audioWorklet.addModule("/static/pcm-worklet.js");
    if (useMic) {
      await addSource(await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } }), 0, "#meterMe");
    }
    if (them === "display") {
      const d = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: { echoCancellation: false, noiseSuppression: false }, systemAudio: "include" });
      d.getVideoTracks().forEach((t) => t.stop());
      if (!d.getAudioTracks().length) throw new Error("No audio was shared. Tick 'Share audio' in the picker (share a tab for Meet/Teams web).");
      await addSource(new MediaStream(d.getAudioTracks()), 1, "#meterThem");
    } else if (them.startsWith("dev:")) {
      await addSource(await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: them.slice(4) }, echoCancellation: false, noiseSuppression: false, autoGainControl: false } }), 1, "#meterThem");
    }
  } catch (e) { cleanupAudio(); return alert("Audio error: " + e.message); }
  listInputs();

  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "start", title: $("#title").value }));
  ws.onmessage = (e) => onLive(JSON.parse(e.data));
  ws.onclose = () => { if (recording) setRecording(false); cleanupAudio(); };
  $("#transcript").innerHTML = ""; $("#suggestions").innerHTML = "";
  setRecording(true);
}

function stop() {
  cleanupAudio();
  ws?.readyState === 1 && ws.send(JSON.stringify({ type: "stop" }));
  setRecording(false);
}

function cleanupAudio() {
  streams.forEach((s) => s.getTracks().forEach((t) => t.stop())); streams = [];
  audioCtx?.close(); audioCtx = null;
}

function setRecording(on) {
  recording = on;
  $("#recDot").classList.toggle("on", on);
  $("#startBtn").hidden = on; $("#stopBtn").hidden = !on;
  $("#sayBtn").disabled = $("#refreshBtn").disabled = !on;
}

function onLive(m) {
  if (m.type === "segment") addSegment(m);
  else if (m.type === "thinking") $("#thinking").hidden = false;
  else if (m.type === "suggestions") { $("#thinking").hidden = true; addSuggestions(m); }
  else if (m.type === "answer") { $("#thinking").hidden = true; addCard("answer", m.question ? `Answer: ${m.question}` : "Say this", `<div class="say-text">${md(m.text)}</div>`); }
  else if (m.type === "status") addCard("", "Status", esc(m.text));
  else if (m.type === "summary") { addCard("say", "Session summary", md(m.session?.summary?.summary || "(empty session)")); ws.close(); }
}

function addSegment(m) {
  const box = $("#transcript");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  const who = m.speaker === "me" ? ME : m.speaker === "note" ? "Note" : "Them";
  const el = document.createElement("div");
  el.className = `seg ${m.speaker}`; el.dataset.t = m.t;
  el.innerHTML = `<span class="ts">${fmt(m.t)}</span><span class="who">${esc(who)}</span>${esc(m.text)}`;
  const after = [...box.children].find((c) => parseFloat(c.dataset.t) > m.t);
  box.insertBefore(el, after || null);
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function addCard(cls, label, html) {
  const el = document.createElement("div");
  el.className = "card " + cls;
  el.innerHTML = `<div class="label">${esc(label)}</div>${html}`;
  $("#suggestions").prepend(el);
}

function addSuggestions(s) {
  const list = (a) => `<ul>${a.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
  let html = "";
  if (s.pending_question) html += `<div class="label">They asked</div><div>${esc(s.pending_question)}</div>`;
  if (s.say_next) html += `<div class="label">Say</div><div class="say-text">${esc(s.say_next)}</div>`;
  if (s.questions_to_ask?.length) html += `<div class="label">Ask</div>${list(s.questions_to_ask)}`;
  if (s.facts?.length) html += `<div class="label">From memory</div>${list(s.facts)}`;
  if (s.flags?.length) html += `<div class="label">⚠ Watch out</div>${list(s.flags)}`;
  if (html) addCard(s.flags?.length ? "say flag" : "say", `@ ${fmt(s.t)}`, html);
}

$("#startBtn").onclick = start;
$("#stopBtn").onclick = stop;
$("#sayBtn").onclick = () => ws?.send(JSON.stringify({ type: "answer", text: "" }));
$("#refreshBtn").onclick = () => ws?.send(JSON.stringify({ type: "suggest" }));
$("#askLive").onsubmit = (e) => {
  e.preventDefault();
  const q = $("#askLiveInput").value.trim();
  if (q && ws?.readyState === 1) ws.send(JSON.stringify({ type: "answer", text: q }));
  $("#askLiveInput").value = "";
};
$("#noteForm").onsubmit = (e) => {
  e.preventDefault();
  const t = $("#noteInput").value.trim();
  if (t && ws?.readyState === 1) ws.send(JSON.stringify({ type: "note", text: t }));
  $("#noteInput").value = "";
};
document.addEventListener("keydown", (e) => {
  if (!recording || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) || e.metaKey || e.ctrlKey) return;
  if (e.key === "r") $("#sayBtn").click();
  else if (e.key === "s") $("#refreshBtn").click();
  else if (e.key === "n") { e.preventDefault(); $("#noteInput").focus(); }
});

// ---------- ask (chat over memory) ----------
const history = [];
$("#chatForm").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#chatInput").value.trim();
  if (!q) return;
  $("#chatInput").value = "";
  const log = $("#chatLog");
  log.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(q)}</div>`);
  const bot = document.createElement("div");
  bot.className = "msg bot md"; bot.textContent = "…"; log.appendChild(bot);
  const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q, history }) });
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "", text = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n"); buf = parts.pop();
    for (const p of parts) {
      const d = p.replace(/^data: /, "");
      if (d === "[DONE]") continue;
      text += JSON.parse(d).text; bot.innerHTML = md(text);
      log.scrollTop = log.scrollHeight;
    }
  }
  history.push({ role: "user", content: q }, { role: "assistant", content: text });
};

// ---------- sessions ----------
async function loadSessions() {
  const list = await api("/api/sessions");
  $("#sessionList").innerHTML = list.map((s) => `<div class="item" data-id="${esc(s.id)}">${esc(s.title)}<small>${esc(s.started.replace("T", " "))} · ${s.segments} lines${s.summarized ? "" : " · not summarized"}</small></div>`).join("") || `<p class="muted">No sessions yet.</p>`;
  $("#sessionList").querySelectorAll(".item").forEach((el) => el.onclick = () => showSession(el.dataset.id));
}
async function showSession(id) {
  document.querySelectorAll("#sessionList .item").forEach((x) => x.classList.toggle("sel", x.dataset.id === id));
  const s = await api(`/api/sessions/${id}`), sm = s.summary || {};
  const lines = s.segments.map((g) => `<div class="seg ${g.speaker}"><span class="ts">${fmt(g.t)}</span><span class="who">${esc(g.speaker === "me" ? ME : g.speaker === "note" ? "Note" : "Them")}</span>${esc(g.text)}</div>`).join("");
  const acts = (sm.action_items || []).map((a) => `- ${a.text} (${a.owner || "?"}${a.due ? ", " + a.due : ""})`).join("\n");
  $("#sessionView").innerHTML = `
    <div class="row"><h2 class="grow" style="margin:0">${esc(s.title)}</h2>
      <button id="resum">Re-summarize</button><button id="del" class="danger">Delete</button></div>
    <div class="scroll md">
      ${md(sm.summary || "_Not summarized yet._")}
      ${sm.decisions?.length ? "<h3>Decisions</h3>" + md(sm.decisions.map((d) => "- " + d).join("\n")) : ""}
      ${acts ? "<h3>Action items</h3>" + md(acts) : ""}
      ${sm.followups ? "<h3>Follow-ups</h3>" + md(sm.followups) : ""}
      ${s.memory_update?.files?.length ? `<p class="muted">Memory updated: ${esc(s.memory_update.files.join(", "))}</p>` : ""}
      <h3>Transcript</h3>${lines || '<p class="muted">Empty.</p>'}
    </div>`;
  $("#resum").onclick = async () => { $("#resum").textContent = "Working…"; await api(`/api/sessions/${id}/summarize`, { method: "POST" }); showSession(id); loadSessions(); };
  $("#del").onclick = async () => { if (confirm("Delete this session and its action items?")) { await api(`/api/sessions/${id}`, { method: "DELETE" }); $("#sessionView").innerHTML = ""; loadSessions(); } };
}
$("#uploadInput").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  $("#sessionList").insertAdjacentHTML("afterbegin", `<p class="muted">Transcribing ${esc(f.name)}…</p>`);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  const s = await r.json();
  await loadSessions(); if (s?.id) showSession(s.id);
};

// ---------- memory ----------
let currentFile = "profile.md";
async function loadFiles() {
  const files = await api("/api/memory/files");
  $("#fileList").innerHTML = files.map((f) => `<div class="item" data-path="${esc(f)}">${esc(f)}</div>`).join("");
  $("#fileList").querySelectorAll(".item").forEach((el) => el.onclick = () => openFile(el.dataset.path));
  openFile(currentFile);
}
async function openFile(path) {
  currentFile = path;
  document.querySelectorAll("#fileList .item").forEach((x) => x.classList.toggle("sel", x.dataset.path === path));
  const f = await api(`/api/memory/file?path=${encodeURIComponent(path)}`);
  $("#filePath").textContent = path; $("#fileContent").value = f.content;
}
$("#saveBtn").onclick = async () => {
  await api("/api/memory/file", { method: "PUT", body: JSON.stringify({ path: currentFile, content: $("#fileContent").value }) });
  $("#saveBtn").textContent = "Saved ✓"; setTimeout(() => $("#saveBtn").textContent = "Save", 1200);
};
$("#newFileBtn").onclick = async () => {
  let p = $("#newFile").value.trim(); if (!p) return;
  if (!p.endsWith(".md")) p += ".md";
  const title = p.split("/").pop().replace(".md", "").replace(/-/g, " ");
  await api("/api/memory/file", { method: "PUT", body: JSON.stringify({ path: p, content: `# ${title}\n\n` }) });
  $("#newFile").value = ""; currentFile = p; loadFiles();
};
$("#searchForm").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#searchInput").value.trim();
  if (!q) return loadFiles();
  const hits = await api(`/api/search?q=${encodeURIComponent(q)}`);
  $("#fileList").innerHTML = hits.map((h) => `<div class="item" data-path="${esc(h.path.replace(/^memory\//, ""))}" data-session="${h.path.startsWith("sessions/") ? 1 : ""}">${esc(h.title)}<small>${esc(h.path)} — ${esc(h.body.slice(0, 120))}…</small></div>`).join("") || `<p class="muted">No matches.</p>`;
  $("#fileList").querySelectorAll(".item").forEach((el) => el.onclick = () => {
    if (el.dataset.session) { document.querySelector('nav [data-tab="sessions"]').click(); showSession(el.dataset.path.replace(/^sessions\//, "").replace(/\.md$/, "")); }
    else openFile(el.dataset.path);
  });
};

// ---------- actions & digest ----------
async function loadActions() {
  const acts = await api("/api/actions");
  $("#actionList").innerHTML = acts.map((a) => `<label class="action"><input type="checkbox" data-id="${a.id}"><span>${esc(a.text)}<br><small>${esc(a.owner || "?")}${a.due ? " · due " + esc(a.due) : ""} · ${esc(a.created.slice(0, 10))}</small></span></label>`).join("") || `<p class="muted">Nothing open.</p>`;
  $("#actionList").querySelectorAll("input").forEach((c) => c.onchange = async () => { await api(`/api/actions/${c.dataset.id}/toggle`, { method: "POST" }); loadActions(); });
}
$("#digestBtn").onclick = async () => {
  $("#digest").innerHTML = '<p class="muted">Thinking…</p>';
  const d = await api("/api/digest");
  $("#digest").innerHTML = md(d.markdown || "_LLM unavailable._");
};
