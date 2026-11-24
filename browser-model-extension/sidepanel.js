const $ = (id) => document.getElementById(id);
const DEFAULTS = { baseUrl: "http://192.168.50.213:1234/v1", model: "local-model", apiKey: "", maxSteps: 15, trusted: false, auto: false, vision: true, simple: false };

async function loadSettings() {
  const s = await chrome.storage.local.get(DEFAULTS);
  $("baseUrl").value = s.baseUrl;
  $("apiKey").value = s.apiKey;
  $("maxSteps").value = s.maxSteps;
  $("trusted").checked = !!s.trusted;
  $("auto").checked = !!s.auto;
  $("vision").checked = s.vision !== false;
  $("simple").checked = !!s.simple;
  // Populate models list with the saved value already in storage (no DOM restore needed).
  await populateModels(true);
}

async function populateModels(preserveValue) {
  const select = $("model");

  // Grab DOM value before wiping options (when called from Save/Test with preserveValue=true).
  const domValue = select.value ? select.value.trim() : null;

  // Authoritative saved model always comes from storage.
  const stored = await chrome.storage.local.get({ model: DEFAULTS.model });
  const savedModel = stored.model || null;

  const base = $("baseUrl").value.trim();
  const key = ($("apiKey").value || "").trim();
  select.innerHTML = "";
  if (!base) {
    select.add(Object.assign(document.createElement("option"), { value: "", text: "Enter endpoint first" }));
    return;
  }
  select.add(Object.assign(document.createElement("option"), { value: "", text: "Detect models…" }));

  try {
    const url = base.endsWith("/models") ? base : base + "/models";
    const res = await fetch(url, { headers: key ? { Authorization: "Bearer " + key } : {} });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    select.innerHTML = "";

    // Use the provider's detected list of models.
    const models = (data.data || []).map(m => m.id).sort();
    if (models.length === 0) throw new Error("none found");

    let target = savedModel;
    if (preserveValue && domValue && domValue !== savedModel) {
      // User selected a different model after save — honor that choice.
      target = domValue;
    }

    if (models.includes(target)) {
      // Target is in the detected list — build normal dropdown and select it.
      models.forEach(id => {
        const opt = document.createElement("option");
        opt.value = id; opt.text = id;
        if (id === target) opt.selected = true;
        select.add(opt);
      });
    } else if (target) {
      // Saved/custom model isn't in provider list — add it as a selectable option at top.
      const customOpt = document.createElement("option");
      customOpt.value = target;
      customOpt.text = `★ ${target} (custom)`;
      customOpt.selected = true;
      select.add(customOpt);
      models.forEach(id => {
        select.add(Object.assign(document.createElement("option"), { value: id, text: id }));
      });
    } else {
      // No saved model — just build the normal list.
      models.forEach(id => {
        select.add(Object.assign(document.createElement("option"), { value: id, text: id }));
      });
    }

  } catch (e) {
    // Detection failed — restore the saved model as a selectable option so cfg() still works.
    select.innerHTML = "";
    const errOpt = document.createElement("option");
    errOpt.value = ""; errOpt.text = `⚠ Detect failed (${e.message})`; errOpt.disabled = true;
    select.add(errOpt);

    let target = savedModel;
    if (preserveValue && domValue && domValue !== savedModel) target = domValue;

    if (target) {
      const opt = document.createElement("option");
      opt.value = target; opt.text = `★ ${target}`; opt.selected = true;
      select.add(opt);
    }
  }
}

function cfg() {
  return {
    baseUrl: $("baseUrl").value.trim() || DEFAULTS.baseUrl,
    model: $("model").value.trim() || DEFAULTS.model,
    apiKey: $("apiKey").value.trim(),
    maxSteps: parseInt($("maxSteps").value, 10) || 15,
    trusted: $("trusted").checked,
    auto: $("auto").checked,
    vision: $("vision").checked,
    simple: $("simple").checked
  };
}
async function saveSettings() { await chrome.storage.local.set(cfg()); }

// --- Rendering ---
const chatEl = () => $("chat");
function nearBottom() {
  const c = chatEl();
  return c.scrollHeight - c.scrollTop - c.clientHeight < 50;
}
function stickScroll(wasAtBottom) {
  if (wasAtBottom) chatEl().scrollTop = chatEl().scrollHeight;
}

function buildImage(dataUrl) {
  // Capture whether we're at the bottom now, so we can re-scroll once the image
  // finishes loading and expands (its height is 0 until then).
  const stick = nearBottom();
  const wrap = document.createElement("div");
  wrap.className = "msg shotwrap";
  const cap = document.createElement("div");
  cap.className = "shotcap";
  cap.textContent = "📷 screenshot";
  wrap.appendChild(cap);
  if (dataUrl) {
    const img = document.createElement("img");
    img.className = "shot";
    img.title = "Click to open full size";
    img.addEventListener("load", () => { if (stick) chatEl().scrollTop = chatEl().scrollHeight; });
    // Open full-size via a blob URL — Chrome blocks navigating a tab to a data: URL.
    img.addEventListener("click", async () => {
      try {
        const blob = await (await fetch(dataUrl)).blob();
        window.open(URL.createObjectURL(blob), "_blank");
      } catch (e) {}
    });
    img.src = dataUrl; // set after listeners so onload always fires
    wrap.appendChild(img);
  } else {
    const note = document.createElement("div");
    note.className = "shotcap";
    note.textContent = "(older screenshot cleared to save space)";
    wrap.appendChild(note);
  }
  return wrap;
}

// Live working-strip label reflecting the current action / step.
let workingStep = 0;
function setWorkingLabel(s) {
  const el = $("workLabel");
  if (!el) return;
  s = String(s).replace(/[.…]+$/, ""); // trailing dots come from the animated ellipsis
  el.textContent = s.charAt(0).toUpperCase() + s.slice(1);
}
function updateWorkingFrom(entry) {
  if (entry.kind === "user") { workingStep = 0; setWorkingLabel("Working"); }
  else if (entry.kind === "tool") { workingStep++; setWorkingLabel(prettyTool(entry.text) + " · step " + workingStep); }
  else if (entry.kind === "think" || entry.kind === "assistant") { setWorkingLabel("Thinking"); }
}

function renderEntry(entry) {
  updateWorkingFrom(entry);
  const { kind, text } = entry;
  if (kind === "image") { chatEl().appendChild(buildImage(entry.dataUrl)); return; }
  const div = document.createElement("div");
  if (kind === "user") { div.className = "msg user"; div.textContent = text; }
  else if (kind === "think" || kind === "assistant") { div.className = "msg assistant"; div.textContent = text; }
  else if (kind === "done") { div.className = "msg final"; div.textContent = text; }
  else if (kind === "tool") { div.className = "msg action"; div.textContent = prettyTool(text); }
  else if (kind === "result") { div.className = "status result"; div.textContent = prettyResult(text); }
  else if (kind === "error") { div.className = "status error"; div.textContent = text; }
  else { div.className = "status result"; div.textContent = text; }
  chatEl().appendChild(div);
}

let renderedCount = 0;
function renderAll(transcript) {
  const wasAtBottom = nearBottom();
  chatEl().innerHTML = "";
  if (!transcript || transcript.length === 0) {
    const g = document.createElement("div");
    g.className = "msg assistant";
    g.textContent = "Hi! Tell me what to do in the browser and I'll narrate each step as I go.";
    chatEl().appendChild(g);
    renderedCount = 0;
    return;
  }
  transcript.forEach(renderEntry);
  renderedCount = transcript.length;
  stickScroll(wasAtBottom || true);
}
function renderNew(transcript) {
  if (!transcript) return;
  if (transcript.length < renderedCount) { renderAll(transcript); return; } // reset/new chat
  const wasAtBottom = nearBottom();
  for (let i = renderedCount; i < transcript.length; i++) renderEntry(transcript[i]);
  renderedCount = transcript.length;
  stickScroll(wasAtBottom);
}

// Turn a raw tool result (JSON) into a readable line, e.g. show what was clicked.
function prettyResult(text) {
  try {
    const o = JSON.parse(text);
    if (o && o.error) return "⚠ " + o.error;
    if (o && o.info) return o.info;
  } catch (e) {}
  return text.length > 160 ? text.slice(0, 160) + " …" : text;
}

function prettyTool(text) {
  const m = text.match(/^(\w+)\((.*)\)$/s);
  if (!m) return text;
  const name = m[1];
  let args = {};
  try { args = JSON.parse(m[2] || "{}"); } catch (e) {}
  switch (name) {
    case "get_page": return "reading the page…";
    case "navigate": return "opening " + (args.url || "");
    case "click": return "clicking element " + args.index;
    case "click_at": return `clicking at (${args.x}, ${args.y}) by sight`;
    case "drag": return `dragging ${args.from} onto ${args.to}`;
    case "type": return `${args.clear === false ? "appending" : "typing"} “${args.text || ""}”` + (args.submit ? " and searching" : "");
    case "select_option": return `selecting “${args.value || ""}” from the list`;
    case "scroll": return "scrolling " + (args.direction || "down");
    case "screenshot": return "taking a screenshot to look at the page…";
    default: return text;
  }
}

// --- Port (commands only; screenshots persist via the transcript) ---
let port = null;
function connect() {
  port = chrome.runtime.connect({ name: "agent" });
  port.onDisconnect.addListener(() => { port = null; });
}

function setRunning(running) {
  $("run").disabled = running;
  $("stop").disabled = !running;
  $("task").disabled = running;
  $("working").hidden = !running;
  if (!running) $("task").focus();
}

function send() {
  const task = $("task").value.trim();
  if (!task) return;
  saveSettings();
  if (!port) connect();
  $("task").value = "";
  $("task").style.height = "auto";
  setRunning(true);
  workingStep = 0; setWorkingLabel("Working");
  $("settings").open = false;
  port.postMessage({ type: "run", cfg: cfg(), task });
}

// --- Debug export: dump exactly what the model saw & did, for troubleshooting.
function msgText(m) {
  if (typeof m.content === "string") return m.content;
  if (Array.isArray(m.content)) return m.content.map((p) => p && p.type === "text" ? p.text : "[image]").join(" ");
  return "";
}
function truncate(s, n) { s = s || ""; return s.length > n ? s.slice(0, n) + " …[+" + (s.length - n) + " chars]" : s; }

async function copyDebug() {
  const s = await chrome.storage.local.get({ conversation: [], transcript: [], model: "", baseUrl: "", trusted: false, auto: false });
  const conv = s.conversation || [];
  const out = [];
  out.push("=== BROWSER AGENT DEBUG LOG ===");
  out.push(`model: ${s.model || "?"}   endpoint: ${s.baseUrl || "?"}   trusted:${!!s.trusted} auto:${!!s.auto}`);
  out.push(`conversation messages: ${conv.length}`);
  out.push("");
  for (const m of conv) {
    if (m.role === "system") { out.push("[SYSTEM] (prompt omitted)"); continue; }
    if (m.role === "user") { out.push("[USER] " + truncate(msgText(m), 2000)); }
    else if (m.role === "assistant") {
      const t = msgText(m).trim();
      if (t) out.push("[ASSISTANT] " + t);
      for (const tc of (m.tool_calls || [])) {
        const fn = tc && tc.function;
        out.push("   → CALL " + ((fn && fn.name) || "?") + "(" + ((fn && fn.arguments) || "") + ")");
      }
      if (!t && !(m.tool_calls || []).length) out.push("[ASSISTANT] (empty)");
    } else if (m.role === "tool") {
      out.push("[TOOL:" + (m.name || "?") + "] " + truncate(msgText(m), 2500));
    }
    out.push("");
  }
  const errors = (s.transcript || []).filter((e) => e.kind === "error");
  if (errors.length) {
    out.push("=== ERRORS ===");
    for (const e of errors) out.push("• " + e.text);
  }
  const dump = out.join("\n");
  const btn = $("copydebug");
  try {
    await navigator.clipboard.writeText(dump);
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = dump; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e2) {}
    ta.remove();
  }
  btn.textContent = `✓ Copied ${dump.length} chars`;
  setTimeout(() => { btn.textContent = "📋 Copy debug log"; }, 1800);
}

// --- Events ---
$("copydebug").addEventListener("click", copyDebug);
$("run").addEventListener("click", send);
$("stop").addEventListener("click", () => {
  if (!port) connect();            // the port drops when the popup closes - reconnect so Stop is delivered
  port.postMessage({ type: "stop" });
  setRunning(false);
});
$("task").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("task").addEventListener("input", () => {
  $("task").style.height = "auto";
  $("task").style.height = Math.min($("task").scrollHeight, 120) + "px";
});
$("newchat").addEventListener("click", () => { if (!port) connect(); port.postMessage({ type: "newchat" }); });
$("gear").addEventListener("click", () => { $("settings").open = !$("settings").open; });
$("baseUrl").addEventListener("change", populateModels);
$("apiKey").addEventListener("change", populateModels);
$("save").addEventListener("click", async () => { await saveSettings(); await populateModels(true); }); // preserve selected model on save
$("test").addEventListener("click", async () => { await populateModels(true); if (!port) connect(); port.postMessage({ type: "test", cfg: cfg() }); });

// Live updates: the transcript and running flag live in storage.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  if (changes.transcript) renderNew(changes.transcript.newValue || []);
  if (changes.running) setRunning(!!changes.running.newValue);
});

// --- Init ---
(async () => {
  await loadSettings();
  connect();
  const s = await chrome.storage.local.get({ transcript: [], running: false });
  renderAll(s.transcript);
  setRunning(!!s.running);
})();
