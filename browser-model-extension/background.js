// Background service worker: runs the agent loop.
// Talks to the local model's OpenAI-compatible /chat/completions endpoint,
// receives tool calls, executes them against a locked tab, feeds results back.

// The UI is a toolbar popup (action.default_popup), so no click handler is needed.

// Keep-alive: Chrome kills idle SWs after ~30s; ping every 15s to stay alive
// during long agent runs. Uses chrome.runtime.onSuspend listener trick (keeps the
// worker marked as "used"). The ping interval itself doesn't guarantee survival,
// but combined with periodic messages it dramatically reduces random kills.
let keepAliveTimer = null;
function startKeepAlive() { if (!keepAliveTimer) keepAliveTimer = setInterval(() => { chrome.runtime.sendMessage({ type: "ping" }).catch(() => {}); }, 15000); }
function stopKeepAlive() { if (keepAliveTimer) { clearInterval(keepAliveTimer); keepAliveTimer = null; } }

const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_page",
      description: "Read the current page as structured text with interactive elements inlined as [n] markers in reading order (e.g. '[12] Submit'). Click or type into one by using n as the index. Call this first and after any navigation/click/scroll to refresh the markers.",
      parameters: { type: "object", properties: {}, required: [] }
    }
  },
  {
    type: "function",
    function: {
      name: "upload_file",
      description: "Set a file path on an <input type='file'> element so the page sees it as if you selected the file manually. index = interactive element index from get_page (must be an upload input); path = local filesystem path to the file to upload.",
      parameters: {
        type: "object",
        properties: {
          index: { type: "integer", description: "Index of the file input element" },
          path: { type: "string", description: "Local path to the file (e.g. C:\\Users\\adamh\\Documents\\resume.pdf)" }
        },
        required: ["index", "path"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_editor",
      description: "Read the full current content of the active code editor on the page (Monaco/CodeMirror). Use this after selecting a file tab in coding challenge sites instead of trying to read code from the page snapshot.",
      parameters: { type: "object", properties: {}, required: [] }
    }
  },
  {
    type: "function",
    function: {
      name: "set_editor_content",
      description: "Set the full content of a code editor on the page (Monaco/CodeMirror). Use this for coding challenges — much more reliable than clicking/tapping in editors. content = the complete text that should be written to the editor.",
      parameters: {
        type: "object",
        properties: {
          content: { type: "string", description: "Full text content to write into the code editor" }
        },
        required: ["content"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "editor_debug",
      description: "Run diagnostics to debug why code editor access is failing. Call this when set_editor_content fails or doesn't seem to work on the current page. It prints detailed info about where Monaco lives (top-level vs iframe), which origins are accessible, and what detection methods succeeded/failed. Check the browser console (F12) after calling it to see the full diagnostic output.",
      parameters: {
        type: "object",
        properties: {}
      }
    }
  },
  {
    type: "function",
    function: {
      name: "click",
      description: "Click the interactive element with the given index from the most recent get_page list.",
      parameters: {
        type: "object",
        properties: { index: { type: "integer", description: "Element index from get_page" } },
        required: ["index"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "type",
      description: "Type text into the input/textarea element with the given index. By default it clears the field first. To ADD text after existing content (e.g. completing pre-filled code), set clear=false to keep what's there and append at the end. Set submit=true to press Enter afterward.",
      parameters: {
        type: "object",
        properties: {
          index: { type: "integer" },
          text: { type: "string" },
          submit: { type: "boolean", description: "Press Enter / submit the form after typing" },
          clear: { type: "boolean", description: "Default true = replace field contents. false = keep existing text and append at the end." }
        },
        required: ["index", "text"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "select_option",
      description: "For autocomplete / combobox / dropdown fields that require picking from a list (e.g. a 'company', 'country', or 'skill' field that says 'select from the list'). Types the value into field 'index', waits for the suggestions, and clicks the matching one. Use this instead of type+click for such fields.",
      parameters: {
        type: "object",
        properties: {
          index: { type: "integer", description: "Index of the autocomplete input field" },
          value: { type: "string", description: "The option text to type and select" }
        },
        required: ["index", "value"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "drag",
      description: "Drag element 'from' and drop it onto element 'to' (both are element indices from get_page). Use for drag-and-drop lists, sliders, sortable items and drop zones.",
      parameters: {
        type: "object",
        properties: {
          from: { type: "integer", description: "Index of the element to drag" },
          to: { type: "integer", description: "Index of the element to drop onto" }
        },
        required: ["from", "to"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "navigate",
      description: "Navigate the active tab to a URL.",
      parameters: {
        type: "object",
        properties: { url: { type: "string" } },
        required: ["url"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "scroll",
      description: "Scroll up or down by roughly one screen. Scrolls the main page; if the page can't scroll, it automatically scrolls the largest inner panel (e.g. a code editor or side panel). To scroll a specific panel, pass 'index' of any element inside it.",
      parameters: {
        type: "object",
        properties: {
          direction: { type: "string", enum: ["up", "down"] },
          index: { type: "integer", description: "Optional: element index inside the panel you want to scroll" }
        },
        required: ["direction"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "screenshot",
      description: "Capture the visible page as an image with a red coordinate grid drawn on it (labels are CSS pixels). Use this to SEE the page and to find where to click things that are hard to click by index (icons, canvas, image maps, custom widgets). After looking, use click_at with coordinates read from the grid.",
      parameters: { type: "object", properties: {}, required: [] }
    }
  },
  {
    type: "function",
    function: {
      name: "click_at",
      description: "Click at an exact pixel location. x and y are CSS pixels in the coordinate system shown by the red grid on the latest screenshot. Use this for elements that are not in the get_page list or that clicking by index failed to hit. Always take a screenshot first so you can read the coordinates.",
      parameters: {
        type: "object",
        properties: {
          x: { type: "integer", description: "Horizontal pixel coordinate from the screenshot grid" },
          y: { type: "integer", description: "Vertical pixel coordinate from the screenshot grid" }
        },
        required: ["x", "y"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "done",
      description: "Call when the task is complete. Provide the final answer / summary for the user.",
      parameters: {
        type: "object",
        properties: { answer: { type: "string" } },
        required: ["answer"]
      }
    }
  }
];

const SYSTEM_PROMPT =
  "You control the user's open browser tab through tools, in a chat. Be conversational. " +
  "Before each tool call, write ONE short first-person sentence of what you're about to do (this is your message content). " +
  "You are already connected to the current tab: when the user says 'this page/site/repo/here' or asks about what they're viewing, " +
  "call get_page immediately — never ask for a URL unless get_page shows an unrelated page. " +
  "Work one tool at a time. Call get_page before interacting; element indices change after navigation, clicks, or scrolling, " +
  "so only use indices from the latest get_page. On dynamic sites the list changes often — if a tool says an element is " +
  "'stale', call get_page again and use the new indices. " +
  "To submit a form or dialog, click the button INSIDE that same form/dialog (e.g. its Submit/Send/Save button), not a " +
  "similarly-named link elsewhere on the page such as in the navigation bar. " +
  "If a field says to pick from a list / is an autocomplete (e.g. company, country, skill), use select_option(index, value) — " +
  "typing alone won't satisfy it; a form won't save until such fields are properly selected. " +
  "If an unsaved form is blocking you (e.g. a 'Discard changes?' prompt keeps appearing), either finish and save that form " +
  "first, or discard it — don't keep retrying the same blocked action. " +
  "Prefer navigating straight to a search-results URL over typing in a site's search box (more reliable); " +
  "e.g. Facebook Marketplace: https://www.facebook.com/marketplace/search/?query=TERMS (optionally &minPrice=&maxPrice=). " +
  "Matching questions: click the box/slot to reveal choices, then get_page and click the right choice. " +
  "Text-box/textarea questions: compose the answer, type it in by index, then click Submit (often disabled until you type). " +
  "To ADD to existing text (e.g. finishing pre-written code) use type with clear=false to append. " +
  "For coding challenge editors (Monaco/CodeMirror/etc.): IMPORTANT — get_page() cannot read editor content (you'll only see line numbers, not code). " +
  "The correct workflow is: click the file tab to select it → call get_editor() to read the full file contents → figure out what changes to make → " +
  "call set_editor_content() with the complete updated file text. Never try to edit code by clicking or typing inside the editor — " +
  "those methods don't work reliably; always use get_editor/set_editor_content. " +
  "Answer a question before scrolling past it; scroll only to see more. " +
  "Call done when finished. Be efficient; don't loop.";

// Appended only when Vision mode is on (a vision-capable model is loaded). Text-
// only models can't use images, so this guidance and the image tools are omitted.
const VISION_ADDON =
  " You can SEE the page: call screenshot to get an image where every clickable element is outlined with a red NUMBER — " +
  "then click by that number. Take a screenshot proactively when the page is visual, the get_page element list looks " +
  "incomplete or confusing, or you can't find an expected control. Use click_at(x,y) only for a target that has no number.";

// Weaker models often emit sloppy tool arguments (numbers as strings, odd
// direction words, non-string text). Normalize them so the call still works.
function coerceArgs(a) {
  if (!a || typeof a !== "object") return {};
  for (const k of ["index", "from", "to", "x", "y"]) {
    if (typeof a[k] === "string" && a[k].trim() !== "" && !isNaN(a[k])) a[k] = Number(a[k]);
  }
  if (typeof a.direction === "string") a.direction = /up|top|back/i.test(a.direction) ? "up" : "down";
  if (a.text != null && typeof a.text !== "string") a.text = String(a.text);
  if (a.value != null && typeof a.value !== "string") a.value = String(a.value);
  if (typeof a.clear === "string") a.clear = !/^(false|no|0)$/i.test(a.clear);
  if (typeof a.submit === "string") a.submit = /^(true|yes|1)$/i.test(a.submit);
  return a;
}

const KNOWN_TOOLS = ["get_page", "get_editor", "set_editor_content", "editor_debug", "upload_file", "click_at", "click", "type", "select_option", "drag", "navigate", "scroll", "screenshot", "done"];

// Dangerous button text patterns that deserve extra caution. If the model is about
// to click an element whose label matches one of these, log a visible warning so
// the user knows it's doing something destructive (they can still hit Stop).
const DANGEROUS_PATTERNS = /delete|cancel\s+subscription|remove|unfollow|block|unsubscribe|log\s*out|sign\s*out|close\s+account|deactivate|discard/i;

// After navigation/click/type that may load a new page, wait for the DOM to stabilize:
// either page load finishes or content stops changing for a short window.
const waitUntilStable = async (tabId) => {
  // Guard against race: if tab is already "complete", don't rely solely on the listener
  // because it only fires on state transitions. Use a shorter timeout if already loaded.
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  const alreadyComplete = !!(tab && tab.status === "complete");

  const loadDone = new Promise((resolve) => {
    let done = false;
    const listener = (id, info) => { if (!done && id === tabId && info.status === "complete") cleanup(); };
    const cleanup = () => { if (done) return; done = true; clearTimeout(timer); chrome.tabs.onUpdated.removeListener(listener); resolve(); };
    const timer = setTimeout(cleanup, alreadyComplete ? 2000 : 6000);
    // An already-"complete" tab (single-page apps) gets no load event: don't
    // resolve at once (that skipped the DOM-quiet check and the next read saw a
    // stale page) - let the quiet poller decide, capped at 2s by the timer.
    if (!alreadyComplete) chrome.tabs.onUpdated.addListener(listener);
  });

  let finished = false;   // stops the poller once the race is decided (it used to poll forever)
  const quietDone = new Promise((resolve) => {
    let lastLen = -1, unchanged = 0, polls = 0;
    const check = () => {
      if (finished || ++polls > 120) { resolve(); return; }
      chrome.tabs.get(tabId).then(async (t) => {
        if (!t || t.status === "loading") { setTimeout(check, 250); return; }
        try {
          const bodyResult = await chrome.scripting.executeScript({ target: { tabId }, func: () => document.body ? document.body.innerText.length : 0 }).catch(() => []);
          const result = bodyResult && bodyResult.length > 0 ? bodyResult[0].result : 0;
          if (result !== lastLen) { unchanged = 0; } else { unchanged++; }
          lastLen = result;
        } catch (e) { unchanged++; }
        if (unchanged >= 3) resolve(); else setTimeout(check, 180);
      }).catch(() => resolve());
    };
    check();
  });

  await Promise.race([loadDone, quietDone]);
  finished = true;
};

// Some (esp. smaller) models emit tool calls as PLAIN TEXT instead of using the
// structured tool-calling format — e.g. "get_page", `click(5)`, or a JSON blob.
// Parse those so the agent still runs the action instead of stalling.
function parseTextToolCall(content) {
  if (!content || typeof content !== "string") return null;
  let s = content.trim();
  const wrap = s.match(/<tool_call>\s*([\s\S]*?)<\/tool_call>/i) || s.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (wrap) s = wrap[1].trim();

  // JSON form: {"name":"click","arguments":{"index":5}} or {"tool":"click","index":5}
  const jm = s.match(/\{[\s\S]*\}/);
  if (jm) {
    try {
      const o = JSON.parse(jm[0]);
      let name = o.name || o.tool || o.action || (o.function && o.function.name);
      let args = o.arguments || o.args || o.parameters || o.params || (o.function && o.function.arguments);
      if (typeof args === "string") { try { args = JSON.parse(args); } catch (e) { args = {}; } }
      if (name && KNOWN_TOOLS.includes(name)) {
        if (!args || typeof args !== "object" || !Object.keys(args).length) {
          const { name: _n, tool: _t, action: _a, function: _f, arguments: _ar, args: _as, parameters: _p, params: _ps, ...rest } = o;
          args = Object.keys(rest).length ? rest : {};
        }
        return { name, args: args || {} };
      }
    } catch (e) {}
  }

  // Function-call form: name(...) with JSON or positional args
  const fm = s.match(/\b(get_page|get_editor|set_editor_content|editor_debug|upload_file|click_at|click|type|select_option|drag|navigate|scroll|screenshot|done)\s*\(([\s\S]*?)\)/i);
  if (fm) {
    const name = fm[1].toLowerCase();
    const inner = fm[2].trim();
    let args = {};
    if (inner) {
      try {
        args = JSON.parse(inner.startsWith("{") ? inner : "{" + inner + "}");
      } catch (e) {
        const parts = inner.split(",").map((p) => p.trim().replace(/^["']|["']$/g, ""));
        if (name === "scroll") args = { direction: parts[0] };
        else if (name === "click") args = { index: parts[0] };
        else if (name === "click_at") args = { x: parts[0], y: parts[1] };
        else if (name === "drag") args = { from: parts[0], to: parts[1] };
        else if (name === "type") args = { index: parts[0], text: parts.slice(1).join(", ") };
        else if (name === "select_option") args = { index: parts[0], value: parts.slice(1).join(", ") };
        else if (name === "set_editor_content") args = { content: parts.join(", ") };
        else if (name === "upload_file") args = { index: parts[0], path: parts.slice(1).join(", ") };
        else if (name === "navigate") args = { url: parts[0] };
        else if (name === "done") args = { answer: parts.join(", ") };
      }
    }
    return { name, args };
  }

  // Bare tool name (short content only, to avoid misreading prose)
  const words = s.toLowerCase().replace(/[^\w]/g, " ").trim().split(/\s+/);
  if (words.length <= 3) {
    for (const t of KNOWN_TOOLS) if (words.includes(t)) return { name: t, args: {} };
  }
  return null;
}

// Run ownership: every Run / Stop / New chat bumps `currentRun`. A run only
// proceeds while the ticket it was given is still current, so a Stop that
// arrives at ANY time (even while the run is queued) cancels it for good.
let currentRun = 0;
let runChain = Promise.resolve();   // runs are strictly queued on this chain
let modelAbort = null;              // aborts the in-flight model request on Stop
const cancelReasons = {};           // run ticket -> why it was cancelled (first reason wins)
function cancelRuns(reason = "stop") {
  if (!(currentRun in cancelReasons)) cancelReasons[currentRun] = reason;
  // runs that never reach stopMsg (e.g. a queued run that bails early) would
  // leave entries behind: keep only recent tickets
  for (const k of Object.keys(cancelReasons)) if (+k < currentRun - 20) delete cancelReasons[k];
  currentRun++;
  if (modelAbort) { try { modelAbort.abort(); } catch (e) {} }
}

const activeTab = async () => {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab;
};

const sendToTab = (tabId, msg, timeoutMs = 30000, frameId = 0) =>
  new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (!done) { done = true; clearTimeout(t); resolve(v); } };
    const t = setTimeout(() => finish({ ok: false, error: "content script did not respond (timed out)" }), timeoutMs);
    try {
      chrome.tabs.sendMessage(tabId, msg, { frameId }, (resp) => {
        if (chrome.runtime.lastError) finish({ ok: false, error: chrome.runtime.lastError.message });
        else finish(resp);
      });
    } catch (e) {
      finish({ ok: false, error: String(e && e.message || e) });
    }
  });

const ensureContentScript = async (tabId) => {
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  } catch (e) {
    // e.g. chrome:// pages — ignore, sendMessage will report the real problem
  }
};

const waitForLoad = (tabId) =>
  new Promise((resolve) => {
    const timeout = setTimeout(() => { cleanup(); resolve(); }, 15000);
    const listener = (id, info) => { if (id === tabId && info.status === "complete") { cleanup(); resolve(); } };
    const cleanup = () => { clearTimeout(timeout); chrome.tabs.onUpdated.removeListener(listener); };
    chrome.tabs.onUpdated.addListener(listener);
  });

// ---- Trusted input via the Chrome DevTools Protocol (isTrusted: true) ----
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const rand = (a, b) => a + Math.random() * (b - a);

const cdp = (tabId, method, params) =>
  new Promise((resolve, reject) => {
    chrome.debugger.sendCommand({ tabId }, method, params || {}, (r) => {
      if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
      else resolve(r);
    });
  });

const attachDebugger = (tabId) =>
  new Promise((resolve, reject) => {
    chrome.debugger.attach({ tabId }, "1.3", () => {
      if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
      else resolve();
    });
  });

const detachDebugger = (tabId) =>
  new Promise((resolve) => { chrome.debugger.detach({ tabId }, () => resolve()); });

async function trustedClick(tabId, x, y) {
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await sleep(rand(40, 120));
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", buttons: 1, clickCount: 1 });
  await sleep(rand(40, 110));
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", buttons: 0, clickCount: 1 });
}

async function trustedClearField(tabId) {
  // Ctrl+A then Delete (modifiers bit 2 = Ctrl in CDP).
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", key: "a", code: "KeyA", windowsVirtualKeyCode: 65, modifiers: 2 });
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", key: "a", code: "KeyA", windowsVirtualKeyCode: 65, modifiers: 2 });
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", key: "Delete", code: "Delete", windowsVirtualKeyCode: 46 });
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", key: "Delete", code: "Delete", windowsVirtualKeyCode: 46 });
}

async function trustedMoveToEnd(tabId) {
  // Ctrl+End moves the caret to the end of the field (append position).
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", key: "End", code: "End", windowsVirtualKeyCode: 35, modifiers: 2 });
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", key: "End", code: "End", windowsVirtualKeyCode: 35, modifiers: 2 });
}

async function trustedType(tabId, text) {
  for (const ch of text) {
    await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", text: ch, key: ch });
    await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", key: ch });
    await sleep(Math.random() < 0.12 ? rand(180, 420) : rand(45, 130));
  }
}

// Find editor rect (Monaco textarea or CodeMirror contentEditable) inside iframe tree using CDP Runtime.evaluate.
async function findEditorRect(tabId) {
  let frameTree;
  try {
    frameTree = await cdp(tabId, "Page.getFrameTree");
  } catch (e) {
    return null;
  }

  // Collect all frame IDs recursively
  const frames = [];
  function walk(ft) {
    if (ft.frame && ft.frame.id) frames.push(ft.frame);
    for (const child of ft.childFrames || []) walk(child);
  }
  walk(frameTree);

  console.log("[TRUSTED_EDITOR] Frames:", JSON.stringify(frames.map(function(f) { try { return { id: f.id, url: new URL(f.url).hostname }; } catch(e) { return { id: f.id, url: "?" }; } })));

  // Evaluate in each frame to find editor element
  for (const frame of frames) {
    try {
      const result = await cdp(tabId, "Runtime.evaluate", {
        frameId: frame.id,
        expression: `(() => {
          // Priority 1: CodeMirror 6 contentEditable div (.cm-content is where keyboard events go).
          {
            const cmContent = document.querySelector('.cm-editor .cm-content');
            if (cmContent && cmContent.offsetParent) {
              const r = cmContent.getBoundingClientRect();
              return { type: 'codemirror6', x: r.left, y: r.top, w: r.width, h: r.height };
            }
          }

          // Priority 2: Monaco textarea (.monaco-editor textarea).
          {
            const ta = document.querySelector('.monaco-editor textarea:not([disabled])');
            if (ta && ta.offsetParent) {
              const r = ta.getBoundingClientRect();
              return { type: 'monaco', x: r.left, y: r.top, w: r.width, h: r.height };
            }
          }

          // Priority 3: Legacy CodeMirror textarea (.CodeMirror textarea).
          {
            const cmTa = document.querySelector('.CodeMirror textarea:not([disabled])');
            if (cmTa && cmTa.offsetParent) {
              const r = cmTa.getBoundingClientRect();
              return { type: 'codemirror-legacy', x: r.left, y: r.top, w: r.width, h: r.height };
            }
          }

          return null;
        })()`,
        awaitPromise: false
      });
      const rect = result.result?.value;
      if (rect && typeof rect.x === "number" && rect.w > 50 && rect.h > 20) {
        const host = (() => { try { return new URL(frame.url).hostname; } catch(e) { return "?"; } })();
        console.log("[TRUSTED_EDITOR] Found", rect.type, "in frame", frame.id, host, JSON.stringify(rect));
        return rect;
      }
    } catch (e) {
      if (!e.message.includes("SecurityError") && !e.message.includes("Access denied")) {
        console.log("[TRUSTED_EDITOR] Evaluate failed:", e.message);
      }
    }
  }

  return null;
}

// Last-resort: write to code editor via real CDP events into the iframe (works even cross-origin).
async function trustedEditorContent(tabId, text) {
  const lines = String(text || "");

  // Find editor using frame-aware Runtime.evaluate (Monaco textarea or CodeMirror contentEditable).
  let rect = await findEditorRect(tabId);

  if (!rect) {
    console.log("[TRUSTED_EDITOR] Falling back to content-script editor_rect");
    let box = await sendToTab(tabId, { action: "editor_rect" });
    if (!box || !box.x) box = { x: 320, y: 250 };
    rect = { x: box.x + (box.w || 200) * 0.4, y: box.y + (box.h || 150) * 0.4 };
  }

  // Click near center-left of the editor to focus it
  const cx = Math.round(rect.x + (rect.w || 100) * 0.3);
  const cy = Math.round(rect.y + (rect.h || 20) * 0.5);
  console.log("[TRUSTED_EDITOR] Focusing at:", { x: cx, y: cy }, "in rect:", JSON.stringify(rect));

  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseMoved", x: cx, y: cy });
  await sleep(200);

  // Single click to focus editor surface.
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mousePressed", x: cx, y: cy, button: "left", buttons: 1, clickCount: 1 });
  await sleep(80);
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseReleased", x: cx, y: cy, button: "left", buttons: 0, clickCount: 1 });
  await sleep(120);

  // Triple-click (select whole line) to reliably ensure focus lands in the editor's input element.
  // Without this, some editors keep focus on surrounding chrome (tabs, toolbar) after a single click.
  for (let i = 1; i <= 3; i++) {
    await cdp(tabId, "Input.dispatchMouseEvent", { type: "mousePressed", x: cx, y: cy, button: "left", buttons: 1, clickCount: i });
    await sleep(70);
    await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseReleased", x: cx, y: cy, button: "left", buttons: 0, clickCount: i });
    if (i < 3) await sleep(100);
  }
  await sleep(280);

  // Select all existing content.
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", key: "a", code: "KeyA", windowsVirtualKeyCode: 65, modifiers: 2 });
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", key: "a", code: "KeyA", windowsVirtualKeyCode: 65, modifiers: 2 });
  await sleep(160);
  // Delete it.
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", key: "Delete", code: "Delete", windowsVirtualKeyCode: 46 });
  await sleep(80);

  for (const ch of lines) {
    if (ch === "\n") {
      const p = { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 };
      await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", text: "\r", ...p });
      await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", ...p });
    } else if (ch === "\t") {
      const p = { key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 };
      await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", text: "\t", ...p });
      await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", ...p });
    } else {
      await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", text: ch, key: ch });
      await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", key: ch });
    }
    await sleep(rand(20, 50));
  }
}

async function trustedDrag(tabId, sx, sy, tx, ty) {
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseMoved", x: sx, y: sy });
  await sleep(120);
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mousePressed", x: sx, y: sy, button: "left", buttons: 1, clickCount: 1 });
  await sleep(120);
  const steps = 16;
  for (let i = 1; i <= steps; i++) {
    const x = sx + (tx - sx) * i / steps;
    const y = sy + (ty - sy) * i / steps;
    await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseMoved", x, y, button: "left", buttons: 1 });
    await sleep(rand(15, 35));
  }
  await sleep(80);
  await cdp(tabId, "Input.dispatchMouseEvent", { type: "mouseReleased", x: tx, y: ty, button: "left", buttons: 0, clickCount: 1 });
}

async function trustedEnter(tabId) {
  const p = { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 };
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyDown", text: "\r", ...p });
  await cdp(tabId, "Input.dispatchKeyEvent", { type: "keyUp", ...p });
}

// Read full file from code editor by evaluating JS directly in editor frames via CDP.
// This bypasses CSP and can reach Monaco/CodeMirror internals that chrome.scripting cannot.
async function readEditorViaCDP(tabId) {
  // Get all frame tree info.
  let frames;
  try { frames = await cdp(tabId, "Target.getTargets"); } catch (e) { return null; }

  for (const t of (frames && frames.targetInfos) || []) {
    if (!t.url || !/^[a-z]+:\/\//i.test(t.url)) continue;

    // Try to evaluate code inside this frame.
    const result = await cdp(tabId, "Runtime.evaluate", {
      expression: `
        (function(){
          // Monaco check.
          try{ if(window.monaco && window.monaco.editor){
            for(const ed of window.monaco.editor.getEditors()){
              try{ const m=ed.getModel(); if(m) return "MONACO:"+String(m.getValue()); }catch(_){}
            }
          }}catch(_){}

          // CodeMirror 5.
          try{ if(window.CodeMirror){
            for(const el of document.querySelectorAll(".CodeMirror")){
              try{ if(el.CodeMirror && el.CodeMirror.getValue) return "CM5:"+String(el.CodeMirror.getValue()); }catch(_){}
            }
          }}catch(_){}

          // CodeMirror 6: look for editor state in common places.
          try{
            for(const el of document.querySelectorAll(".cm-editor")){
              const s = el.cmState || el.editor?.state || el._editor?.state;
              if(s && s.doc){ let t=""; for(const l of s.doc.iterLines()) t+=l+"\n"; return "CM6:"+t.slice(0,-1); }
            }}catch(_){}

          // Try common global editor instances.
          try{
            for(const key of ["monacoEditor","editorInstance","codeEditor","currentEditor"]){
              const ed = window[key];
              if(ed && typeof ed.getValue==="function") return "GLOBAL:"+String(ed.getValue());
            }}catch(_){}

          // Last resort: hidden textarea inside editor container.
          try{ for(const ta of document.querySelectorAll(".monaco-editor textarea,.CodeMirror textarea")){
            if(ta.value) return "TA:"+ta.value; }}catch(_){}

          return null;
        })()
      `.trim(),
      contextId: (t.url ? frames.targetInfos.find(x => x.url === t.url) : undefined)?.contextId || undefined
    }).catch(() => ({ result: { type: "undefined" } }));

    const val = (result && result.result && result.result.value);
    if (val && typeof val === "string") {
      // Parse the prefix.
      const colonIndex = val.indexOf(":");
      if (colonIndex > 0) {
        return { source: val.slice(0, colonIndex), content: val.slice(colonIndex + 1) };
      }
    }
  }

  return null;
}

async function executeTool(name, args, tabId, cfg) {
  switch (name) {
    case "get_page": {
      await ensureContentScript(tabId);
      const s = await sendToTab(tabId, { action: "get_page" });
      if (s && s.error && !s.text) return { error: s.error };
      const out = { url: s.url, title: s.title, scroll: s.scroll, elements: s.count, page: s.text };
      if (s.editor_note) out.editor_note = s.editor_note;
      return out;
    }
    case "upload_file": {
      await ensureContentScript(tabId);
      const el = await sendToTab(tabId, { action: "upload_file", index: args.index, path: args.path });
      return el || { error: "Could not upload file." };
    }
    case "get_editor": {
      const attempts = [];
      const readEditorFromAllFrames = async (world) => {
        const allRes = await chrome.scripting.executeScript({
          target: { tabId, allFrames: true },
          world,
          func: () => {
            const textFromViewLines = (root) => {
              try {
                const lines = Array.from(root.querySelectorAll(".view-lines .view-line"));
                if (!lines.length) return null;
                return lines.map((line) => (line.textContent || "").replace(/\u00a0/g, " ")).join("\n");
              } catch (_) {
                return null;
              }
            };

            const deepQueryAll = (selectors) => {
              const out = [];
              const seen = new Set();
              const visitRoot = (root) => {
                if (!root || seen.has(root)) return;
                seen.add(root);
                for (const sel of selectors) {
                  try { out.push(...Array.from(root.querySelectorAll(sel))); } catch (_) {}
                }
                let all = [];
                try { all = Array.from(root.querySelectorAll("*")); } catch (_) {}
                for (const el of all) {
                  if (el.shadowRoot) visitRoot(el.shadowRoot);
                }
              };
              visitRoot(document);
              return Array.from(new Set(out));
            };

            const visibleText = (el) => {
              try {
                const cs = getComputedStyle(el);
                const r = el.getBoundingClientRect();
                if (cs.display === "none" || cs.visibility === "hidden" || r.width < 20 || r.height < 10) return null;
                return (el.innerText || el.textContent || "").replace(/\u00a0/g, " ");
              } catch (_) {
                return null;
              }
            };

            const codeScore = (text) => {
              text = String(text || "");
              let score = 0;
              if (/[{}<>;]/.test(text)) score += 3;
              if (/\b(const|let|var|function|class|return|querySelector|DOCTYPE|html|head|body|script|button|div)\b/.test(text)) score += 4;
              if (/\n/.test(text)) score += 2;
              if (text.length > 20) score += 1;
              return score;
            };

            const textFromLineNodes = (root) => {
              const selectors = [
                ".cm-line",
                ".view-line",
                ".ace_line",
                ".token-line",
                "[class*='line'][class*='Line']",
                "[data-line-number]",
                "[role='presentation'] > span"
              ];
              try {
                const lines = deepQueryAll(selectors).filter((el) => root === document || root.contains(el));
                if (lines.length < 1) return null;
                const text = lines.map((line) => line.textContent || "").join("\n").replace(/\u00a0/g, " ");
                return text.trim().length || lines.length ? text : null;
              } catch (_) {
                return null;
              }
            };

            // CodeMirror 6 API: look for the editorState in editor instances.
            // CM6 uses state.doc as a tree structure — iterate to get full content.
            const readCodeMirror6 = (root) => {
              try {
                const editors = root.querySelectorAll(".cm-editor");
                if (!editors.length) return null;
                for (const el of editors) {
                  try {
                    // CM6 stores editor state on the instance attached to the DOM.
                    // In many setups, a global cm namespace or editor instances live here.
                    const win = el.ownerDocument.defaultView || window;
                    const getState = () => {
                      // Try common places where CM6 editors store their state.
                      if (win.cm && typeof win.cm.getEditor === "function") return win.cm.getEditor().state;
                      if (win.__cmEditor && win.__cmEditor.state) return win.__cmEditor.state;
                      if (el.cmState || el.editor?.state || el._editor?.state) {
                        const s = el.cmState || el.editor?.state || el._editor?.state;
                        if (s.doc) return s;
                      }
                      return null;
                    };
                    const state = getState();
                    if (state && state.doc) {
                      let text = "";
                      for (const line of state.doc.iterLines()) text += line + "\n";
                      // Remove trailing newline that iterLines adds.
                      if (text.endsWith("\n")) text = text.slice(0, -1);
                      return { found: true, source: "codemirror6-api", content: text };
                    }
                  } catch (_) {}
                }
              } catch (_) {}
              return null;
            };

            const readDomEditor = () => {
              try {
                if (window.monaco && window.monaco.editor) {
                  const editors = window.monaco.editor.getEditors();
                  for (const ed of editors) {
                    try {
                      const m = ed.getModel();
                      if (m) return { found: true, source: "monaco-api", content: String(m.getValue()) };
                    } catch (_) {}
                  }
                }
              } catch (_) {}

              try {
                if (window.ace && window.ace.edit) {
                  for (const el of Array.from(document.querySelectorAll(".ace_editor"))) {
                    try {
                      const ed = window.ace.edit(el);
                      if (ed && ed.getValue) return { found: true, source: "ace-api", content: String(ed.getValue()) };
                    } catch (_) {}
                  }
                }
              } catch (_) {}

              try {
                for (const el of Array.from(document.querySelectorAll(".CodeMirror"))) {
                  if (el.CodeMirror && el.CodeMirror.getValue) {
                    return { found: true, source: "codemirror-api", content: String(el.CodeMirror.getValue()) };
                  }
                }
              } catch (_) {}

              // CodeMirror 6 API check.
              const cm6Result = readCodeMirror6(document);
              if (cm6Result && cm6Result.found) return cm6Result;

              const textareaSelectors = [
                ".monaco-editor textarea",
                ".editor-scrollable textarea",
                "[class*='monaco'] textarea",
                ".CodeMirror textarea",
                ".cm-content textarea",
                "textarea"
              ];
              for (const el of deepQueryAll(textareaSelectors)) {
                try {
                  if (el && typeof el.value === "string") {
                    return { found: true, source: `textarea:${el.tagName.toLowerCase()}`, content: el.value };
                  }
                } catch (_) {}
              }

              try {
                const monaco = deepQueryAll([".monaco-editor", ".monaco-workbench", "[class*='monaco-editor']"])[0];
                if (monaco) {
                  const visible = textFromViewLines(monaco);
                  return { found: true, source: visible === null ? "monaco-dom-empty" : "monaco-dom-visible", content: visible === null ? "" : visible };
                }
              } catch (_) {}

              try {
                const cm = deepQueryAll([".CodeMirror", ".cm-editor", ".cm-content"])[0];
                if (cm) return { found: true, source: "codemirror-dom", content: textFromLineNodes(cm) || visibleText(cm) || "" };
              } catch (_) {}

              try {
                const editable = deepQueryAll(["[contenteditable='true'][role='textbox']", "[contenteditable='true']"])[0];
                if (editable) return { found: true, source: "contenteditable", content: editable.innerText || editable.textContent || "" };
              } catch (_) {}

              try {
                const candidates = deepQueryAll([
                  "[role='code']",
                  "pre",
                  "code",
                  "[class*='code']",
                  "[class*='Code']",
                  "[class*='editor']",
                  "[class*='Editor']",
                  "[data-testid*='editor']",
                  "[data-test*='editor']"
                ]);
                let best = null;
                for (const el of candidates.slice(0, 600)) {
                  const lineText = textFromLineNodes(el);
                  const text = lineText || visibleText(el);
                  if (text === null) continue;
                  const score = codeScore(text);
                  if (!best || score > best.score || (score === best.score && text.length > best.text.length)) {
                    best = { el, text, score };
                  }
                }
                if (best && best.score >= 3) {
                  return { found: true, source: "code-dom-heuristic", content: best.text };
                }
                if (best && /editor/i.test(best.el.className || "")) {
                  return { found: true, source: "editor-dom-empty", content: best.text || "" };
                }
              } catch (_) {}

              return { found: false };
            };

            return readDomEditor();
          }
        });

        attempts.push(`${world}:${allRes.length} frame result(s)`);
        const hits = allRes.map((r) => r.result).filter((r) => r && r.found && typeof r.content === "string");
        attempts.push(`${world}:hits=${hits.map((h) => `${h.source}:${h.content.length}`).join(",") || "none"}`);
        const best = hits.find((r) => r.content.length > 0) || hits[0] || null;

        // Detect truncated reads from virtualized editors (CodeMirror6): if content has obvious signs of being cut off,
        // warn the model instead of letting it blindly overwrite with partial content.
        if (best && best.found && ["codemirror-dom", "codemirror-api", "code-dom-heuristic"].includes(best.source)) {
          const content = best.content;
          const trimmed = content.trim();

          // Check for obvious truncation patterns in code that looks like a complete file.
          let isTruncated = false;

          // HTML files: if it starts like an HTML file but doesn't have closing </html>, likely truncated.
          if (trimmed.startsWith("<!DOCTYPE html") && !/<\/html>\s*$/m.test(trimmed)) {
            isTruncated = true;
          } else if (trimmed.startsWith("<html") && !/<\/html>\s*$/m.test(trimmed)) {
            isTruncated = true;
          }

          // Any content ending mid-tag or mid-attribute.
          if (!isTruncated) {
            isTruncated = /\b(class[^"]*"[^"]|\bclass\s*=\s*["'][^"'])$/.test(trimmed) ||
                         /href=["'][^"']*$/m.test(content) ||
                         (content.trim().length > 20 && /[<\(=]\s*$/.test(trimmed));
          }

          if (isTruncated) {
            return {
              found: true,
              source: best.source,
              content: content + "\n\n!!!IMPORTANT: This editor uses virtualization — you can only see currently visible lines here. " +
                "To read the complete file: scroll DOWN inside the editor (use scroll with direction=down and index of any element in the editor area) " +
                "and call get_editor() again repeatedly until you've seen all content. Combine what you see into one complete result before writing. " +
                "DO NOT use this truncated output as-is when calling set_editor_content — it will delete everything below!"
            };
          }
        }
        return best;
      };

      try {
        const hit = await readEditorFromAllFrames("MAIN");
        if (hit) return { ok: true, content: hit.content, source: hit.source };
      } catch (e) { attempts.push(`MAIN_ERROR:${String(e && e.message || e)}`); }

      try {
        const hit = await readEditorFromAllFrames("ISOLATED");
        if (hit) return { ok: true, content: hit.content, source: hit.source };
      } catch (e) { attempts.push(`ISOLATED_ERROR:${String(e && e.message || e)}`); }

      // Content-script fallback (CodeMirror / simple editors).
      await ensureContentScript(tabId);
      const r = await sendToTab(tabId, { action: "get_editor" });
      if (r && r.ok) return r;
      attempts.push(`content_script:${JSON.stringify(r || null).slice(0, 500)}`);

      // Last resort: try CDP-based evaluation inside editor frames (bypasses CSP).
      if (!(cfg && cfg.trusted)) { await attachDebugger(tabId); }
      try { const cdpResult = await readEditorViaCDP(tabId); attempts.push(`cdp:${JSON.stringify(cdpResult || null).slice(0, 300)}`); if (cdpResult && cdpResult.source) return { ok: true, content: cdpResult.content, source: cdpResult.source }; } catch (e) { attempts.push(`cdp_error:${String(e.message||e).slice(0,150)}`); }
      try { await detachDebugger(tabId); } catch (_) {}

      return { ok: false, error: "Could not read code editor.", diag: attempts };
    }
    case "editor_debug": {
      // Diagnostic helper: tells us exactly how the page/editor is laid out and where we can/cannot reach.
      const diag = [];

      await ensureContentScript(tabId);
      const tab = await chrome.tabs.get(tabId).catch(() => ({}));
      diag.push(`PAGE_URL=${tab.url || "?"}`);

      // Content-script editor_rect.
      try {
        const rect = await sendToTab(tabId, { action: "editor_rect" });
        diag.push(`EDITOR_RECT=${JSON.stringify(rect)}`);
      } catch (e) { diag.push(`EDITOR_RECT_ERROR=${String(e.message || e)}`); }

      // CDP-based Monaco find across frames (cross-origin aware).
      try {
        const rect = await findEditorRect(tabId);
        diag.push(`CDP_EDITOR_RECT=${JSON.stringify(rect)}`);
      } catch (e) { diag.push(`CDP_EDITOR_ERROR=${String(e.message || e)}`); }

      // Top-level main-world scan for Monaco + iframe map.
      try {
        const mainResult = await chrome.scripting.executeScript({
          target: { tabId },
          world: "MAIN",
          func: () => {
            const out = { monacoTop: false, editorCountTop: 0, frames: [] };
            // Map iframes.
            try {
              for (const f of Array.from(document.querySelectorAll("iframe")).slice(0, 25)) {
                let src = ""; let origin = "?"; let accessible = false;
                try { src = f.src || ""; } catch (_) {}
                try { origin = new URL(src).origin || "?"; } catch (_) {}
                try { const w = f.contentWindow; const d = f.contentDocument || w.document; accessible = !!d.body; } catch (_) {}
                out.frames.push({ src, origin, accessible });
              }
            } catch (_) {}
            // Monaco at top?
            try {
              if (window.monaco && window.monaco.editor) {
                const editors = window.monaco.editor.getEditors();
                for (const ed of editors) {
                  try {
                    const m = ed.getModel();
                    if (m) { out.monacoTop = true; out.editorCountTop++; break; }
                  } catch (_) {}
                }
              }
            } catch (_) {}
            return out;
          }
        });
        const result = mainResult && mainResult.length > 0 ? mainResult[0].result : null;
        diag.push(`MAIN_WORLD_SCAN=${JSON.stringify(result)}`);
      } catch (e) { diag.push(`MAIN_WORLD_SCAN_ERROR=${String(e.message || e)}`); }

      // All-frames scan for Monaco.
      try {
        const allRes = await chrome.scripting.executeScript({
          target: { tabId, allFrames: true },
          world: "MAIN",
          func: () => {
            if (!window.monaco || !window.monaco.editor) return null;
            const editors = window.monaco.editor.getEditors();
            for (const ed of editors) {
              try {
                const m = ed.getModel();
                if (m && String(m.getValue() || "").length > 0) return { frameOrigin: location.origin, frameHref: location.href };
              } catch (_) {}
            }
            return null;
          }
        });
        const hit = allRes.find(r => r.result !== null);
        if (hit) diag.push(`ALL_FRAMES_MONACO=${JSON.stringify(hit.result)}`);
        else diag.push("ALL_FRAMES_MONACO=none_found");
      } catch (e) { diag.push(`ALL_FRAMES_MONACO_ERROR=${String(e.message || e)}`); }

      // Try dynamic script injection for Monaco.
      try {
        await sendToTab(tabId, { action: "editor_inject_scan" });
      } catch (e) { diag.push(`INJECT_SCAN_ERROR=${String(e.message || e)}`); }

      // Read editor content via get_editor to see if we can at least read.
      try {
        const r = await sendToTab(tabId, { action: "get_editor" });
        if (r && r.ok && r.content && r.content.length > 0) {
          diag.push(`GET_EDITOR_OK=${r.content.split('\n').length} lines`);
        } else {
          diag.push("GET_EDITOR_FAILED");
        }
      } catch (e) { diag.push(`GET_EDITOR_ERROR=${String(e.message || e)}`); }

      console.log("[BROWSER_AGENT_DIAGNOSTIC]", diag.join(";"));
      // Return full diagnostics in the tool response so the model can report them.
      return { ok: true, info: "Editor diagnostics:", diag };
    }
    case "set_editor_content": {
      // Monaco lives in the page's main world; content scripts can't reach window.monaco.
      // We inject into MAIN world and recursively search all frames (many editors are deeply nested).
      const text = String(args.content || "");

      let diag = "";

      // Try top-level main world first: scan this doc + same-origin iframes from within it.
      try {
        const editorSetResult = await chrome.scripting.executeScript({
          target: { tabId },
          world: "MAIN",
          func: (val) => {
            const scan = (doc, depth = 0) => {
              if (!doc || depth > 8) return null;
              try {
                for (const f of Array.from(doc.querySelectorAll("iframe")).slice(0, 30)) {
                  let fw; try { fw = f.contentWindow; } catch (_) { continue; }
                  let fd; try { fd = f.contentDocument || f.contentWindow?.document; } catch (_) { continue; }
                  const r = scan(fd, depth + 1); if (r) return r;
                }
              } catch (_) {}
              // Check this window for Monaco API.
              try {
                if (!doc.defaultView.monaco || !doc.defaultView.monaco.editor) return null;
                const editors = doc.defaultView.monaco.editor.getEditors();
                for (const ed of editors) {
                  try {
                    const m = ed.getModel();
                    if (m) {
                      m.setValue(val);
                      return { type: "monaco-api", chars: val.length };
                    }
                  } catch (_) {}
                }
              } catch (_) {}
              return null;
            };
            return scan(window.document) || { tried: true, noMonacoFound: true };
          },
          args: [text]
        });
        const result = editorSetResult && editorSetResult.length > 0 ? editorSetResult[0].result : null;
        if (result && result.type) return { ok: true, ...result }; // success path
        if (result) diag = JSON.stringify(result);
      } catch (e) { diag = String(e && e.message || e); }

      // If top-level scan failed, also try direct main-world injection into ALL frames.
      // This hits same-origin iframe contexts directly where their Monaco API lives.
      try {
        const allRes = await chrome.scripting.executeScript({
          target: { tabId, allFrames: true },
          world: "MAIN",
          func: (val) => {
            if (!window.monaco || !window.monaco.editor) return null;
            const editors = window.monaco.editor.getEditors();
            for (const ed of editors) {
              try {
                const m = ed.getModel();
                if (m) {
                  m.setValue(val);
                  return { type: "monaco-api-frame", frameLocation: window.location.href, chars: val.length };
                }
              } catch (_) {}
            }
            return null;
          },
          args: [text]
        });
        const hit = allRes.find(r => r.result && r.result.type);
        if (hit && hit.result) return { ok: true, ...hit.result }; // success path
      } catch (_) {}

      // If main-world Monaco API didn't work, try the content-script path (CodeMirror / simple editors).
      const r = await sendToTab(tabId, { action: "set_editor_content_mainworld_done", content: text });
      if (r && r.ok) return r;

      // Last resort for cross-origin iframes (TripleTen): write via real CDP keystrokes.
      // This works regardless of iframe origin because it sends OS-level input events.
      let attachedHere = false;
      if (!(cfg && cfg.trusted)) {
        await attachDebugger(tabId);
        attachedHere = true;
      }
      try {
        // Focus the editor iframe area, select all, delete, then type.
        await trustedEditorContent(tabId, text);
        await sleep(800);
        return { ok: true, type: "cdp", info: `Wrote ${text.length} chars to code editor via CDP keystrokes` };
      } catch (e) {
        diag += `;CDP_error:${String(e.message || e)}`;
      } finally {
        if (attachedHere) await detachDebugger(tabId);
      }

      // Return diagnostics so the model knows what happened.
      return { ok: false, error: "Could not write to code editor.", diag: `${diag}` };
    }
    case "click": {
      if (cfg && cfg.trusted) {
        const loc = await sendToTab(tabId, { action: "locate", index: args.index });
        if (!loc || !loc.ok) return loc || { ok: false, error: "locate failed" };
        await sendToTab(tabId, { action: "cursor_move", x: loc.x, y: loc.y });
        await sleep(340);           // let the cursor visibly arrive
        await sendToTab(tabId, { action: "cursor_click", x: loc.x, y: loc.y });
        await sleep(340);           // let the ripple show before any navigation
        await trustedClick(tabId, loc.x, loc.y);
        await sleep(800);
        return { ok: true, info: `Clicked [${args.index}] ${loc.label} (trusted)` };
      }
      const r = await sendToTab(tabId, { action: "click", index: args.index });
      await sleep(800);
      return r;
    }
    case "type": {
      const doClear = args.clear !== false; // default true
      if (cfg && cfg.trusted) {
        const loc = await sendToTab(tabId, { action: "locate", index: args.index });
        if (!loc || !loc.ok) return loc || { ok: false, error: "locate failed" };
        await sendToTab(tabId, { action: "cursor_move", x: loc.x, y: loc.y });
        await sleep(rand(120, 260));
        await sendToTab(tabId, { action: "cursor_click", x: loc.x, y: loc.y });
        await trustedClick(tabId, loc.x, loc.y); // focus the field
        await sleep(rand(80, 200));
        if (doClear) await trustedClearField(tabId);
        else await trustedMoveToEnd(tabId); // append: put caret at the end
        await trustedType(tabId, args.text);
        if (args.submit) { await sleep(rand(200, 500)); await trustedEnter(tabId); }
        return { ok: true, info: `${doClear ? "Typed into" : "Appended to"} [${args.index}]${args.submit ? " and submitted" : ""} (trusted)` };
      }
      return await sendToTab(tabId, { action: "type", index: args.index, text: args.text, submit: !!args.submit, clear: doClear });
    }
    case "scroll":
      return await sendToTab(tabId, { action: "scroll", direction: args.direction || "down", index: args.index });
    case "select_option": {
      const value = String(args.value != null ? args.value : (args.text != null ? args.text : ""));
      if (cfg && cfg.trusted) {
        const loc = await sendToTab(tabId, { action: "locate", index: args.index });
        if (!loc || !loc.ok) return loc || { ok: false, error: "locate failed" };
        await sendToTab(tabId, { action: "cursor_move", x: loc.x, y: loc.y });
        await sleep(200);
        await trustedClick(tabId, loc.x, loc.y);
        await sleep(150);
        await trustedClearField(tabId);
        await trustedType(tabId, value);
        await sleep(750); // let suggestions load
        return await sendToTab(tabId, { action: "pick_option", index: args.index, value });
      }
      return await sendToTab(tabId, { action: "select_option", index: args.index, value });
    }
    case "drag": {
      if (cfg && cfg.trusted) {
        const c = await sendToTab(tabId, { action: "centers", from: args.from, to: args.to });
        if (!c || !c.ok) return c || { ok: false, error: "Could not locate drag targets." };
        await sendToTab(tabId, { action: "cursor_move", x: c.sx, y: c.sy });
        await sleep(200);
        await trustedDrag(tabId, c.sx, c.sy, c.tx, c.ty);
        await sleep(600);
        return { ok: true, info: `Dragged [${args.from}] onto [${args.to}] (trusted)` };
      }
      const r = await sendToTab(tabId, { action: "drag", from: args.from, to: args.to });
      await sleep(600);
      return r;
    }
    case "screenshot": {
      try {
        const raw = await captureTab(tabId);
        const mk = await sendToTab(tabId, { action: "marks" });
        const w = (mk && mk.w) || 1280, h = (mk && mk.h) || 800;
        const marks = (mk && mk.marks) || [];
        if (marks.length) {
          const annotated = await annotateMarks(raw, marks, w, h);
          return {
            ok: true,
            info: "Screenshot captured",
            _dataUrl: annotated,
            _visionNote: "Screenshot with each interactive element outlined and labeled with a red NUMBER. " +
              "To click one, call click with that number as the index (e.g. click index 12). " +
              "Only use click_at(x,y) for something that has no number."
          };
        }
        const annotated = await annotateGrid(raw, w, h);
        return {
          ok: true, info: "Screenshot captured", _dataUrl: annotated,
          _visionNote: `Screenshot with a red coordinate grid (CSS pixels), viewport ${w}x${h}. Use click_at with x,y from the grid.`
        };
      } catch (e) {
        return { ok: false, error: "Screenshot failed: " + (e && e.message || e) };
      }
    }
    case "click_at": {
      const x = Number(args.x), y = Number(args.y);
      if (!isFinite(x) || !isFinite(y)) return { ok: false, error: "click_at needs numeric x and y." };
      if (cfg && cfg.trusted) {
        // Snap to the nearest numbered element so imprecise vision coords still land.
        const sp = await sendToTab(tabId, { action: "snap_point", x, y });
        const px = sp && sp.snapped ? sp.x : x;
        const py = sp && sp.snapped ? sp.y : y;
        const what = sp && sp.label ? ` — ${sp.label}` : "";
        await sendToTab(tabId, { action: "cursor_move", x: px, y: py });
        await sleep(340);
        await sendToTab(tabId, { action: "cursor_click", x: px, y: py });
        await sleep(340);
        await trustedClick(tabId, px, py);
        await sleep(800);
        return { ok: true, info: `Clicked ${sp && sp.snapped ? "" : "at (" + x + ", " + y + ")"}${what} (trusted)` };
      }
      const r = await sendToTab(tabId, { action: "click_at", x, y });
      await sleep(800);
      return r;
    }
    case "navigate": {
      let url = args.url;
      if (!/^https?:\/\//i.test(url)) url = "https://" + url;
      await chrome.tabs.update(tabId, { url });
      await waitForLoad(tabId);
      await ensureContentScript(tabId);
      return { ok: true, info: "Navigated to " + url };
    }
    default:
      return { error: "unknown tool " + name };
  }
}

// Only the MOST RECENT page snapshot and screenshot are useful — older ones are
// stale (indices change after any action). Replace old ones with tiny stubs so
// they don't cost tokens on every call, without losing any live information.
function compactForModel(messages) {
  let lastPage = -1, lastImg = -1;
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m.role === "tool" && m.name === "get_page") lastPage = i;
    if (m.role === "user" && Array.isArray(m.content) && m.content.some((p) => p && p.type === "image_url")) lastImg = i;
  }
  return messages.map((m, i) => {
    if (m.role === "tool" && m.name === "get_page" && i !== lastPage) {
      return { role: "tool", tool_call_id: m.tool_call_id, name: m.name, content: "[old page snapshot omitted — call get_page for the current page]" };
    }
    if (m.role === "user" && Array.isArray(m.content) && m.content.some((p) => p && p.type === "image_url") && i !== lastImg) {
      return { role: "user", content: "[earlier screenshot omitted]" };
    }
    return m;
  });
}

// Retry wrapper for model calls with exponential backoff (up to 2 retries).
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1500;

async function callModel(cfg, messages, onChunk) {
  const base = cfg.baseUrl.replace(/\/+$/, "");
  const simple = !!cfg.simple;
  const vision = !simple && cfg.vision !== false;

  let msgs = compactForModel(messages);
  if (msgs[0] && msgs[0].role === "system") {
    msgs = msgs.slice();
    msgs[0] = { role: "system", content: SYSTEM_PROMPT + (vision ? VISION_ADDON : "") };
  }

  let tools;
  if (simple) {
    const allow = new Set(["get_page", "get_editor", "set_editor_content", "click", "type", "select_option", "navigate", "scroll", "done"]);
    tools = TOOLS.filter((t) => allow.has(t.function.name));
  } else if (vision) {
    tools = TOOLS;
  } else {
    tools = TOOLS.filter((t) => t.function.name !== "screenshot" && t.function.name !== "click_at");
  }

  const body = JSON.stringify({
    model: cfg.model || "local-model",
    messages: msgs,
    tools,
    tool_choice: "auto",
    temperature: 0.6,
    stream: true
  });

  const controller = new AbortController();
  modelAbort = controller;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await fetch(base + "/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(cfg.apiKey ? { Authorization: "Bearer " + cfg.apiKey } : {})
        },
        body,
        signal: controller.signal
      });
      if (!resp.ok) {
        const errBody = await resp.text().catch(() => "");
        // Some errors are not retryable (e.g. auth failures). Retry only 5xx / timeouts.
        if (resp.status < 500 && resp.status !== 429) throw new Error(`Model HTTP ${resp.status}: ${errBody.slice(0, 300)}`);
        throw new Error(`Model HTTP ${resp.status} (retryable): ${errBody.slice(0, 200)}`);
      }
      return await streamResponse(resp, onChunk);
    } catch (e) {
      if (controller.signal.aborted) throw new Error("aborted by user");
      if (attempt < MAX_RETRIES && /retryable|503|502|504|timeout/i.test(String(e))) {
        // Log but continue; caller gets the final result.
        await sleep(RETRY_DELAY_MS * (1 << attempt));
        continue;
      }
      throw e;
    }
  }
}

// Read an OpenAI-style SSE stream, yield chunks via onChunk(callback), assemble message.
async function streamResponse(resp, onChunk) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let content = "";
  const toolCalls = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const s = line.trim();
      if (!s || !s.startsWith("data:")) continue;
      const d = s.slice(5).trim();
      if (d === "[DONE]") return { role: "assistant", content, tool_calls: toolCalls };
      try {
        const j = JSON.parse(d);
        const delta = j.choices && j.choices[0] && j.choices[0].delta;
        if (!delta) continue;
        if (delta.content) {
          content += delta.content;
          if (onChunk) onChunk(delta.content);
        }
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            let t = toolCalls[tc.index];
            if (!t) { t = { index: tc.index, id: tc.id || "", type: "function", function: { name: "", arguments: "" } }; toolCalls[tc.index] = t; }
            if (tc.id) t.id = tc.id;
            if (tc.function && tc.function.name) t.function.name += tc.function.name;
            if (tc.function && tc.function.arguments) t.function.arguments += tc.function.arguments;
          }
        }
      } catch (e) { /* ignore malformed chunk */ }
    }
  }
  return { role: "assistant", content, tool_calls: toolCalls };
}

// The visible transcript is persisted to storage so it survives popup closes
// and even keeps updating while the popup is shut. The popup renders from it.
// The conversation (messages sent to the model) is ALSO persisted, so context
// survives the service worker being killed by Chrome when idle.
let transcript = [];
let conversation = null;
let saveTimer = null;
const ready = chrome.storage.local.get({ transcript: [], conversation: null }).then((s) => {
  transcript = s.transcript || [];
  conversation = s.conversation || null;
});

function saveTranscript() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    chrome.storage.local.set({ transcript }).catch(() => {
      // Storage full (e.g. big screenshots): drop stored image data and retry
      // so the text transcript still saves instead of the whole write failing.
      transcript = transcript.map((e) => e.kind === "image" ? { kind: "image", dataUrl: null } : e);
      chrome.storage.local.set({ transcript }).catch(() => {});
    });
  }, 80);
}
let lastThinkIndex = null; // the live "think" line of the CURRENT model call

function log(_port, kind, text, run) {
  if (run !== undefined && run !== currentRun) return;   // a cancelled run can't write into a new chat
  transcript.push({ kind, text });
  if (transcript.length > 400) {
    const drop = transcript.length - 400;
    transcript = transcript.slice(drop);
    if (lastThinkIndex != null) lastThinkIndex = lastThinkIndex >= drop ? lastThinkIndex - drop : null;
  }
  saveTranscript();
}
function beginThinkStream() { lastThinkIndex = null; }
// onChunk passes the FULL text streamed so far, so REPLACE the line's text.
// (It used to append the cumulative text on every chunk: "a", "a ab", ...)
function logThinkAppend(fullText) {
  const trimmed = fullText.trim();
  if (!trimmed) return;
  if (lastThinkIndex != null && lastThinkIndex < transcript.length) {
    transcript[lastThinkIndex].text = trimmed;
    saveTranscript(); // persist streaming updates so the chat shows them live
  } else {
    log(null, "think", trimmed);
    lastThinkIndex = transcript.length - 1;
  }
}
// After the call: put the final text on the live line (or remove an empty one).
function endThinkStream(finalText) {
  if (lastThinkIndex != null && lastThinkIndex < transcript.length) {
    if (finalText) transcript[lastThinkIndex].text = finalText;
    else transcript.splice(lastThinkIndex, 1);
    saveTranscript();
  } else if (finalText) {
    log(null, "think", finalText);
  }
  lastThinkIndex = null;
}
function setRunningFlag(v) { chrome.storage.local.set({ running: v }).catch(() => {}); }

// Capture the visible tab as a compressed JPEG (smaller than PNG for storage).
async function captureTab(tabId) {
  const tab = await chrome.tabs.get(tabId);
  return await chrome.tabs.captureVisibleTab(tab.windowId, { format: "jpeg", quality: 55 });
}

async function blobToDataURL(blob) {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
  return "data:image/jpeg;base64," + btoa(bin);
}

// Draw a labelled coordinate grid (in CSS pixels) over the screenshot so the
// vision model can read off x,y for click_at. The capture image is dpr-scaled,
// so multiply CSS positions by dpr when drawing.
async function annotateGrid(dataUrl, cssW, cssH) {
  try {
    const blob = await (await fetch(dataUrl)).blob();
    const bmp = await createImageBitmap(blob);
    // Derive the real image->CSS scale from the actual capture size, so it is
    // correct regardless of display scaling / devicePixelRatio.
    const scaleX = bmp.width / cssW;
    const scaleY = bmp.height / cssH;
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bmp, 0, 0);
    const step = 100; // CSS px between grid lines
    ctx.lineWidth = 1;
    ctx.font = `${Math.round(11 * scaleY)}px sans-serif`;
    ctx.textBaseline = "top";
    for (let x = step; x < cssW; x += step) {
      const px = x * scaleX;
      ctx.strokeStyle = "rgba(255,40,40,0.30)";
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, bmp.height); ctx.stroke();
      ctx.fillStyle = "rgba(255,40,40,0.95)";
      ctx.fillText(String(x), px + 2, 2);
    }
    for (let y = step; y < cssH; y += step) {
      const py = y * scaleY;
      ctx.strokeStyle = "rgba(255,40,40,0.30)";
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(bmp.width, py); ctx.stroke();
      ctx.fillStyle = "rgba(255,40,40,0.95)";
      ctx.fillText(String(y), 2, py + 2);
    }
    const out = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.6 });
    return await blobToDataURL(out);
  } catch (e) {
    return dataUrl; // if annotation fails, fall back to the raw screenshot
  }
}

// Draw numbered boxes on each interactive element (set-of-marks) so the model
// can click by NUMBER instead of guessing pixel coordinates.
async function annotateMarks(dataUrl, marks, cssW, cssH) {
  try {
    const blob = await (await fetch(dataUrl)).blob();
    const bmp = await createImageBitmap(blob);
    const sx = bmp.width / cssW, sy = bmp.height / cssH;
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bmp, 0, 0);
    ctx.textBaseline = "top";
    const fs = Math.max(10, Math.round(11 * sy));
    ctx.font = `bold ${fs}px sans-serif`;
    for (const m of marks) {
      const x = m.x * sx, y = m.y * sy, w = m.w * sx, h = m.h * sy;
      ctx.strokeStyle = "rgba(230,30,30,0.9)";
      ctx.lineWidth = Math.max(1, Math.round(sy));
      ctx.strokeRect(x, y, w, h);
      const label = String(m.i);
      const tw = ctx.measureText(label).width;
      const bw = tw + 6, bh = fs + 4;
      ctx.fillStyle = "rgba(230,30,30,0.95)";
      ctx.fillRect(x, y, bw, bh);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 3, y + 2);
    }
    const out = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.65 });
    return await blobToDataURL(out);
  } catch (e) {
    return dataUrl;
  }
}

// Persist a screenshot into the chat transcript. Keeps image data for only the
// most recent few shots so stored size stays bounded; older ones are cleared.
const MAX_STORED_IMAGES = 4;
function logImage(dataUrl) {
  let kept = 0;
  for (let i = transcript.length - 1; i >= 0; i--) {
    if (transcript[i].kind === "image" && transcript[i].dataUrl) {
      kept++;
      if (kept >= MAX_STORED_IMAGES) transcript[i].dataUrl = null;
    }
  }
  transcript.push({ kind: "image", dataUrl });
  saveTranscript();
}

// Keep the conversation bounded so long chats don't overflow the model's
// context window or Chrome's storage — but keep the IMPORTANT earlier turns.
// Strategy: always keep the system prompt; keep the most recent turns fully
// (with tool_call/tool pairing intact); and from older turns keep only the
// "storyline" — the user's requests and the assistant's text (narration /
// answers) — dropping bulky tool results and screenshots.
const MAX_MSGS = 48;   // when to start trimming
const RECENT = 24;     // most-recent messages kept verbatim
const MAX_STORY = 30;  // cap on compact older messages kept

function messageText(m) {
  if (typeof m.content === "string") return m.content.trim();
  if (Array.isArray(m.content)) return m.content.filter((p) => p && p.type === "text").map((p) => p.text).join(" ").trim();
  return "";
}

function trimConversation() {
  if (!conversation || conversation.length <= MAX_MSGS) return;
  const system = conversation[0];
  const body = conversation.slice(1);

  // The original task (first real user message) — must never be lost, both for
  // context and because some chat templates require a user message to exist.
  const firstUser = body.find((m) => m.role === "user" && messageText(m) && !/^Screenshot\b/.test(messageText(m)));
  const firstUserText = firstUser ? messageText(firstUser) : null;

  const recentStart = Math.max(0, body.length - RECENT);
  const older = body.slice(0, recentStart);
  let recent = body.slice(recentStart);

  // Compact storyline from older turns: user goals + assistant text only.
  let story = [];
  for (const m of older) {
    if (m.role === "user") {
      const t = messageText(m);
      if (t && !/^Screenshot\b/.test(t)) story.push({ role: "user", content: t });
    } else if (m.role === "assistant") {
      const t = messageText(m);
      if (t) story.push({ role: "assistant", content: t }); // drop tool_calls/tool payloads
    }
    // tool messages (page reads, results) are dropped
  }
  // Keep only the most recent slice of the storyline if it's long.
  if (story.length > MAX_STORY) story = story.slice(story.length - MAX_STORY);

  // Recent window must not begin with an orphan tool message.
  while (recent.length && recent[0].role === "tool") recent.shift();

  const next = [system, ...story, ...recent];
  // Guarantee a user message survived — otherwise the model server errors with
  // "No user query found in messages." Re-insert the original task if needed.
  if (firstUserText && !next.some((m) => m.role === "user" && messageText(m))) {
    next.splice(1, 0, { role: "user", content: firstUserText });
  }

  // Mutate the array IN PLACE (do not reassign) so other references to it —
  // notably `messages` inside runAgent — stay pointing at the same array.
  conversation.length = 0;
  for (const m of next) conversation.push(m);
}

// Persist the conversation, stripping heavy screenshot data (replaced with a
// short text note) so stored context stays small but textual memory is kept.
let convSaveTimer = null;
function saveConversation() {
  trimConversation();
  clearTimeout(convSaveTimer);
  convSaveTimer = setTimeout(() => {
    const slim = (conversation || []).map((m) => {
      if (Array.isArray(m.content)) {
        return { ...m, content: m.content.map((p) => p && p.type === "image_url" ? { type: "text", text: "[screenshot]" } : p) };
      }
      return m;
    });
    chrome.storage.local.set({ conversation: slim }).catch(() => {});
  }, 120);
}

function resetConversation() {
  conversation = [{ role: "system", content: SYSTEM_PROMPT }];
  transcript = [];
  chrome.storage.local.set({ transcript, conversation }).catch(() => {});
}

// Actions that should trigger a stable-wait after executing (to avoid stale reads).
const ACTIONS_NEED_STABLE = new Set(["click", "click_at", "type", "navigate", "scroll", "select_option", "upload_file", "drag"]);

async function runAgent(cfg, task, port, myRun) {
  const stopped = () => myRun !== currentRun;
  const stopMsg = () => {   // say why THIS run stopped (it used to always claim "Stopped by user.")
    const why = cancelReasons[myRun] || "stop";
    delete cancelReasons[myRun];   // read once; the map must not grow forever
    if (why === "stop") log(port, "error", "Stopped by user.");
    else if (why === "superseded") log(port, "result", "Previous task replaced by your new message.");
  };
  const logRun = (kind, text) => log(port, kind, text, myRun);
  if (stopped()) return;
  await ready;
  startKeepAlive(); // help prevent Chrome from killing the SW mid-run

  // LOCK ON TO THE ACTIVE TAB — we operate on THIS tab for the entire run.
  // Switching tabs mid-task is now safe (the agent keeps working in the original).
  const workTab = await activeTab();
  if (!workTab) { logRun("error", "No active tab."); return; }
  let attached = false;

  // High-realism mode: attach debugger. If DevTools is open on this tab, fall back gracefully.
  if (cfg.trusted) {
    try {
      await attachDebugger(workTab.id);
      attached = true;
      logRun("result", "High-realism input ON (debugger attached — banner is expected).");
    } catch (e) {
      cfg.trusted = false;
      logRun("error", "Could not enable high-realism input (" + e.message + "). Close DevTools for this tab if open. Falling back to standard input.");
    }
  }

  // Ensure content script is loaded into the locked tab before we start.
  await ensureContentScript(workTab.id);

  if (!conversation) resetConversation();
  conversation.push({ role: "user", content: task });
  saveConversation();
  logRun("user", task);
  const messages = conversation;

  const recentSigs = [];
  let lastPageSig = null;

  const maxSteps = cfg.maxSteps || 15;
  const AUTO_CAP = 200;
  const stepCap = cfg.auto ? Math.max(maxSteps, AUTO_CAP) : maxSteps;

  // Streaming: model outputs appear live in the chat as a single updating think entry.
  let currentThinkChunk = "";
  const onChunk = (chunk) => {
    currentThinkChunk += chunk;
    logThinkAppend(currentThinkChunk); // updates last think line in-place (no growing duplicates)
  };

  try {
  for (let step = 0; step < stepCap; step++) {
    if (stopped()) { stopMsg(); return; }
    if (cfg.auto && step === maxSteps) logRun("result", `Auto mode: past ${maxSteps} steps, still working…`);

    let msg;
    currentThinkChunk = "";
    beginThinkStream();   // each model call gets its own live "thinking" line
    try {
      msg = await callModel(cfg, messages, onChunk);
    } catch (e) {
      // streaming text (if any) is already on screen in the live think line
      if (stopped()) { stopMsg(); return; }
      logRun("error", "Model call failed: " + e.message);
      return;
    }
    if (stopped()) { stopMsg(); return; }

    messages.push(msg);
    saveConversation();

    let toolCalls = msg.tool_calls || [];
    let parsedFromText = false;
    if (toolCalls.length === 0 && msg.content) {
      const parsed = parseTextToolCall(msg.content);
      if (parsed) {
        toolCalls = [{ id: "txt_" + step, type: "function", function: { name: parsed.name, arguments: JSON.stringify(parsed.args || {}) } }];
        msg.tool_calls = toolCalls;
        parsedFromText = true;
      }
    }

    // Finalize the live think line with the complete text (instead of adding a
    // second copy). A tool call written as text isn't "thinking" - drop it.
    endThinkStream(msg.content && !parsedFromText ? msg.content.trim() : "");

    if (toolCalls.length === 0) {
      logRun("done", msg.content || "(model returned no tool call and no content)");
      return;
    }

    const answered = new Set();
    for (const tc of toolCalls) {
      if (stopped()) {
        // Answer every tool call of this assistant turn that didn't run, or
        // strict OpenAI-compatible servers reject the NEXT request (400: a
        // tool_call without a tool reply) - which happened on every follow-up
        // sent while a batch of tool calls was running.
        for (const t of toolCalls) {
          if (t && t.id && !answered.has(t.id)) {
            messages.push({ role: "tool", tool_call_id: t.id, name: (t.function && t.function.name) || "unknown", content: "cancelled (task was stopped)" });
            answered.add(t.id);
          }
        }
        saveConversation();
        stopMsg();
        return;
      }
      const fn = tc && tc.function;
      if (!fn || !fn.name) {
        if (tc && tc.id) { messages.push({ role: "tool", tool_call_id: tc.id, name: "unknown", content: "malformed tool call ignored" }); answered.add(tc.id); }
        continue;
      }
      let args = {};
      try {
        const raw = fn.arguments;
        args = (raw && typeof raw === "object") ? raw : JSON.parse(raw || "{}");
      } catch (e) { args = {}; }
      args = coerceArgs(args);
      const name = fn.name;

      // Dangerous-action warning: if about to click something with suspicious text, log a visible warning.
      // The user can still stop immediately; this is purely visibility (no confirmation blocking for now).
      if (name === "click" && args.index != null) {
        try {
          const loc = await sendToTab(workTab.id, { action: "locate", index: args.index });
          if (loc && loc.ok && loc.label && DANGEROUS_PATTERNS.test(loc.label)) {
            logRun("error", `⚠ Potentially destructive click: "${loc.label}" — watch closely and hit Stop if wrong.`);
          }
        } catch (e) { /* ignore; continue with the action */ }
      }

      if (name === "done") {
        const answer = args.answer || "(done)";
        messages.push({ role: "tool", tool_call_id: tc.id, name, content: "ok" });
        answered.add(tc.id);
        for (const t of toolCalls) {
          if (t && t.id && !answered.has(t.id)) {
            messages.push({ role: "tool", tool_call_id: t.id, name: (t.function && t.function.name) || "unknown", content: "skipped (task complete)" });
          }
        }
        messages.push({ role: "assistant", content: answer });
        saveConversation();
        logRun("done", answer);
        return;
      }

      logRun("tool", `${name}(${JSON.stringify(args)})`);

      let result;
      try {
        result = await executeTool(name, args, workTab.id, cfg);
      } catch (e) {
        result = { ok: false, error: `Tool '${name}' failed: ${(e && e.message) || e}` };
      }

      // After state-changing actions, wait for the DOM to stabilize so subsequent get_page reads aren't stale.
      if (ACTIONS_NEED_STABLE.has(name)) {
        await waitUntilStable(workTab.id);
        await ensureContentScript(workTab.id);
      }

      let imageUrl = null;
      let visionNote = "Screenshot of the current visible page:";
      if (result && result._dataUrl) { imageUrl = result._dataUrl; delete result._dataUrl; }
      if (result && result._visionNote) { visionNote = result._visionNote; delete result._visionNote; }
      if (imageUrl) logImage(imageUrl);

      const short = JSON.stringify(result);
      logRun("result", short.length > 500 ? short.slice(0, 500) + " …" : short);

      const OBSERVE = new Set(["get_page", "get_editor", "click", "screenshot"]); // never warn about clicks or reads — file tab switches are fine
      let toolContent = JSON.stringify(result).slice(0, 8000);
      if (!OBSERVE.has(name)) {
        const sig = name + ":" + JSON.stringify(args);
        recentSigs.push(sig);
        if (recentSigs.length > 8) recentSigs.shift();
        const repeats = recentSigs.filter((s) => s === sig).length;
        if (repeats >= 3) {
          toolContent += `\n\n⚠ You have done this exact action ${repeats} times with no effect. It is NOT working — do something different: a different element, dismiss any open dialog, or move to another section.`;
        }
      }
      if (name === "get_page" && result && result.page) {
        const psig = (result.url || "") + "|" + result.page.length + "|" + result.page.slice(0, 160);
        if (psig === lastPageSig) {
          toolContent += "\n\nNOTE: this page is identical to your previous get_page — your last action had no visible effect. Do something different (a different element, scroll, or approach).";
        }
        lastPageSig = psig;
      }

      messages.push({ role: "tool", tool_call_id: tc.id, name, content: toolContent });
      answered.add(tc.id);

      if (imageUrl) {
        messages.push({
          role: "user",
          content: [
            { type: "text", text: visionNote },
            { type: "image_url", image_url: { url: imageUrl } }
          ]
        });
      }
      saveConversation();
    }
  }
  logRun("error", cfg.auto
    ? `Stopped at the auto-mode safety limit (${stepCap} steps). Send another message to continue.`
    : `Reached the ${maxSteps}-step limit. Turn on Auto mode to keep going, or send another message to continue.`);
  } finally {
    stopKeepAlive();
    try { await sendToTab(workTab.id, { action: "hide_cursor" }); } catch (e) {}
    // cleanup lines are always shown (logRun drops writes from a cancelled run)
    if (attached) { await detachDebugger(workTab.id); log(port, "result", "Debugger detached."); }
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "agent") return;
  port.onMessage.addListener(async (m) => {
    if (m.type === "run") {
      // Cancel whatever is running, take a ticket, and queue behind it
      // SYNCHRONOUSLY (before any await) so runs can never overlap. The old
      // run's cleanup (debugger detach on the same tab, running flag off,
      // "finished") finishes before this one starts. If Stop / New chat / a
      // newer Run arrives while we wait, the ticket is stale and we bail.
      cancelRuns("superseded");
      const ticket = currentRun;
      const prev = runChain;
      let release;
      runChain = new Promise((r) => { release = r; });
      try { await prev; } catch (e) {}
      if (ticket !== currentRun) {
        // cancelled while queued: if that was a Stop, say so (the run it
        // replaced already logged "replaced by your new message")
        if (cancelReasons[ticket] === "stop") log(port, "error", "Stopped by user.");
        delete cancelReasons[ticket];
        release();
        return;
      }
      setRunningFlag(true);
      try {
        await runAgent(m.cfg, m.task, port, ticket);
      } catch (e) {
        log(port, "error", "Agent crashed: " + (e && e.message || e));
      } finally {
        if (ticket === currentRun) {          // a newer run owns the UI state otherwise
          setRunningFlag(false);
          try { port.postMessage({ type: "finished" }); } catch (e) {}
        }
        release();
      }
    } else if (m.type === "newchat") {
      cancelRuns("newchat"); // halt any in-flight run so it can't write into the new chat
      setRunningFlag(false);
      try { await runChain; } catch (e) {}   // let it finish logging into the OLD chat
      resetConversation();
      log(port, "assistant", "New chat started. What would you like me to do?");
    } else if (m.type === "stop") {
      cancelRuns();
      setRunningFlag(false);
    } else if (m.type === "test") {
      try {
        const base = m.cfg.baseUrl.replace(/\/+$/, "");
        const r = await fetch(base + "/models", {
          headers: m.cfg.apiKey ? { Authorization: "Bearer " + m.cfg.apiKey } : {}
        });
        const body = await r.text();
        log(port, r.ok ? "result" : "error", `GET /models -> ${r.status}: ${body.slice(0, 300)}`);
      } catch (e) {
        log(port, "error", "Test failed: " + e.message);
      }
    }
  });
});
