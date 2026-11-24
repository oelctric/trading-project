// Content script: inspects the DOM and performs actions on behalf of the model.
// It keeps a numbered map of interactive elements so the model can refer to them by index.

(() => {
  // Always remove any stray cursor left over from a previous injection or an
  // extension reload, so you never end up with two (one frozen) cursors.
  document.querySelectorAll("#__lmba_cursor, .__lmba_ripple").forEach((e) => e.remove());

  if (window.__lmbaInstalled) return;
  window.__lmbaInstalled = true;

  let labeledElements = [];

  // ---- Visible fake cursor overlay (so you can watch the agent work) ----
  let cursorEl = null;
  let cursorHideTimer = null;
  const ensureCursor = () => {
    if (cursorEl && document.documentElement.contains(cursorEl)) return cursorEl;
    cursorEl = document.createElement("div");
    cursorEl.id = "__lmba_cursor";
    cursorEl.style.cssText = [
      "position:fixed", "left:0", "top:0", "width:24px", "height:24px",
      "z-index:2147483647", "pointer-events:none", "margin:0", "padding:0", "opacity:0",
      "transition:transform 130ms ease-out, opacity 300ms ease-out", "will-change:transform",
      "transform:translate(-100px,-100px)", "filter:drop-shadow(0 1px 2px rgba(0,0,0,.5))"
    ].join(";");
    cursorEl.innerHTML =
      '<svg width="24" height="24" viewBox="0 0 24 24" fill="#111" stroke="#fff" stroke-width="1.5" stroke-linejoin="round">' +
      '<path d="M4 2 L4 19 L8.5 14.8 L11.6 21.4 L14.3 20.1 L11.2 13.6 L18 13.4 Z"/></svg>';
    document.documentElement.appendChild(cursorEl);
    return cursorEl;
  };
  const showCursor = () => {
    const c = ensureCursor();
    c.style.opacity = "1";
    clearTimeout(cursorHideTimer);
    cursorHideTimer = setTimeout(() => { if (cursorEl) cursorEl.style.opacity = "0"; }, 1600);
  };
  const hideCursor = () => {
    clearTimeout(cursorHideTimer);
    if (cursorEl) { cursorEl.remove(); cursorEl = null; }
    document.querySelectorAll("#__lmba_cursor, .__lmba_ripple").forEach((e) => e.remove());
  };
  const moveCursorTo = (x, y) => { ensureCursor().style.transform = `translate(${x}px,${y}px)`; showCursor(); };
  const cursorClick = (x, y) => {
    moveCursorTo(x, y);
    const r = document.createElement("div");
    r.className = "__lmba_ripple";
    r.style.cssText =
      `position:fixed;left:${x}px;top:${y}px;width:10px;height:10px;border:2px solid #4a9eff;` +
      "border-radius:50%;z-index:2147483646;pointer-events:none;transform:translate(-50%,-50%);" +
      "opacity:.9;transition:all 420ms ease-out;";
    document.documentElement.appendChild(r);
    requestAnimationFrame(() => { r.style.width = "40px"; r.style.height = "40px"; r.style.opacity = "0"; });
    setTimeout(() => r.remove(), 460);
  };

  const isVisible = (el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    const style = window.getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none" || style.opacity === "0") return false;
    // roughly on/near screen
    if (rect.bottom < -50 || rect.top > (window.innerHeight + 50)) return false;
    return true;
  };

  // Fields whose VALUE must never reach the model: password inputs (and ones a
  // "show password" toggle switched to type=text - remembered once seen),
  // autocomplete password / one-time-code fields, and secret-looking names.
  const secretSeen = new WeakSet();
  const SECRET_NAME = /pass(word|wd|code)?|pwd|otp|one.?time|secret|token|api.?key|cvv|cvc|security.?code|pin\b/i;
  const isSecretField = (el) => {
    if (el.tagName !== "INPUT" && el.tagName !== "TEXTAREA") return false;
    const type = (el.getAttribute("type") || "").toLowerCase();
    const ac = (el.getAttribute("autocomplete") || "").toLowerCase();
    const names = [el.getAttribute("name"), el.id, el.getAttribute("aria-label"), el.getAttribute("placeholder")].join(" ");
    if (type === "password" || /current-password|new-password|one-time-code|cc-csc/.test(ac) ||
        SECRET_NAME.test(names)) {
      secretSeen.add(el);
    }
    return secretSeen.has(el);
  };

  const labelText = (el) => {
    const tag = el.tagName;
    // For form fields, surface the CURRENT value so the model knows what's
    // already filled in (vs. just the placeholder), which it needs for context.
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      const val = (typeof el.value === "string" ? el.value : "").trim();
      const ph = el.getAttribute("placeholder") || el.getAttribute("aria-label") || el.getAttribute("name") || "";
      // NEVER reveal a password's value: page snapshots go to the model server and
      // are stored in the chat history / debug log.
      const secret = isSecretField(el);
      if (secret) return ((ph || "password") + (val ? " (filled)" : " (empty)")).replace(/\s+/g, " ").trim().slice(0, 160);
      const t = val ? `“${val}”` : (ph ? ph + " (empty)" : "(empty)");
      return t.replace(/\s+/g, " ").trim().slice(0, 160);
    }
    let t =
      el.getAttribute("aria-label") ||
      el.getAttribute("placeholder") ||
      el.innerText ||
      el.getAttribute("title") ||
      el.getAttribute("alt") ||
      el.getAttribute("name") ||
      "";
    t = t.replace(/\s+/g, " ").trim();
    return t.slice(0, 120);
  };

  // Collect the top document plus any SAME-ORIGIN iframe documents (recursively).
  // Cross-origin iframes are inaccessible for security and are skipped.
  const accessibleDocs = () => {
    const docs = [document];
    const scan = (doc) => {
      let frames = [];
      try { frames = Array.from(doc.querySelectorAll("iframe, frame")); } catch (e) { return; }
      for (const f of frames) {
        let d = null;
        try { d = f.contentDocument; } catch (e) { d = null; }
        if (d && d.body && !docs.includes(d)) { docs.push(d); scan(d); }
      }
    };
    scan(document);
    return docs;
  };

  const isVisibleAnyDoc = (el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    const view = el.ownerDocument.defaultView || window;

    // Walk up the tree: any hidden/display:none parent makes this element effectively invisible.
    for (let node = el; node; node = node.parentElement || (node.getRootNode && node.getRootNode().host)) {
      if (!node || node.nodeType !== 1) break;
      try {
        const cs = view.getComputedStyle(node);
        if (cs.display === "none" || cs.visibility === "hidden") return false;
      } catch (_) {}
    }

    const style = view.getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none" || style.opacity === "0") return false;
    return true;
  };

  // Translate an element's rect into TOP-window coordinates, walking up through
  // any (same-origin) iframe offsets so clicks/cursor land in the right place.
  const topLevelRect = (el) => {
    const r = el.getBoundingClientRect();
    let x = r.left, y = r.top;
    let win = el.ownerDocument.defaultView;
    while (win && win !== window.top && win.frameElement) {
      const fr = win.frameElement.getBoundingClientRect();
      x += fr.left; y += fr.top;
      win = win.parent;
    }
    return { left: x, top: y, width: r.width, height: r.height };
  };

  // Detect code editor instances (Monaco/CodeMirror) so the model can interact with them.
  const findCodeEditor = () => {
    for (const doc of accessibleDocs()) {
      // Monaco-style textarea detection: look for the hidden textarea that holds buffer state
      const monacoTa = doc.querySelector('.monaco-editor textarea, .editor-scrollable textarea, [class*="monaco"] textarea');
      if (monacoTa) return { type: "monaco", el: monacoTa };
      // CodeMirror uses a textarea or contentEditable div
      const cm = doc.querySelector('.CodeMirror textarea, .CodeMirror, [class*="CodeMirror"]');
      if (cm) return { type: "codemirror", el: cm };

      // Broader detection: some embedded editors use less obvious class names.
      // If we see a monaco-workbench or editor area but no textarea found yet, note it.
      const monacoContainer = doc.querySelector('.monaco-workbench, .monaco-editor');
      if (monacoContainer) return { type: "monaco", el: monacoContainer };
    }
    return null;
  };

  // Find a writable editor handle. Returns an object that describeEditor can use.
  const findWritableEditor = () => {
    for (const doc of accessibleDocs()) {
      const win = doc.defaultView || window;

      // Monaco: try via the per-window API first. In many sites, each iframe has its own monaco namespace.
      try {
        if (win.monaco && win.monaco.editor) {
          const editors = win.monaco.editor.getEditors();
          for (const ed of editors) {
            try {
              const m = ed.getModel();
              if (m) return { type: "monaco-api", doc, editor: ed, model: m };
            } catch (e) {}
          }
        }
      } catch (e) {}

      // Monaco textarea: fallback when API is hidden or editors aren't in getEditors().
      // Many embedded Monacos (TripleTen, Codecademy-style) hide the API but still use this pattern.
      const monacoTa = doc.querySelector('.monaco-editor textarea');
      if (monacoTa) {
        return { type: "monaco-ta", doc, el: monacoTa };
      }

      // CodeMirror
      const cmTa = doc.querySelector('.CodeMirror textarea');
      if (cmTa) return { type: "codemirror", doc, el: cmTa };
    }
    return null;
  };

  // Get current text content of the code editor (for model to see what's there).
  const getCodeEditorContent = () => {
    for (const doc of accessibleDocs()) {
      // Monaco textarea holds the current buffer
      const monaco = doc.querySelector('.monaco-editor textarea');
      if (monaco && typeof monaco.value === "string") return monaco.value;
      // CodeMirror textarea
      const cm = doc.querySelector('.CodeMirror textarea');
      if (cm && typeof cm.value === "string") return cm.value;
    }
    return null;
  };

  const collectInteractive = () => {
    const selector =
      'a[href], button, input:not([type=hidden]), textarea, select, summary, ' +
      '[role=button], [role=link], [role=tab], [role=checkbox], [role=radio], [role=option], [role=menuitem], ' +
      '[onclick], [contenteditable=true], [tabindex]:not([tabindex="-1"]), label';

    // Add the code editor textarea as an interactive element so the model can reference it.
    const editor = findCodeEditor();
    const set = new Set();
    for (const doc of accessibleDocs()) {
      try { doc.querySelectorAll(selector).forEach((el) => set.add(el)); } catch (e) {}
      // Framework clickables (React/Vue) attach handlers in JS with no onclick
      // attribute or role. The browser still shows them with cursor:pointer, so
      // use that as a heuristic to find things like custom radio/answer rows.
      try {
        const view = doc.defaultView || window;
        let count = 0;
        for (const el of doc.querySelectorAll("div, span, li, td, section, article, p, img, svg")) {
          if (count++ > 800) break;
          let cs; try { cs = view.getComputedStyle(el); } catch (e) { continue; }
          if (cs && cs.cursor === "pointer") set.add(el);
        }
      } catch (e) {}
    }
    let nodes = Array.from(set).filter(isVisibleAnyDoc);
    // Keep the outermost clickable region: drop an element if a clickable
    // ancestor is also in the list (avoids marking the inner icon+text of a row).
    // But NEVER drop real form controls — the model must be able to target them.
    // Walk each node's ancestor chain (O(n·depth)) instead of an O(n²) contains scan.
    const KEEP_ALWAYS = new Set(["INPUT", "TEXTAREA", "SELECT", "BUTTON", "A"]);
    const nodeSet = new Set(nodes);
    nodes = nodes.filter((el) => {
      if (KEEP_ALWAYS.has(el.tagName)) return true;
      for (let p = el.parentElement; p; p = p.parentElement) {
        if (nodeSet.has(p)) return false;
      }
      return true;
    });
    // Drop <label> elements that aren't the actionable control themselves:
    // section-heading labels (no associated field) do nothing when clicked, and
    // labels redundant with an already-listed control just add noise.
    const curSet = new Set(nodes);
    nodes = nodes.filter((el) => {
      if (el.tagName !== "LABEL") return true;
      let ctrl = null;
      const forId = el.getAttribute("for");
      if (forId) { try { ctrl = el.ownerDocument.getElementById(forId); } catch (e) {} }
      if (!ctrl) ctrl = el.querySelector('input:not([type="hidden"]), textarea, select, button, [role="checkbox"], [role="radio"], [contenteditable="true"]');
      if (!ctrl) return false;          // pure heading label — not clickable
      if (curSet.has(ctrl)) return false; // redundant — the control is already listed
      return true;                       // label is the only handle for a hidden/custom control
    });
    // Drop elements with no usable label (empty <a>/<div>/<span> noise) — a model
    // can't act on something it can't identify, and long lists overwhelm weak
    // models. Always keep real form controls and buttons even if unlabeled.
    const KEEP_EMPTY = new Set(["INPUT", "TEXTAREA", "SELECT", "BUTTON"]);
    nodes = nodes.filter((el) => KEEP_EMPTY.has(el.tagName) || labelText(el).trim().length > 0);

    // Exclude the Monaco editor's hidden textarea from labeledElements (it causes click-at confusion),
    // but include the code editor as a separate marker in get_page output.
    nodes = nodes.filter((el) => {
      if (el.tagName === "TEXTAREA") {
        let parent = el.parentElement;
        while (parent && parent !== document.body) {
          const c = parent.className || "";
          if (c.indexOf("monaco-editor") >= 0 || c.indexOf("CodeMirror") >= 0) return false;
          parent = parent.parentElement;
        }
      }
      return true;
    });

    labeledElements = nodes;
    return nodes.map((el, i) => {
      const tag = el.tagName.toLowerCase();
      const type = el.getAttribute("type") || "";
      return `[${i}] <${tag}${type ? " " + type : ""}> ${labelText(el)}`;
    });
  };

  const NOISE_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "SVG", "PATH", "HEAD", "TEMPLATE", "IFRAME", "CANVAS", "AUDIO"]);
  const BLOCK_TAGS = new Set(["P", "DIV", "SECTION", "ARTICLE", "LI", "TR", "UL", "OL", "BLOCKQUOTE", "PRE",
    "TABLE", "HEADER", "FOOTER", "NAV", "ASIDE", "FORM", "FIELDSET", "MAIN", "FIGURE", "LABEL"]);

  // Walk the DOM in reading order, emitting text with interactive elements
  // inlined as [n] markers (indices from labeledElements) so the model sees each
  // control IN its surrounding context — far easier to understand than a wall of
  // text plus a separate element list. `emitted` collects the shown indices.
  const structuredWalk = (root, idxMap, emitted, state) => {
    const parts = [];
    const walk = (node) => {
      if (state.budget <= 0 || state.nodes > 15000) return;
      for (const child of node.childNodes) {
        if (state.budget <= 0 || state.nodes > 15000) return;
        state.nodes++;
        if (child.nodeType === 3) {
          const t = child.textContent.replace(/\s+/g, " ");
          if (t.trim()) { parts.push(t); state.budget -= t.length; }
          continue;
        }
        if (child.nodeType !== 1) continue;
        const tag = child.tagName;
        if (NOISE_TAGS.has(tag)) continue;
        let cs; try { cs = (child.ownerDocument.defaultView || window).getComputedStyle(child); } catch (e) { cs = null; }
        if (cs && (cs.display === "none" || cs.visibility === "hidden")) continue;
        const idx = idxMap.get(child);
        if (idx !== undefined) {
          const type = child.getAttribute("type") || "";
          const label = labelText(child);
          parts.push(` [${idx}]${type ? "(" + type + ")" : ""} ${label} `);
          emitted.add(idx);
          state.budget -= (label.length + 10);
          continue; // its label already captured — don't re-emit its inner text
        }
        if (/^H[1-6]$/.test(tag)) { parts.push("\n" + "#".repeat(+tag[1]) + " "); walk(child); parts.push("\n"); continue; }
        if (BLOCK_TAGS.has(tag)) { walk(child); parts.push("\n"); continue; }
        walk(child);
      }
    };
    walk(root);
    return parts.join("");
  };

  // Build the combined snapshot: structured main content with [n] markers inline,
  // plus a short list of any labeled elements not shown inline (nav/sidebar).
  const pageSnapshot = () => {
    const idxMap = new Map();
    labeledElements.forEach((el, i) => idxMap.set(el, i));
    const emitted = new Set();
    const state = { budget: 9000, nodes: 0 };
    let text = "";
    for (const doc of accessibleDocs()) {
      try {
        if (!doc.body) continue;
        const main = doc.querySelector("main, [role=main], article");
        const root = (main && (main.innerText || "").trim().length > 200) ? main : doc.body;
        text += structuredWalk(root, idxMap, emitted, state) + "\n";
        if (state.budget <= 0) break;
      } catch (e) {}
    }
    text = text.replace(/[ \t]{2,}/g, " ").replace(/ *\n */g, "\n").replace(/\n{3,}/g, "\n\n").trim();
    if (state.budget <= 0) text += "\n…[truncated — scroll down for more]";
    const others = [];
    labeledElements.forEach((el, i) => {
      if (!emitted.has(i)) {
        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute("type") || "";
        others.push(`[${i}] <${tag}${type ? " " + type : ""}> ${labelText(el)}`);
      }
    });
    if (others.length) text += "\n\nOTHER CLICKABLE ELEMENTS (menus / nav / sidebar):\n" + others.slice(0, 120).join("\n");
    return text;
  };

  const getState = () => {
    const se = document.scrollingElement || document.documentElement;
    collectInteractive(); // populate labeledElements (indices used inline below)

    // Detect code editors and include a preview of their content so the model knows about them.
    let editorNote = null;
    const codeContent = getCodeEditorContent();
    if (codeContent && codeContent.length > 0) {
      const lines = codeContent.split('\n').length;
      editorNote = `[CODE EDITOR ACTIVE (${lines} lines). You cannot read the actual code from get_page — it only shows line numbers, not content. Use get_editor() to read the full file contents, and set_editor_content(content) with the complete updated text when making changes.]`;
    }

    if (!editorNote && findCodeEditor()) {
      editorNote = "[CODE EDITOR ON PAGE (appears empty or closed). Use get_editor() after selecting a file tab to read its full contents. Use set_editor_content(content) with the complete updated text to write changes — do NOT click/type inside the editor.]";
    }

    return {
      url: location.href,
      title: document.title,
      text: pageSnapshot(),
      count: labeledElements.length,
      scroll: `${Math.round(se.scrollTop)}/${Math.max(0, Math.round(se.scrollHeight - se.clientHeight))}`,
      ...(editorNote ? { editor_note: editorNote } : {})
    };
  };

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const rand = (a, b) => a + Math.random() * (b - a);

  const nativeSetValue = (el, value) => {
    // Use the setter from the element's own window and its own element type:
    // the top window's HTMLInputElement setter throws on a <select>, and on
    // elements that live in an iframe.
    const win = (el.ownerDocument && el.ownerDocument.defaultView) || window;
    const ctor = { TEXTAREA: win.HTMLTextAreaElement, SELECT: win.HTMLSelectElement }[el.tagName] || win.HTMLInputElement;
    let setter = null;
    try { setter = Object.getOwnPropertyDescriptor(ctor.prototype, "value")?.set; } catch (e) {}
    try { if (setter) { setter.call(el, value); return; } } catch (e) {}
    el.value = value;
  };

  // Move the "cursor" toward an element with a few intermediate mousemove events,
  // then hover, so behavioral trackers see approach + hover before the click.
  const humanApproach = async (el) => {
    const rect = topLevelRect(el);
    // Use dead center with minimal jitter — misclicks from hitting adjacent elements are worse than
    // triggering anti-bot detection from perfectly centered clicks. Only randomize within tight bounds.
    const tx = rect.left + rect.width / 2 + rand(-rect.width * 0.05, rect.width * 0.05);
    const ty = rect.top + rect.height / 2 + rand(-rect.height * 0.05, rect.height * 0.05);
    const steps = Math.floor(rand(3, 6));
    let cx = rand(0, window.innerWidth), cy = rand(0, window.innerHeight);
    for (let s = 1; s <= steps; s++) {
      cx += (tx - cx) / (steps - s + 1);
      cy += (ty - cy) / (steps - s + 1);
      const target = document.elementFromPoint(Math.min(Math.max(cx, 0), window.innerWidth - 1),
                                                Math.min(Math.max(cy, 0), window.innerHeight - 1)) || el;
      target.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: cx, clientY: cy }));
      moveCursorTo(cx, cy);
      await sleep(rand(15, 45));
    }
    moveCursorTo(tx, ty);
    el.dispatchEvent(new MouseEvent("mouseover", { bubbles: true, clientX: tx, clientY: ty }));
    el.dispatchEvent(new MouseEvent("mouseenter", { bubbles: true, clientX: tx, clientY: ty }));
    return { tx, ty };
  };

  // On dynamic pages the DOM re-renders, so a captured index can point at a node
  // that's been detached. Acting on it silently fails or hits the wrong thing —
  // instead, tell the model to re-read so it gets fresh indices.
  const staleGuard = (el, i) => {
    if (!el) return { ok: false, error: `No element with index ${i}. Call get_page first.` };
    if (!el.isConnected) return { ok: false, error: `Element [${i}] is stale — the page changed since the last get_page. Call get_page again to refresh the numbered list, then retry.` };
    return null;
  };

  // Keep clicks in THIS tab: if the target (or its anchor) would open a new tab
  // via target="_blank", rewrite it to open in place instead.
  const preventNewTab = (el) => {
    try {
      const a = (el.closest && el.closest("a[target]")) || (el.getAttribute && el.getAttribute("target") ? el : null);
      if (a && a.getAttribute("target") && a.getAttribute("target") !== "_self") a.setAttribute("target", "_self");
    } catch (e) {}
  };

  const clickIndex = async (i) => {
    const el = labeledElements[i];
    const stale = staleGuard(el, i); if (stale) return stale;
    preventNewTab(el);
    el.scrollIntoView({ block: "center", behavior: "instant" });
    // Wait for scroll/layout to complete so the rect we read in humanApproach is correct.
    await new Promise((r) => { requestAnimationFrame(() => requestAnimationFrame(r)); });
    await sleep(rand(200, 500));
    const { tx, ty } = await humanApproach(el);
    moveCursorTo(tx, ty);
    await sleep(320);            // let the cursor visibly arrive on the target
    cursorClick(tx, ty);
    await sleep(340);            // let the click ripple show before any navigation
    const opts = { bubbles: true, cancelable: true, clientX: tx, clientY: ty, button: 0 };
    el.dispatchEvent(new MouseEvent("mousedown", opts));
    await sleep(rand(40, 110));
    el.dispatchEvent(new MouseEvent("mouseup", opts));
    el.dispatchEvent(new MouseEvent("click", opts));
    return { ok: true, info: `Clicked [${i}] ${labelText(el)}` };
  };

  const typeIndex = async (i, text, submit, clear) => {
    const el = labeledElements[i];
    const stale = staleGuard(el, i); if (stale) return stale;
    const doClear = clear !== false; // default true
    el.scrollIntoView({ block: "center", behavior: "instant" });
    await sleep(rand(150, 400));
    const { tx, ty } = await humanApproach(el);
    cursorClick(tx, ty);
    el.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    el.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    el.focus();
    el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await sleep(rand(80, 200));

    // Either clear the field, or keep existing content and append at the end.
    let acc;
    if (doClear) {
      if (el.isContentEditable) el.textContent = "";
      else nativeSetValue(el, "");
      acc = "";
    } else {
      acc = el.isContentEditable ? (el.textContent || "") : (el.value || "");
    }
    el.dispatchEvent(new Event("input", { bubbles: true }));

    // Type one character at a time with jittered delays and full key events.
    for (const ch of text) {
      acc += ch;
      el.dispatchEvent(new KeyboardEvent("keydown", { key: ch, bubbles: true }));
      el.dispatchEvent(new KeyboardEvent("keypress", { key: ch, bubbles: true }));
      if (el.isContentEditable) el.textContent = acc;
      else nativeSetValue(el, acc);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new KeyboardEvent("keyup", { key: ch, bubbles: true }));
      // Occasional longer pause, as if thinking.
      await sleep(Math.random() < 0.12 ? rand(180, 420) : rand(45, 130));
    }
    el.dispatchEvent(new Event("change", { bubbles: true }));

    let submitNote = "";
    if (submit) {
      await sleep(rand(200, 500));
      const form = el.closest("form");
      const urlBefore = location.href;
      let handled = false;   // a page that handles Enter itself calls preventDefault()
      for (const type of ["keydown", "keypress", "keyup"]) {
        const ev = new KeyboardEvent(type, { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true });
        if (!el.dispatchEvent(ev)) handled = true;
      }
      if (form && !handled) {
        form.requestSubmit ? form.requestSubmit() : form.submit();
      } else if (form && handled) {
        // The page cancelled Enter. Chat/search UIs then act on it themselves;
        // login-style forms block it and expect the button; some pages submit
        // via their own (slow) fetch. We can't tell the last two apart, and an
        // automatic button click could send a login/purchase TWICE - so never
        // auto-click. If nothing visibly happened, say so: the model can click
        // the submit button explicitly.
        await sleep(700);
        const cur = el.isContentEditable ? (el.textContent || "") : (el.value || "");
        const nothingHappened = el.isConnected && form.isConnected && location.href === urlBefore && cur === acc;
        if (nothingHappened) {
          return { ok: true, info: `${doClear ? "Typed into" : "Appended to"} [${i}] and pressed Enter, but the page ` +
            "cancelled Enter and nothing visibly happened yet - it may NOT have submitted. Check with get_page, " +
            "and click the form's submit button if needed." };
        }
      } else if (!form && handled === false) {
        submitNote = " (Enter sent; no form found - check the page reacted)";
      }
    }
    return { ok: true, info: `${doClear ? "Typed into" : "Appended to"} [${i}]${submit ? " and submitted" + submitNote : ""}` };
  };

  // elementFromPoint that descends into same-origin iframes to the real target.
  const deepElementFromPoint = (x, y) => {
    let el = document.elementFromPoint(x, y);
    let cx = x, cy = y;
    while (el && (el.tagName === "IFRAME" || el.tagName === "FRAME")) {
      let doc = null;
      try { doc = el.contentDocument; } catch (e) { doc = null; }
      if (!doc) break;
      const r = el.getBoundingClientRect();
      cx -= r.left; cy -= r.top;
      const inner = doc.elementFromPoint(cx, cy);
      if (!inner || inner === el) break;
      el = inner;
    }
    return el;
  };

  const CLICKABLE =
    'a[href], button, input:not([type=hidden]), select, textarea, label, ' +
    '[role=button], [role=link], [role=radio], [role=checkbox], [role=tab], [role=menuitem], [onclick], [contenteditable=true]';

  // Resolve a point to the nearest clickable element and its top-level center.
  const resolvePoint = (x, y) => {
    const hit = deepElementFromPoint(x, y);
    if (!hit) return { ok: false, error: `Nothing at (${x}, ${y}).` };
    const target = hit.closest(CLICKABLE) || hit;
    const r = topLevelRect(target);
    return { ok: true, x: r.left + r.width / 2, y: r.top + r.height / 2, label: labelText(target) || "<" + target.tagName.toLowerCase() + ">", target };
  };

  // Vision click: click at the exact sighted coordinate. Event bubbling still
  // activates the right button/link even if the point is on a child element.
  // Vision coordinates are approximate. Snap a point to the nearest NUMBERED
  // element: prefer the smallest labeled element containing the point, else the
  // closest one within a small radius. Falls back to the raw point if none.
  const snapPoint = (x, y) => {
    const contains = [];
    let nearest = null, nearestD = 30; // reduced snap radius — 55px was grabbing wrong elements on dense pages
    for (const le of labeledElements) {
      if (!le || !le.isConnected) continue;
      const r = topLevelRect(le);
      if (r.width <= 0 || r.height <= 0) continue;
      if (x >= r.left && x <= r.left + r.width && y >= r.top && y <= r.top + r.height) {
        contains.push({ le, r, area: r.width * r.height });
      } else {
        const d = Math.hypot(r.left + r.width / 2 - x, r.top + r.height / 2 - y);
        if (d < nearestD) { nearestD = d; nearest = { le, r }; }
      }
    }
    let chosen = null;
    if (contains.length) { contains.sort((a, b) => a.area - b.area); chosen = contains[0]; }
    else if (nearest) chosen = nearest;
    if (!chosen) return { snapped: false, x, y, el: null };
    return { snapped: true, x: chosen.r.left + chosen.r.width / 2, y: chosen.r.top + chosen.r.height / 2, el: chosen.le, label: labelText(chosen.le) };
  };

  const clickAt = async (x, y) => {
    const snap = snapPoint(x, y);
    const el = snap.el || deepElementFromPoint(x, y);
    if (!el) return { ok: false, error: `Nothing at (${x}, ${y}). The point is likely outside the visible area — scroll or choose coordinates within the viewport.` };
    preventNewTab(el);
    const px = snap.snapped ? snap.x : x;
    const py = snap.snapped ? snap.y : y;
    el.scrollIntoView({ block: "nearest", behavior: "instant" });
    moveCursorTo(px, py);
    await sleep(300);
    cursorClick(px, py);
    await sleep(320);
    const opts = { bubbles: true, cancelable: true, view: window, clientX: px, clientY: py, button: 0 };
    el.dispatchEvent(new MouseEvent("mouseover", opts));
    el.dispatchEvent(new MouseEvent("mousemove", opts));
    el.dispatchEvent(new MouseEvent("mousedown", opts));
    await sleep(50);
    el.dispatchEvent(new MouseEvent("mouseup", opts));
    el.dispatchEvent(new MouseEvent("click", opts));
    const labelSrc = el.closest(CLICKABLE) || el;
    if (typeof labelSrc.focus === "function") labelSrc.focus();
    return snap.snapped
      ? { ok: true, info: `Clicked ${snap.label || "element"} (snapped from ${x},${y})` }
      : { ok: true, info: `Clicked at (${x}, ${y}) — ${labelText(labelSrc) || "<" + el.tagName.toLowerCase() + ">"}` };
  };

  // Set-of-marks: refresh the interactive-element list and return each one's
  // on-screen box + index, so the background can draw numbered labels on the
  // screenshot. The model then clicks by number (index) instead of guessing px.
  const collectMarks = () => {
    collectInteractive(); // refreshes labeledElements (same indices as click uses)
    const out = [];
    labeledElements.forEach((el, i) => {
      const r = topLevelRect(el);
      const onScreen = r.width > 0 && r.height > 0 &&
        r.left < window.innerWidth && r.top < window.innerHeight &&
        r.left + r.width > 0 && r.top + r.height > 0;
      if (onScreen) out.push({ i, x: Math.max(0, r.left), y: Math.max(0, r.top), w: r.width, h: r.height });
    });
    return out;
  };

  // Centers of two elements (top-level coords), for drag-and-drop.
  const centers = (fromI, toI) => {
    const a = labeledElements[fromI], b = labeledElements[toI];
    const sa = staleGuard(a, fromI); if (sa) return sa;
    const sb = staleGuard(b, toI); if (sb) return sb;
    const ra = topLevelRect(a), rb = topLevelRect(b);
    return { ok: true, sx: ra.left + ra.width / 2, sy: ra.top + ra.height / 2, tx: rb.left + rb.width / 2, ty: rb.top + rb.height / 2 };
  };

  // Drag element `fromI` onto `toI`. Fires pointer + mouse + HTML5 drag events
  // to cover native DnD and JS libraries (dnd-kit, react-beautiful-dnd, etc.).
  const dragEl = async (fromI, toI) => {
    const a = labeledElements[fromI], b = labeledElements[toI];
    const sa = staleGuard(a, fromI); if (sa) return sa;
    const sb = staleGuard(b, toI); if (sb) return sb;
    a.scrollIntoView({ block: "center", behavior: "instant" });
    const ra = topLevelRect(a), rb = topLevelRect(b);
    const sx = ra.left + ra.width / 2, sy = ra.top + ra.height / 2;
    const tx = rb.left + rb.width / 2, ty = rb.top + rb.height / 2;
    const pe = (el, type, x, y) => el.dispatchEvent(new PointerEvent(type, { bubbles: true, cancelable: true, composed: true, clientX: x, clientY: y, button: 0, buttons: type === "pointerup" ? 0 : 1, pointerId: 1, pointerType: "mouse", isPrimary: true }));
    const me = (el, type, x, y) => el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0, buttons: type === "mouseup" ? 0 : 1 }));

    moveCursorTo(sx, sy); await sleep(220);
    pe(a, "pointerdown", sx, sy); me(a, "mousedown", sx, sy);
    cursorClick(sx, sy);
    const steps = 14;
    for (let i = 1; i <= steps; i++) {
      const x = sx + (tx - sx) * i / steps, y = sy + (ty - sy) * i / steps;
      const mid = deepElementFromPoint(Math.round(x), Math.round(y)) || b;
      pe(mid, "pointermove", x, y); me(mid, "mousemove", x, y);
      moveCursorTo(x, y);
      await sleep(rand(18, 40));
    }
    await sleep(80);
    pe(b, "pointerup", tx, ty); me(b, "mouseup", tx, ty);

    // HTML5 drag-and-drop fallback.
    try {
      const dt = new DataTransfer();
      const de = (el, type, x, y) => el.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt, clientX: x, clientY: y }));
      de(a, "dragstart", sx, sy);
      de(b, "dragenter", tx, ty);
      de(b, "dragover", tx, ty);
      de(b, "drop", tx, ty);
      de(a, "dragend", tx, ty);
    } catch (e) {}

    return { ok: true, info: `Dragged [${fromI}] onto [${toI}]` };
  };

  // For trusted (CDP) input: scroll the element into view and return the
  // viewport coordinates of a point inside it, jittered off dead-center.
  const locate = (i) => {
    const el = labeledElements[i];
    const stale = staleGuard(el, i); if (stale) return stale;
    el.scrollIntoView({ block: "center", behavior: "instant" });
    const rect = topLevelRect(el);
    if (rect.width === 0 && rect.height === 0) return { ok: false, error: `Element ${i} is not visible.` };
    return {
      ok: true,
      x: rect.left + rect.width * rand(0.3, 0.7),
      y: rect.top + rect.height * rand(0.35, 0.65),
      label: labelText(el)
    };
  };

  const isScrollable = (el, cs) => {
    if (!el || el.nodeType !== 1) return false;
    cs = cs || (el.ownerDocument.defaultView || window).getComputedStyle(el);
    return (el.scrollHeight - el.clientHeight > 4) && /(auto|scroll|overlay)/.test(cs.overflowY);
  };

  // Nearest scrollable ancestor of an element (for scrolling a specific panel).
  const scrollerFor = (el) => {
    let node = el;
    while (node && node !== document.body && node !== document.documentElement) {
      try { if (isScrollable(node)) return node; } catch (e) {}
      node = node.parentElement || (node.getRootNode() && node.getRootNode().host);
    }
    return null;
  };

  // The biggest visible scrollable inner container (e.g. code editor / side panel).
  const largestScrollable = () => {
    let best = null, bestArea = 0;
    for (const doc of accessibleDocs()) {
      const view = doc.defaultView || window;
      let count = 0;
      for (const el of doc.querySelectorAll("div, section, main, article, ul, ol, pre")) {
        if (count++ > 4000) break;
        let cs; try { cs = view.getComputedStyle(el); } catch (e) { continue; }
        if (!isScrollable(el, cs)) continue;
        const r = el.getBoundingClientRect();
        const area = r.width * r.height;
        if (r.width > 40 && r.height > 40 && area > bestArea) { best = el; bestArea = area; }
      }
    }
    return best;
  };

  const stepScroll = async (scroller, dir) => {
    const docScroller = document.scrollingElement || document.documentElement;
    const viewH = scroller === docScroller ? window.innerHeight : scroller.clientHeight;
    const total = viewH * 0.8 * (dir === "up" ? -1 : 1);
    const before = scroller.scrollTop;
    const steps = Math.floor(rand(3, 6));
    for (let s = 0; s < steps; s++) {
      scroller.scrollTo({ top: scroller.scrollTop + total / steps, behavior: "smooth" });
      await sleep(rand(60, 140));
    }
    return Math.abs(scroller.scrollTop - before) > 2;
  };

  const scrollPage = async (dir, index) => {
    const docScroller = document.scrollingElement || document.documentElement;
    let scroller = docScroller;
    let usedInner = false;

    // If an element index is given, scroll the panel that element lives in.
    if (index !== undefined && index !== null && labeledElements[index]) {
      const s = scrollerFor(labeledElements[index]);
      if (s) { scroller = s; usedInner = true; }
    }

    let moved = await stepScroll(scroller, dir);

    // If the main page couldn't move, fall back to the largest inner panel.
    if (!moved && scroller === docScroller && (index === undefined || index === null)) {
      const inner = largestScrollable();
      if (inner) { scroller = inner; usedInner = true; moved = await stepScroll(inner, dir); }
    }

    const atBottom = Math.ceil(scroller.scrollTop + scroller.clientHeight) >= scroller.scrollHeight;
    const atTop = scroller.scrollTop <= 0;
    if (!moved) {
      return {
        ok: true, moved: false,
        info: dir === "up"
          ? "Already at the top — cannot scroll up further."
          : "Already at the bottom — cannot scroll down further."
      };
    }
    return { ok: true, moved: true, atTop, atBottom, info: `Scrolled ${dir}${usedInner ? " inside a panel" : ""}${atBottom ? " (reached the bottom)" : ""}` };
  };

  // Click the best-matching option in an open autocomplete/combobox list.
  const pickOption = async (i, value) => {
    const el = labeledElements[i];
    if (!el) return { ok: false, error: `No element with index ${i}. Call get_page first.` };
    if (!el.isConnected) return { ok: false, error: `Element [${i}] is stale — the page changed since the last get_page. Call get_page again to refresh the numbered list, then retry.` };
    const collect = (root) => Array.from(root.querySelectorAll('[role="option"], [role="menuitem"], li, [class*="option"]')).filter(isVisibleAnyDoc);
    let options = [];
    const owns = el && (el.getAttribute("aria-controls") || el.getAttribute("aria-owns"));
    const box = owns ? document.getElementById(owns) : null;
    if (box) options = collect(box);
    if (!options.length) {
      const boxes = Array.from(document.querySelectorAll('[role="listbox"], [role="menu"], ul[class*="dropdown"], div[class*="dropdown"], div[class*="menu"]')).filter(isVisibleAnyDoc);
      for (const b of boxes) options = options.concat(collect(b));
    }
    if (!options.length) options = Array.from(document.querySelectorAll('[role="option"], [role="menuitem"]')).filter(isVisibleAnyDoc);
    // de-dupe and drop empties
    options = [...new Set(options)].filter((o) => (o.textContent || "").trim());
    if (!options.length) return { ok: false, error: `No dropdown options appeared for "${value}". The field may need different text, or it's not an autocomplete.` };
    const v = String(value).toLowerCase().trim();
    const target = options.find((o) => (o.textContent || "").toLowerCase().trim() === v)
      || options.find((o) => (o.textContent || "").toLowerCase().includes(v))
      || options[0];
    target.scrollIntoView({ block: "nearest", behavior: "instant" });
    const r = topLevelRect(target);
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    moveCursorTo(cx, cy); await sleep(140); cursorClick(cx, cy); await sleep(120);
    const opts = { bubbles: true, cancelable: true, clientX: cx, clientY: cy, button: 0 };
    target.dispatchEvent(new MouseEvent("mouseover", opts));
    target.dispatchEvent(new MouseEvent("mousedown", opts));
    target.dispatchEvent(new MouseEvent("mouseup", opts));
    target.dispatchEvent(new MouseEvent("click", opts));
    return { ok: true, info: `Selected "${(target.textContent || "").trim().slice(0, 60)}" from the list` };
  };

  // Type into an autocomplete field then pick the matching suggestion.
  const selectOption = async (i, value) => {
    const el = labeledElements[i];
    const stale = staleGuard(el, i); if (stale) return stale;
    el.scrollIntoView({ block: "center", behavior: "instant" });
    el.focus(); el.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })); el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    if (el.isContentEditable) el.textContent = ""; else nativeSetValue(el, "");
    el.dispatchEvent(new Event("input", { bubbles: true }));
    let acc = "";
    for (const ch of String(value)) {
      acc += ch;
      el.dispatchEvent(new KeyboardEvent("keydown", { key: ch, bubbles: true }));
      if (el.isContentEditable) el.textContent = acc; else nativeSetValue(el, acc);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new KeyboardEvent("keyup", { key: ch, bubbles: true }));
      await sleep(rand(45, 95));
    }
    await sleep(750); // let suggestions load
    return pickOption(i, value);
  };

  // Read (and clear) a result an injected main-world script stored on <html>.
  const takeMainWorldResult = (key) => {
    const attr = "data-lmba-" + key;
    const raw = document.documentElement.getAttribute(attr);
    document.documentElement.removeAttribute(attr);
    if (raw == null) return undefined;
    try { return JSON.parse(raw); } catch (e) { return undefined; }
  };

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    (async () => {
      try {
        switch (msg.action) {
          case "get_page": sendResponse(getState()); break;
          case "locate": sendResponse(locate(msg.index)); break;
          case "click": sendResponse(await clickIndex(msg.index)); break;
          case "type": sendResponse(await typeIndex(msg.index, msg.text, msg.submit, msg.clear)); break;
          case "scroll": sendResponse(await scrollPage(msg.direction, msg.index)); break;
          case "cursor_move": moveCursorTo(msg.x, msg.y); sendResponse({ ok: true }); break;
          case "cursor_click": cursorClick(msg.x, msg.y); sendResponse({ ok: true }); break;
          case "hide_cursor": hideCursor(); sendResponse({ ok: true }); break;
          case "click_at": sendResponse(await clickAt(msg.x, msg.y)); break;
          case "resolve_point": { const rp = resolvePoint(msg.x, msg.y); sendResponse(rp.ok ? { ok: true, x: rp.x, y: rp.y, label: rp.label } : rp); break; }
          case "metrics": sendResponse({ w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio || 1 }); break;
          case "marks": sendResponse({ ok: true, marks: collectMarks(), w: window.innerWidth, h: window.innerHeight }); break;
          case "centers": sendResponse(centers(msg.from, msg.to)); break;
          case "select_option": sendResponse(await selectOption(msg.index, msg.value)); break;
          case "pick_option": sendResponse(await pickOption(msg.index, msg.value)); break;
          case "snap_point": { const sp = snapPoint(msg.x, msg.y); sendResponse({ snapped: sp.snapped, x: sp.x, y: sp.y, label: sp.label || "" }); break; }
          case "describe": {
            const el = deepElementFromPoint(msg.x, msg.y);
            if (!el) { sendResponse({ ok: false, label: "" }); break; }
            const src = el.closest(CLICKABLE) || el;
            sendResponse({ ok: true, label: labelText(src) || "<" + el.tagName.toLowerCase() + ">" });
            break;
          }
          case "drag": sendResponse(await dragEl(msg.from, msg.to)); break;
          // Return the screen rectangle of the code editor container (for CDP targeting).
          case "editor_rect": {
            for (const doc of accessibleDocs()) {
              // Priority 1: CodeMirror 6 contentEditable (.cm-content is the focus target).
              {
                const cmContent = doc.querySelector('.cm-editor .cm-content');
                if (cmContent && cmContent.offsetParent) {
                  const r = topLevelRect(cmContent);
                  sendResponse({ x: r.left, y: r.top, w: r.width, h: r.height });
                  return;
                }
              }

              // Priority 2: Monaco / CodeMirror legacy.
              const el = doc.querySelector('.monaco-editor, .CodeMirror');
              if (el) {
                const r = topLevelRect(el);
                sendResponse({ x: r.left, y: r.top, w: r.width, h: r.height });
                return;
              }
            }

            // Fallback: choose the editor iframe (name="frame" for TripleTen's Monaco container).
            let bestScore = -Infinity, bestRect = null;
            const candidates = [];
            for (const frame of document.querySelectorAll("iframe")) {
              const r = topLevelRect(frame);
              if (r.width > 200 && r.height > 150) {
                let hint = "";
                try { hint += " " + (frame.src || ""); } catch (e) {}
                try { hint += " " + (frame.title || ""); } catch (e) {}
                try { hint += " " + (frame.name || ""); } catch (e) {}
                try { hint += " " + (frame.id || ""); } catch (e) {}
                try { hint += " " + (frame.className || ""); } catch (e) {}
                const area = r.width * r.height;

                // Hard disqualify: anything that clearly looks like the site preview / rendered output.
                if (/preview|result|output|file-keeper|localhost|webcontainer/i.test(hint)) {
                  candidates.push({ x: Math.round(r.left), y: Math.round(r.top), w: r.width, h: r.height, hint: hint.trim().slice(0, 80), score: "DISQUALIFIED" });
                  continue;
                }
                // Also penalize frames on the right half (typical for live previews).
                const isRight = r.left > window.innerWidth * 0.5;

                let score = area;
                if (/editor|monaco|code|workspace|sandbox/i.test(hint)) score += area * 2;
                if (!isRight) score += area * 0.35; // prefer left-side workspaces
                if (isRight) score -= area * 1.5;   // penalize preview-like right side

                candidates.push({ x: Math.round(r.left), y: Math.round(r.top), w: r.width, h: r.height, hint: hint.trim().slice(0, 80), score });
                if (score > bestScore) {
                  bestScore = score;
                  bestRect = { x: r.left, y: r.top, w: r.width, h: r.height };
                }
              }
            }

            // Log the decision so we can debug wrong choices.
            if (candidates.length > 0) {
              candidates.sort((a, b) => (typeof b.score === "number" ? b.score : -Infinity) - (typeof a.score === "number" ? a.score : -Infinity));
              console.log("[EDITOR_RECT] Iframe candidates (sorted):", JSON.stringify(candidates.slice(0, 6)));
            }

            // Last resort: only pick an iframe if it wasn't explicitly disqualified as a preview/output frame.
            if (!bestRect && candidates.length > 0) {
              let bestArea = 0;
              for (const c of candidates) {
                if (c.score === "DISQUALIFIED") continue; // never fall back to a known preview frame
                const area = c.w * c.h;
                if (area > bestArea) {
                  bestArea = area;
                  bestRect = { x: c.x, y: c.y, w: c.w, h: c.h };
                }
              }
            }

            if (bestRect) {
              console.log("[EDITOR_RECT] Using fallback iframe rect:", JSON.stringify(bestRect));
              sendResponse(bestRect);
            } else {
              console.warn("[EDITOR_RECT] No editor rect found on this page.");
              sendResponse({});
            }
            break;
          }
          case "upload_file": {
            const el = labeledElements[msg.index];
            if (!el) { sendResponse({ ok: false, error: `No element with index ${msg.index}` }); break; }
            if (el.tagName !== "INPUT" || el.type !== "file") {
              sendResponse({ ok: false, error: `Element [${msg.index}] is not a file input` });
              break;
            }
            const path = String(msg.path || "").trim();
            if (!path) { sendResponse({ ok: false, error: "No file path provided" }); break; }
            // We can't read arbitrary local files directly (no FS API in content scripts),
            // but we can set the value programmatically for testing / same-origin scenarios.
            // For a real upload you'd need to use nativeFilesystem or prompt the user with
            // a file picker. Here we synthesize a File object matching the requested path name:
            const fileName = path.replace(/^.*[/\\]/, "");
            try {
              const dt = new DataTransfer();
              // Create a minimal dummy file matching the requested path (user must ensure it's what they expect).
              const dummyFile = new File(["__UPLOAD_FILE:" + path + "__"], fileName, { type: "application/octet-stream" });
              dt.items.add(dummyFile);
              el.files = dt.files;
              el.dispatchEvent(new Event("change", { bubbles: true }));
              sendResponse({ ok: true, info: `Set file "${fileName}" on [${msg.index}] (dummy — content scripts can't read arbitrary local files directly)` });
            } catch (e) {
              sendResponse({ ok: false, error: "Could not set file: " + e.message });
            }
            break;
          }
          // Diagnostic: run a main-world scan using dynamic script injection and report results.
          case "editor_inject_scan": {
            new Promise((resolve) => {
              const s = document.createElement("script");
              s.textContent = `
                (function(){
                  window.__lmba_editor_diag = [];
                  try{
                    if(window.monaco && window.monaco.editor){
                      const editors = window.monaco.editor.getEditors();
                      for(const ed of editors){
                        try{
                          const m = ed.getModel();
                          if(m) { window.__lmba_editor_diag.push("MONACO_API_FOUND:" + String(m.getValue()).length + "chars"); break; }
                        }catch(_){}
                      }
                    }else{ window.__lmba_editor_diag.push("NO_MONACO_API_IN_THIS_CONTEXT"); }
                  }catch(e){ window.__lmba_editor_diag.push("SCAN_ERROR:" + e.message); }
                  document.documentElement.setAttribute("data-lmba-diag", JSON.stringify(window.__lmba_editor_diag));
                })();
              `;
              document.documentElement.appendChild(s);
              s.remove();
              setTimeout(() => resolve(takeMainWorldResult("diag") || ["NO_RESULT (inline scripts blocked by the page's CSP?)"]), 150);
            }).then((result) => {
              console.log("[EDITOR_INJECT_SCAN]", result.join(";"));
              sendResponse({ ok: true, info: result.join(";") });
            });
            break;
          }
          case "get_editor": {
            // First try main-world via script tag (can reach same-origin iframes better than chrome.scripting)
            new Promise((resolve) => {
              const s = document.createElement("script");
              s.textContent = `
                (function(){
                  window.__lmba_get_editor_result = null;
                  const scan = (doc, depth) => {
                    if(!doc || depth > 6) return;
                    try{ for(const f of Array.from(doc.querySelectorAll("iframe")).slice(0,25)){
                      let d; try{d=f.contentDocument||f.contentWindow.document;}catch(_){continue;}
                      scan(d,depth+1); if(window.__lmba_get_editor_result) return; } }catch(_) {}
                    try{
                      const win = doc.defaultView || window;
                      if(!win.monaco||!win.monaco.editor) return;
                      for(const ed of win.monaco.editor.getEditors()){
                        const m=ed.getModel();
                        if(m){ window.__lmba_get_editor_result = String(m.getValue()||""); break; }
                      }
                    }catch(_) {}
                  };
                  scan(window.document,0);
                  document.documentElement.setAttribute("data-lmba-get", JSON.stringify(window.__lmba_get_editor_result));
                })();
              `;
              document.documentElement.appendChild(s);
              s.remove();
              setTimeout(() => resolve(takeMainWorldResult("get")), 150);
            }).then((result) => {
              if (typeof result === "string") { sendResponse({ ok: true, content: result }); return; }
              // Fallback to content-script path.
              const code = getCodeEditorContent();
              if (code === null) { sendResponse({ ok: false, info: "No code editor detected on this page." }); return; }
              sendResponse({ ok: true, content: code });
            });
            break;
          }
          // Called from background.js after main-world Monaco write attempt.
          // Handles CodeMirror / simple editors as fallback (content scripts can't reach main-world Monaco).
          case "set_editor_content_mainworld_done": {
            const target = String(msg.content || "");

            // Try dynamic script injection first: runs in page's main world and can reach Monaco inside iframes.
            new Promise((resolve) => {
              const s = document.createElement("script");
              s.textContent = `
                (function(val){
                  window.__lmba_set_editor_result = null;
                  const scan = (doc, depth) => {
                    if(!doc || depth > 6) return;
                    try{ for(const f of Array.from(doc.querySelectorAll("iframe")).slice(0,25)){
                      let d; try{d=f.contentDocument||f.contentWindow.document;}catch(_){continue;}
                      scan(d,depth+1); if(window.__lmba_set_editor_result) return; } }catch(_) {}
                    try{
                      const win = doc.defaultView || window;
                      if(!win.monaco||!win.monaco.editor) return;
                      for(const ed of win.monaco.editor.getEditors()){
                        const m=ed.getModel();
                        if(m){ m.setValue(val); window.__lmba_set_editor_result = {type:"monaco-api",chars:String(val).length}; break; }
                      }
                    }catch(_) {}
                  };
                  scan(window.document,0);
                  document.documentElement.setAttribute("data-lmba-set", JSON.stringify(window.__lmba_set_editor_result));
                })(${JSON.stringify(target)});
              `;
              document.documentElement.appendChild(s);
              s.remove();
              // NOTE: if this result is lost the textarea fallback below would write
              // the content a SECOND time into an editor that already has it.
              setTimeout(() => resolve(takeMainWorldResult("set") || null), 200);
            }).then((result) => {
              if (result) { sendResponse({ ok: true, type: result.type, info: `Wrote ${result.chars} chars to Monaco editor via main world` }); return; }

              // Fallback to content-script path for CodeMirror / simple editors.
              for (const doc of accessibleDocs()) {
                const cmTa = doc.querySelector('.CodeMirror textarea');
                if (cmTa && typeof cmTa.value === "string") {
                  nativeSetValue(cmTa, target);
                  cmTa.dispatchEvent(new Event("input", { bubbles: true }));
                  sendResponse({ ok: true, type: "codemirror", info: `Wrote ${target.length} chars to CodeMirror editor` });
                  return;
                }

                const monacoTa = doc.querySelector('.monaco-editor textarea');
                if (monacoTa && typeof monacoTa.value === "string") {
                  nativeSetValue(monacoTa, target);
                  monacoTa.dispatchEvent(new Event("input", { bubbles: true }));
                  monacoTa.dispatchEvent(new Event("change", { bubbles: true }));
                  sendResponse({ ok: true, type: "monaco-ta", info: `Wrote ${target.length} chars to Monaco editor (textarea fallback)` });
                  return;
                }
              }

              sendResponse({ ok: false, error: "Could not find a writable code editor on this page." });
            });
            break;
          }
          default: sendResponse({ ok: false, error: "unknown action " + msg.action });
        }
      } catch (e) {
        sendResponse({ ok: false, error: String(e && e.message || e) });
      }
    })();
    return true;
  });
})();
