// DOM tests for content.js fixes. Needs jsdom:
//   JSDOM_PATH=<dir containing node_modules/jsdom> node tests/content.test.js
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const { JSDOM } = require(path.join(process.env.JSDOM_PATH || "", "node_modules", "jsdom"));

function load(html) {
  const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  let handler = null;
  w.chrome = { runtime: { onMessage: { addListener: (f) => { handler = f; } } } };
  // jsdom has no layout: give every element a visible box
  w.Element.prototype.getBoundingClientRect = () => ({ left: 10, top: 10, width: 100, height: 20, right: 110, bottom: 30 });
  w.Element.prototype.scrollIntoView = () => {};
  w.requestAnimationFrame = (f) => setTimeout(f, 0);
  w.document.elementFromPoint = () => null;   // no layout engine in jsdom
  w.eval(fs.readFileSync(process.env.CONTENT_JS || path.join(__dirname, "..", "content.js"), "utf8"));
  const send = (msg) => new Promise((res) => handler(msg, {}, res));
  return { w, send };
}

(async () => {
  // 1. password values never reach the model
  {
    const { w, send } = load(`<form><input id="u" value="alice"><input id="p" type="password" placeholder="Password" value="hunter2"></form>`);
    const st = await send({ action: "get_page" });
    assert.ok(!st.text.includes("hunter2"), "password leaked: " + st.text);
    assert.ok(st.text.includes("Password (filled)"), st.text);
    assert.ok(st.text.includes("alice"));  // normal fields still visible
    console.log("ok  password value hidden from page snapshot");
  }
  // 1b. "show password" toggle (type=password -> text) and OTP / autocomplete fields stay hidden
  {
    const { w, send } = load(`<input id="p" type="password" value="hunter2"><input id="o" autocomplete="one-time-code" value="482913"><input id="pin" name="card_pin" value="1234">`);
    await send({ action: "get_page" });
    w.document.getElementById("p").setAttribute("type", "text");      // user clicked the eye icon
    const st = await send({ action: "get_page" });
    for (const s of ["hunter2", "482913", "1234"]) assert.ok(!st.text.includes(s), "leaked " + s + ": " + st.text);
    console.log("ok  toggled password / OTP / PIN values stay hidden");
  }
  // 2. submit: page handles Enter itself -> form must NOT be submitted again
  {
    const { w, send } = load(`<form id="f"><input id="q"></form>`);
    let submits = 0;
    w.document.getElementById("f").addEventListener("submit", (e) => { e.preventDefault(); submits++; });
    // chat-style page: handles Enter itself, sends, and clears the box
    w.document.getElementById("q").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submits++; e.target.value = ""; } });
    await send({ action: "get_page" });
    const r = await send({ action: "type", index: 0, text: "hi", submit: true });
    assert.ok(r.ok, JSON.stringify(r));
    assert.strictEqual(submits, 1, "submitted " + submits + " times");
    console.log("ok  no double submit when the page handles Enter");
  }
  // 2b. form that blocks Enter (login-style, or submits via its own fetch)
  {
    const { w, send } = load(`<form id="f"><input id="u"><button type="submit" id="b">Log in</button></form>`);
    let submits = 0;
    w.document.getElementById("f").addEventListener("submit", (e) => { e.preventDefault(); submits++; });
    w.document.getElementById("u").addEventListener("keydown", (e) => { if (e.key === "Enter") e.preventDefault(); });
    await send({ action: "get_page" });
    const r = await send({ action: "type", index: 0, text: "bob", submit: true });
    // never auto-clicks (a page submitting via its own slow fetch would be sent TWICE);
    // instead it reports honestly so the model can click the button itself
    assert.strictEqual(submits, 0, "auto-submitted " + submits + " times");
    assert.ok(/may NOT have submitted/.test(r.info), r.info);
    console.log("ok  form that blocks Enter: no auto-click, honest 'may not have submitted' report");
  }
  // 3. typing into a <select> no longer throws
  {
    const { send } = load(`<select id="s"><option value="a">A</option><option value="b">B</option></select>`);
    await send({ action: "get_page" });
    const r = await send({ action: "type", index: 0, text: "b" });
    assert.ok(r.ok, JSON.stringify(r));
    console.log("ok  typing into <select> works");
  }
  // 4. drag on a stale element is refused
  {
    const { w, send } = load(`<button id="a">A</button><button id="b">B</button>`);
    await send({ action: "get_page" });
    w.document.getElementById("a").remove();
    const r = await send({ action: "drag", from: 0, to: 1 });
    assert.ok(!r.ok && /stale/.test(r.error), JSON.stringify(r));
    console.log("ok  drag refuses stale elements");
  }
  process.exit(0);
})().catch((e) => { console.error("FAIL", e); process.exit(1); });
