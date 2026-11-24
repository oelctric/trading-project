// Offline tests for background.js fixes. Run: node tests/background.test.js
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const store = {};
const listeners = [];
const chrome = {
  storage: { local: {
    get: async (d) => ({ ...d, ...store }),
    set: async (o) => { Object.assign(store, o); },
  }, onChanged: { addListener() {} } },
  runtime: { onConnect: { addListener: (f) => listeners.push(f) }, sendMessage: async () => {}, lastError: null },
  tabs: { query: async () => [], get: async () => ({ status: "complete" }), onUpdated: { addListener() {}, removeListener() {} } },
  scripting: { executeScript: async () => [] },
  debugger: {},
};
let fetchImpl = null;
const ctx = { chrome, console, setTimeout, clearTimeout, setInterval, clearInterval,
  fetch: (...a) => fetchImpl(...a), AbortController, TextDecoder, TextEncoder, URL };
vm.createContext(ctx);
const src = fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8");
vm.runInContext(src + "\n;globalThis.__t = { parseTextToolCall, logThinkAppend, beginThinkStream, endThinkStream, callModel, cancelRuns, getTranscript: () => transcript, getRun: () => currentRun, TOOLS, ACTIONS_NEED_STABLE };", ctx);
const T = ctx.__t;

(async () => {
  // 1. text-form tool calls for drag/select_option now recognized
  assert.deepStrictEqual(JSON.parse(JSON.stringify(T.parseTextToolCall('{"name":"select_option","arguments":{"index":3,"value":"Blue"}}'))),
    { name: "select_option", args: { index: 3, value: "Blue" } });
  assert.strictEqual(T.parseTextToolCall("drag(4, 9)").name, "drag");
  assert.deepStrictEqual(JSON.parse(JSON.stringify(T.parseTextToolCall("drag(4, 9)").args)), { from: "4", to: "9" });
  console.log("ok  text tool calls (select_option, drag)");
  const names = T.TOOLS.map((x) => x.function.name);
  assert.ok(names.includes("drag"), "drag missing from TOOLS schema");
  assert.ok(T.ACTIONS_NEED_STABLE.has("drag"));
  console.log("ok  drag offered to tool-calling models + stable-wait");

  // 2. streaming think line: replaced, not cumulatively appended; one line per call
  T.beginThinkStream();
  T.logThinkAppend("a"); T.logThinkAppend("a b"); T.logThinkAppend("a b c");
  T.endThinkStream("a b c d");
  let thinks = T.getTranscript().filter((e) => e.kind === "think");
  assert.strictEqual(thinks.length, 1); assert.strictEqual(thinks[0].text, "a b c d");
  T.beginThinkStream(); T.logThinkAppend("second"); T.endThinkStream("second call");
  thinks = T.getTranscript().filter((e) => e.kind === "think");
  assert.strictEqual(JSON.stringify(thinks.map((e) => e.text)), JSON.stringify(["a b c d", "second call"]));
  T.beginThinkStream(); T.logThinkAppend("click(3)"); T.endThinkStream("");   // tool-as-text: removed
  assert.strictEqual(T.getTranscript().filter((e) => e.kind === "think").length, 2);
  console.log("ok  live think line (no duplication, one per call)");

  // 3. Stop aborts an in-flight model call
  fetchImpl = (url, opts) => new Promise((_, rej) => {
    opts.signal.addEventListener("abort", () => rej(Object.assign(new Error("The operation was aborted"), { name: "AbortError" })));
  });
  const p = T.callModel({ baseUrl: "http://x/v1", model: "m" }, [{ role: "system", content: "s" }], null);
  const runBefore = T.getRun();
  setTimeout(() => T.cancelRuns(), 20);
  await assert.rejects(p, /aborted by user/);
  assert.ok(T.getRun() > runBefore);   // older runs are invalidated, can't resume
  console.log("ok  stop aborts model call and invalidates the run");

  // 4. Stop while a new run is QUEUED behind an old one: the queued run must not start
  {
    const port = { name: "agent", onMessage: { addListener(f) { port.h = f; } }, postMessage() {}, onDisconnect: { addListener() {} } };
    listeners[0](port);
    let release;
    // occupy the chain with a fake "old run"
    vm.runInContext("runChain = new Promise((r) => { globalThis.__rel = r; });", ctx);
    release = ctx.__rel;
    let started = false;
    ctx.runAgent = async () => { started = true; };
    vm.runInContext("runAgent = globalThis.runAgent;", ctx);
    const pRun = port.h({ type: "run", cfg: {}, task: "x" });
    await new Promise((r) => setTimeout(r, 10));
    port.h({ type: "stop" });          // Stop arrives while the run waits in the queue
    release();
    await pRun;
    assert.strictEqual(started, false, "queued run started despite Stop");
    const last = ctx.__t.getTranscript().slice(-1)[0];
    assert.ok(last && /Stopped by user/.test(last.text), "queued Stop not reported: " + JSON.stringify(last));
    console.log("ok  Stop cancels a run that is still queued");
  }
})().catch((e) => { console.error("FAIL", e); process.exit(1); });
