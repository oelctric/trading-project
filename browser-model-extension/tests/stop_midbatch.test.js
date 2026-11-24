// Stop in the middle of a batch of tool calls must leave NO unanswered tool_call
// (strict OpenAI-compatible servers reject the next request otherwise).
// Run: node tests/stop_midbatch.test.js
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const store = {};
let ctxRef = null;
const chrome = {
  storage: { local: { get: async (d) => ({ ...d, ...store }), set: async (o) => { Object.assign(store, o); } },
             onChanged: { addListener() {} } },
  runtime: { onConnect: { addListener() {} }, sendMessage: async () => {}, lastError: null },
  tabs: {
    query: async () => [{ id: 1, windowId: 1 }],
    get: async () => ({ status: "complete", windowId: 1 }),
    onUpdated: { addListener() {}, removeListener() {} },
    sendMessage: (tabId, msg, opts, cb) => {
      // the FIRST tool (get_page) is running when the user presses Stop
      if (msg.action === "get_page") ctxRef.cancelRuns("stop");
      setTimeout(() => cb({ ok: true, url: "x", text: "page", count: 0 }), 5);
    },
  },
  scripting: { executeScript: async () => [{ result: 0 }] },
  debugger: {},
};
const sse = [
  { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "get_page", arguments: "{}" } }] } }] },
  { choices: [{ delta: { tool_calls: [{ index: 1, id: "c2", function: { name: "scroll", arguments: "{\"direction\":\"down\"}" } }] } }] },
].map((j) => "data: " + JSON.stringify(j) + "\n").join("") + "data: [DONE]\n";
const fetchImpl = async () => new Response(sse, { status: 200 });
const ctx = { chrome, console, setTimeout, clearTimeout, setInterval, clearInterval, fetch: fetchImpl,
              AbortController, TextDecoder, TextEncoder, URL, Response };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8") +
  "\n;globalThis.cancelRuns = cancelRuns; globalThis.__t = { runAgent, getConv: () => conversation, getRun: () => currentRun, cancelReasons };", ctx);
ctxRef = ctx;

(async () => {
  const T = ctx.__t;
  // 1. stop mid-batch -> every tool_call answered
  const ticket = T.getRun();
  await T.runAgent({ baseUrl: "http://x/v1", model: "m", maxSteps: 3, vision: false }, "do it", null, ticket);
  const conv = T.getConv();
  const asst = conv.find((m) => m.role === "assistant" && (m.tool_calls || []).length === 2);
  assert.ok(asst, "assistant turn with 2 tool calls not recorded");
  const replies = new Set(conv.filter((m) => m.role === "tool").map((m) => m.tool_call_id));
  assert.ok(replies.has("c1") && replies.has("c2"), "unanswered tool_call: " + [...replies]);
  console.log("ok  stop mid-batch leaves no unanswered tool_call");
  // 2. the stop reason is per run: Stop then a quick new message keeps "stop" for that run
  const r = T.getRun();
  ctx.cancelRuns("stop");
  ctx.cancelRuns("superseded");
  assert.strictEqual(T.cancelReasons[r], "stop");
  console.log("ok  stop reason recorded per run (not overwritten by a later cancel)");
})().catch((e) => { console.error("FAIL", e); process.exit(1); });
