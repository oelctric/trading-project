# Local Model Browser Agent

A Chrome extension that lets your local LLM (LM Studio or any OpenAI-compatible
endpoint) *see and control the browser*. You give it a task in plain language;
the model reads the page, clicks, types, and navigates through tool calls until
it produces an answer.

## Install (unpacked)

1. Open `chrome://extensions`
2. Toggle **Developer mode** on (top-right)
3. Click **Load unpacked**
4. Select this folder: `Desktop/browser-model-extension`
5. Pin the extension and click its icon to open the side panel
   (or right-click the toolbar icon → the panel opens on the active tab).

## Configure

In the side panel's **Connection** section:

- **Model endpoint** — the OpenAI-compatible base URL. For your LM Studio server:
  `http://192.168.50.213:1234/v1`
- **Model name** — the model id loaded in LM Studio (e.g. `local-model`, or whatever
  `GET /v1/models` reports). Click **Test connection** to list available models.
- **API key** — leave blank for LM Studio.

Click **Test connection** to confirm the browser can reach the server.

## Requirements for the model

Browser control uses **OpenAI-style tool/function calling**. The model you load in
LM Studio must support tool calls (most modern instruct models do — e.g. Qwen,
Llama-3.x-Instruct, Hermes). LM Studio must have the local server running.

If the server is on another machine (192.168.50.213), make sure LM Studio is set to
serve on the network (not just localhost) and the Windows firewall allows it.

## Use

1. Open a normal web page (not `chrome://` pages — extensions can't script those).
2. Type a task, e.g. *"Find the top story on Hacker News and tell me its title and points."*
3. Click **Run**. Watch the log: grey = model thinking, blue = tool call,
   green = tool result, yellow = final answer.

## Tools the model can call

| Tool | What it does |
|------|--------------|
| `get_page` | Read URL, title, visible text, and a numbered list of clickable elements |
| `click` | Click an element by its index |
| `type` | Type text into an input (optionally submit) |
| `navigate` | Go to a URL |
| `scroll` | Scroll up/down one screen |
| `screenshot` | Capture the visible tab as an image for vision models |
| `done` | Finish and return the answer |

### Screenshots are never saved

The `screenshot` tool uses `chrome.tabs.captureVisibleTab`, which returns an
in-memory data URL. That image is placed directly into the message sent to your
model and is **never written to disk, `chrome.storage`, downloads, or the log**.
It only exists for the duration of that model call. Your model must be a
vision-capable model for this to be useful (e.g. a Qwen-VL / Llama-Vision loaded
in LM Studio).

### Confirm each action

Enable **Confirm each action** in the Connection section to get a browser prompt
before every `click`, `type`, or `navigate`. Declining tells the model the action
was rejected so it can try something else. Read-only tools (`get_page`, `scroll`,
`screenshot`) run without a prompt.

## Notes & safety

- The agent acts on your **currently active tab** with your logged-in sessions.
  Only give it tasks you're comfortable running under your own account.
- Element indices come from the latest `get_page`; the system prompt tells the model
  to refresh them after navigation.
- Max agent steps is capped (default 15) to prevent runaway loops. Use **Stop** anytime.
- Everything runs locally between your browser and your model — no cloud calls.


## Bug fixes (Nov 2026)

- Stop, then sending a new task, could resume the old run (two agents in one tab). Runs now
  have ids, and Stop aborts the in-flight model request.
- Streaming "thinking" text was appended cumulatively ("a", "a ab", ...) onto one stale line
  and then logged again in full. It now updates one line per model call.
- Text-format `select_option` / `drag` tool calls were ignored by the parser.
- The page-stability poller could run forever on a tab stuck "loading".
- Side panel: Stop did nothing if the port had dropped; Test ignored the selected model.

- **Password values were sent to the model** (and stored in chat history/debug log): page
  snapshots showed every input's current value. Password fields now show only "(filled)".
- "type ... submit" fired Enter AND `form.requestSubmit()` -> searches/messages/posts sent twice
  on pages that handle Enter. Now submits only if the page didn't handle Enter.
- Typing into a `<select>` (or a field in an iframe) threw: the value setter came from the top
  window's `HTMLInputElement`. Now uses the element's own window and type.
- `drag` skipped the stale-element check.
- Main-world editor scripts returned results through `window.__lmba_*`, which an isolated content
  script can never see -> reads always fell back, and writes could be applied TWICE (main world
  succeeded, then the textarea fallback wrote again). Results now pass via a DOM attribute.

Tests: `node tests/background.test.js` (offline, mocked `chrome`);
`JSDOM_PATH=<dir with node_modules/jsdom> node tests/content.test.js` (DOM tests; they fail on
the old code - e.g. the old snapshot leaked the test password).

Second code review (Nov 2026):
- Stop pressed while a new run was queued behind the old one was lost; two quick Runs could
  start together. Runs are now strictly queued with tickets - any Stop / New chat / newer Run
  invalidates a queued or running one. New chat waits for the old run to finish logging, and a
  cancelled run can no longer write into the new chat.
- "type ... submit" trusted preventDefault alone: login-style forms that block Enter were never
  submitted but reported "submitted". Now, if nothing happened after Enter, the form's submit
  button is pressed (and the result says so).
- waitUntilStable returned instantly on already-loaded (single-page-app) tabs, so the next read
  could see a stale DOM; it now waits for the DOM to go quiet (max 2s).

Third code review: the round-2 "press the submit button if Enter did nothing" fallback could
submit a login/checkout TWICE on pages that cancel Enter and submit via their own slow fetch.
It no longer auto-clicks; it reports "may NOT have submitted" so the model clicks explicitly.
A run replaced by a new message no longer shows "Stopped by user." (it says it was replaced).

Fourth code review: stopping during a batch of tool calls left some tool_calls unanswered, so
strict OpenAI-compatible servers rejected the next request (400) - now every unrun call gets a
"cancelled" reply (`tests/stop_midbatch.test.js`, fails on the old code). The stop reason is now
recorded per run, so Stop + a quick new message still reports "Stopped by user.".

Eighth code review: the per-run stop-reason map is cleaned up (it used to grow forever).

Ninth code review: secret fields stay hidden after a "show password" toggle (type=password ->
text), and autocomplete password / one-time-code / CVC fields and secret-looking names (pin, otp,
token...) are redacted too. A Stop that cancels a queued task reports "Stopped by user."; cleanup
lines like "Debugger detached." are shown even for a stopped run.
