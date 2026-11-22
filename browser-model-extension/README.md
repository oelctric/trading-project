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
