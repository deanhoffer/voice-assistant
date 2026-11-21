# Local LLM + voice assistant (osprey)

Self-hosted Qwen3.6-35B-A3B (llama.cpp, Vulkan) + Whisper speech-to-text + an always-on voice assistant. Everything runs on localhost; nothing leaves the machine.

## Commands
```
ai-toggle             # toggle llama-server + voice assistant on/off
ai-toggle on|off      # start / stop (off frees most of the VRAM + ~10 GiB RAM for gaming); on also starts whisper, off leaves it
whisper [on|off|status] # whisper-server alone (also used by Campfire push-to-talk, SillyTavern); full gaming: ai-toggle off && whisper off
ai-toggle status      # unit states, memory, VRAM
ai-toggle log         # follow the voice assistant: transcripts, (ignored) lines, replies, commands
ai-toggle mem [topic] # list the assistant's memory topics / edit one
ai-toggle chat        # type to the assistant; same session as voice (a voice request opens this in a foot window)
aider                 # coding agent in the current git repo, uses the local model
```
Voice: say **"Osprey, <request>"** (or "Osprey", pause, then the request within 8 s). Everything else is transcribed and ignored. Mute the mic to stop listening entirely. Replies appear as notifications (mako) and in `ai-toggle log`. Shell commands only run after you pick **Yes, run it** in the fuzzel popup; sudo commands are never run, you get the command instead. Say **"Osprey, new session"** (or "new topic", optionally followed by a request) to clear the conversation; memory is kept. It also clears itself after 10 min idle. Say **"Osprey, think: <request>"** (or "Osprey, think", then the request) for Qwen's thinking mode on that one request: better reasoning, ~3x slower (about 20 s).

## Pieces
| Piece | Where |
|---|---|
| llama-server, `127.0.0.1:8180` (OpenAI API at `/v1`) | `~/.config/systemd/user/llama-server.service` |
| whisper-server, `127.0.0.1:8181` (`/inference`) | `~/.config/systemd/user/whisper-server.service` |
| voice assistant | `~/dev/repos/voice-assistant/voice-assistant.py`, `voice-assistant.service` |
| group / failure popup | `local-ai.target` (llama-server, voice-assistant; whisper-server comes in via voice-assistant's `Wants=` and isn't stopped with it), `ai-failed-notify.service` |
| CLI | `~/dev/repos/voice-assistant/ai-toggle` → `~/.local/bin/ai-toggle`, `…/whisper` → `~/.local/bin/whisper` |
| models | `~/models/llm/`, `~/models/whisper/` |
| aider config | `~/.aider.conf.yml`, `~/.aider.model.metadata.json` |
| Continue (VS Code) | `~/.continue/config.yaml`; `~/.continue/rules/dean.md` → assistant memory `dean.md` |
| OOM killer | `/etc/default/earlyoom` (kills AI servers first) |

Started at login from the sway config (`exec systemctl --user start local-ai.target`). Port 8080 is Caddy (video server), hence 8180/8181.

## Tuning
- **VRAM split:** `--n-cpu-moe` in `llama-server.service` = number of expert layers (of 40, ~0.45 GiB each) kept in system RAM. Target ≈14 GiB total VRAM with everything running (`ai-toggle status`). Lower it for speed if there's headroom, raise it if VRAM runs out. Then `systemctl --user daemon-reload && ai-toggle off && ai-toggle on`.
- **GPU pinning:** `GGML_VK_VISIBLE_DEVICES=0` in both server units = RX 7800 XT (`llama-server --list-devices`; Vulkan1 is the iGPU).
- **Wake word:** `WAKE_WORDS` at the top of `voice-assistant.py`, then `systemctl --user restart voice-assistant`.
- **Speech detection:** if background noise keeps triggering transcriptions (lots of `(ignored)` noise in the log), raise `MIN_RMS` / `SPEECH_FACTOR`; if your voice is missed, lower them.
- **Thinking mode:** off by default (faster). Remove `--chat-template-kwargs '{"enable_thinking":false}'` from `llama-server.service` to turn it on for everything; Qwen recommends `--temp 0.6 --top-p 0.95 --presence-penalty 0` for thinking-mode coding.
- **Coding client:** aider auto-applies edits and commits each one (`/undo` reverts). For an agent that also runs shell commands on its own, opencode can point at the same `http://127.0.0.1:8180/v1` endpoint, at the cost of a ~10K-token system prompt.

## Assistant memory
Shared with Claude Code, using Claude's memory schema (one markdown file per topic):
```
---
name: video-server
description: one-line summary (shown in the index)
type: user | feedback | project | reference
triggers: video, videos, caddy     # words in a request that auto-load the topic
---
- fact
```
- **Claude's memory** (`~/.claude/projects/-home-dean/memory/`) is **read-only** to the assistant and wins on name clashes. Keep the `triggers:` lines when editing those files.
- **The assistant's own notes** go to `~/.local/share/voice-assistant/memory/` (`remember` / `rewrite_memory` tools). Every save, rewrite and delete needs a fuzzel Yes, like commands.
- `user` and `feedback` topics are in every session's system prompt; the others are listed in a one-line index and load when a trigger word (or the topic name) is in the request, or via `recall`. The system prompt stays fixed per session so llama-server reuses its cached prefix.
- **Curation:** ask Claude to "review Osprey's memory": it checks the assistant's notes against the system, moves good ones into its own memory and fixes or deletes the rest.
- `ai-toggle mem` lists both; `ai-toggle mem <topic>` edits one of the assistant's notes. Changes apply from the next session ("new session", 10 min idle, or a service restart).

## Memory safeguards
- Model weights are mmapped: under memory pressure the kernel drops the CPU-side experts (LLM gets slower) instead of OOMing.
- Caps: llama-server 16G, whisper-server 3G, voice-assistant 512M (`MemoryMax`).
- `OOMScoreAdjust=500` on all AI units, so the kernel kills them before games/Firefox; a SIGKILL doesn't restart them, and a "Local AI stopped" notification appears.
- earlyoom kills llama-server/whisper-server first when RAM < 5% and swap < 50%.

## To-do
- **Dictation into the focused window:** Whisper transcript → `wtype` (official repos) to type into whatever window has focus, for aider/VS Code/browser. Trigger undecided — not a keybind.
