# TwinMind Local

A local, private clone of [TwinMind](https://twinmind.com) for Apple Silicon Macs. It listens to your calls and meetings, transcribes them on-device, suggests what **you** would say while the call is happening, and after the call files everything into a markdown "second brain" that it uses next time.

## What it does (mapped to TwinMind's features)

| TwinMind feature | This clone |
|---|---|
| Capture all day, private by design | Live capture of your mic (**Me**) and call audio (**Them**) as separate channels. Whisper runs on the Mac's GPU via MLX. Audio is thrown away after transcription unless you set `keep_audio = true`. |
| 140+ languages, real-time translation | Whisper auto-detects about 99 languages per chunk. Set `translate = true` to get English output. |
| Proactive help during a conversation | **Your twin** panel. Every 20 s, and right away when the other side asks a question, it suggests what to say next, questions to ask, relevant facts from memory, and warnings (commitments, contradictions). Press **R** for "What would I say?" or type the question you were asked. |
| Ask any AI about your life | **Ask** tab: chat over all your memories and transcripts, with sources cited. |
| Organize into a second brain / private Wikipedia | After each call it extracts a summary, decisions, action items, people, projects, and things you said about yourself, then writes them into `data/memory/*.md`. |
| Daily digest, action items | **Actions** tab: open items plus a generated briefing for the day. |
| Upload & transcribe audio | **Sessions → Upload audio** (m4a/mp3/wav/mp4). |
| On-device mode | Set `provider = "ollama"` and nothing leaves the Mac. |

Not cloned: phone and Apple Watch capture, Gmail/Calendar integrations, cloud sync, speaker identification within "Them" (every remote speaker is labelled "Them").

## The memory layer (the "think like me" part)

```
data/memory/
  profile.md          <- YOU write this. Who you are, how you think, how you talk, your positions,
                         and what never to say. The twin treats it as authoritative. Never auto-edited.
  learned.md          <- auto-appended after each call, only from lines *you* spoke. Review it weekly
                         and move anything durable into profile.md.
  people/<name>.md    <- one page per person: dated timeline of what they said or want
  projects/<name>.md  <- one page per project
data/sessions/*.md    <- every call: summary, decisions, action items, full transcript
data/index.db         <- SQLite FTS5 search index + action items
```

Every twin call gets: `profile.md`, the last 40 lines of `learned.md`, and the top search hits for the current conversation. All of it is plain markdown, so you can also edit it in Obsidian or VS Code (use **Memory → Save** or restart the app so the search index picks up the changes).

**How good the twin is depends almost entirely on how specific `profile.md` is.** A generic profile gives you generic suggestions.

## Setup on a Mac (M-series)

```bash
git clone <this repo> && cd twinmind
brew install ffmpeg            # needed only for "Upload audio"
./twin                         # first run creates .venv, installs mlx-whisper, opens the UI
```

Then pick an LLM:

- **Claude (best reasoning):** `export ANTHROPIC_API_KEY=...` (or `ant auth login`) before `./twin`. The default model is `claude-opus-5`. Transcripts are sent to the Anthropic API.
- **Fully local:** `brew install ollama && ollama pull gemma4:26b`, then set `provider = "ollama"` in `config.toml`. After the download, nothing leaves the Mac. Its reasoning is noticeably weaker than Claude's.

### Choosing a local model (by unified memory)

For live suggestions, the model spends most of its time reading the 4–6K-token prompt (profile + memories + transcript) before it writes anything. Mixture-of-experts models (few active parameters) read the prompt several times faster than dense models, so prefer them.

| Mac memory | `ollama_model` | Notes |
|---|---|---|
| 24 GB | `gemma4:12b` (7.6 GB) | same model for live and summaries |
| 32–36 GB | **`gemma4:26b`** (18 GB, ~4B active) | default. Leaves room for Whisper, Chrome and the KV cache |
| 48 GB+ | `qwen3.5:35b-a3b` (24 GB) | A/B test it against gemma4:26b on your own calls |

Avoid dense 27–31B models for live use: the prompt alone takes 10–25 s to read. Models are stored in `~/.ollama/models` (set `OLLAMA_MODELS` to use an external SSD); Whisper is cached in `~/.cache/huggingface`.

The first run downloads Whisper `large-v3-turbo` (about 1.6 GB). If your machine has little memory, set `model = "small"`.

### Capturing the other side of a call

The browser can only hear your mic. To capture what the other side says:

1. **Browser calls (Meet, Teams web, Zoom web):** choose **Share tab/system audio**, pick the meeting tab, and tick *Share tab audio*.
2. **Desktop apps (Zoom, Teams, Slack, FaceTime):** install [BlackHole 2ch](https://existential.audio/blackhole/) (`brew install blackhole-2ch`). In *Audio MIDI Setup*, create a **Multi-Output Device** with your headphones and BlackHole, and set it as the system output. In TwinMind, choose **Device: BlackHole 2ch ← call audio**.

**Wear headphones.** On speakers, your mic also picks up the other side, and those lines get attributed to you. That pollutes `learned.md`.

### Make it feel like an app

In Chrome, open `http://127.0.0.1:8765`, then go to **⋮ → Cast, save and share → Install page as app**. It gets its own Dock icon and window.

## Configuration

Copy `config.example.toml` to `config.toml` (the launcher does this for you). Useful settings:

- `user.name`: the name used in prompts and the transcript.
- `live.suggest_interval_sec`: lower values mean more suggestions and more API spend.
- `llm.live_model`: set this to a faster model if in-call latency feels slow.
- `live.silence_rms`: raise it if a noisy room produces junk transcript lines.

Any key can also be overridden with an environment variable, for example `TWIN_LLM_PROVIDER=ollama` or `TWIN_STT_MODEL=small`.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q      # uses fake STT/LLM; no model download or API key needed
```

Code layout: `app/stt.py` (chunking and Whisper backends), `app/llm.py` (Claude and Ollama), `app/memory.py` (markdown memory and search), `app/twin.py` (prompts), `app/main.py` (FastAPI and WebSocket), `static/` (UI).

## Privacy and safety notes

- Tell people when you record them. Recording without consent is illegal in some places (for example two-party-consent US states and much of the EU).
- With `provider = "anthropic"`, transcripts and memory snippets are sent to the Claude API. Use Ollama for conversations that must stay confidential.
- The server listens on `127.0.0.1` only. Don't change `host` to `0.0.0.0`: there is no authentication.
