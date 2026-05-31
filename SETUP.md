# Setup Guide

Detailed setup instructions for kynetic-agents. For an overview of what kynetic-agents is, see [README.md](README.md).

## Prerequisites

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Official install docs (Homebrew, pipx, winget, etc.): https://docs.astral.sh/uv/getting-started/installation/

### Install and auth `gh` (optional)

The `gh` CLI automates GitHub repo creation during setup. If `gh` isn't available, `--github` still works — it prints instructions for creating the repo manually on GitHub and prompts you for the remote URL.

```bash
# macOS
brew install gh

# Ubuntu / Debian
sudo apt install gh
```

```powershell
# Windows
winget install --id GitHub.cli
```

Then authenticate:

```bash
gh auth login
gh auth status
```

Official docs: https://cli.github.com/

### Install attachment-handling tools (optional)

The built-in `view-attachment` skill inspects image, PDF, and document attachments by shelling out to system CLI tools. These are **not** bundled with the `kynetic-agents` package — they come from the host. Without them, the skill still runs but its image options silently no-op (it falls back to file metadata and asking the human).

| Tool | Provides | Used for |
|---|---|---|
| `tesseract` | OCR | extracting text from screenshots/photos |
| `convert` / `identify` (ImageMagick) | image metadata, conversion, pixel sampling | dimensions, format, simple-graphic tracing |
| `potrace` | bitmap → SVG tracing | reasoning about charts/line art |
| `pdftotext` (poppler-utils) | PDF text extraction | reading PDF attachments |

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr imagemagick potrace poppler-utils
```

```bash
# macOS
brew install tesseract imagemagick potrace poppler
```

Vision note: attachments are always passed to the model as **file paths**, never inlined as image content — so even vision-capable models rely on this skill (and these tools) to inspect attachments. There is no config knob to enable inline/native image input. See `kynetic_agents/discord.py` (`_save_attachments`) for the rationale.

## Quick start

```bash
uvx kynetic-agents setup --home my-agent --github
cd my-agent
uv run kynetic-agents
```

`kynetic-agents setup` bootstraps the target directory with:

- `state/`, `skills/`, `blocks/` — agent workspace directories
- `logs/events.jsonl`, `logs/chat-history.jsonl`, `logs/journal.jsonl` — event, chat transcript, and journal logs
- `scheduler.yaml` — scheduled job definitions
- `config.yaml` — model and runtime configuration
- `checkpoint.md` — post-journal reflection prompt
- `.env` — template for secrets
- `pyproject.toml`, `uv.lock` — Python dependencies

It also:
- Runs `uv init` and `uv add kynetic-agents`
- Checks git identity (prompts for `user.name` and `user.email` if missing)
- Checks git remote (prompts for remote URL if `origin` is missing)
- Detects OS and generates service files in `services/`:
  - Linux: `kynetic-agents.service` (systemd user unit)
  - macOS: `ai.kynetic-agents.<name>.plist` (launchd agent)
  - Windows: Task Scheduler install/uninstall PowerShell scripts
- Prints a CLI walkthrough with links for model and Discord setup

### Installed mode (alternative)

If you prefer a local project install instead of `uvx`:

```bash
uv init --python 3.11
uv add kynetic-agents
uv run kynetic-agents setup --home .
uv run kynetic-agents
```

## GitHub repo setup

kynetic-agents auto-syncs with git after each turn, so set up a repo + remote early. Keep it **private** — agent memory and logs can contain sensitive context.

**Recommended:**

```bash
uvx kynetic-agents setup --home my-agent --github
```

**Manual fallback (GitHub CLI):**

```bash
cd my-agent
gh repo create <repo-name> --private --source=. --remote=origin
git add .
git commit -m "Initial commit"
git push -u origin HEAD
```

**Manual fallback (GitHub website):**

1. Create a new **private** empty repo on GitHub (no README, no `.gitignore`, no license).
2. In your project directory:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:<your-user>/<repo-name>.git
git push -u origin main
```

HTTPS alternative: `git remote add origin https://github.com/<your-user>/<repo-name>.git`

## Environment variables

Start from the template:

```bash
cp .env.example .env
```

**Required:**

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | API key for the model provider |
| `ANTHROPIC_BASE_URL` | Endpoint URL (e.g., `https://api.minimax.io/anthropic`) |
| `DISCORD_TOKEN` | Discord bot token |

**Optional:**

| Variable | Purpose |
|---|---|
| `DISCORD_TEST_CHANNEL_ID` | Enables live send-message tests |
| `OPEN_STRIX_TEST_MODEL` | Model override for tests |

## Model configuration

### Default: MiniMax M2.5

```yaml
# config.yaml
model: MiniMax-M2.5
model_max_retries: 6
```

```bash
# .env
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
ANTHROPIC_API_KEY=your-key-here
```

MiniMax docs:
- Anthropic compatibility + model IDs: https://platform.minimax.io/docs/api-reference/text-anthropic-api
- AI coding tools guide: https://platform.minimax.io/docs/guides/text-ai-coding-tools

### Alternative: Kimi K2.5

```bash
# .env
ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic
```

Set `model` in `config.yaml` to the current Kimi model ID.

Moonshot docs:
- Overview: https://platform.moonshot.ai/docs/overview
- K2 update: https://platform.moonshot.ai/blog/posts/Kimi_API_Newsletter

### Model config behavior

- If `model` has no `:` (e.g., `MiniMax-M2.5`), kynetic-agents treats it as Anthropic-provider: `anthropic:MiniMax-M2.5`
- If `model` includes `provider:model` (e.g., `openai:gpt-4o-mini`), it passes through unchanged

Any model with an Anthropic-compatible API works. Just set `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`.

## Interfaces

kynetic-agents reaches you through three interfaces:

| | Discord | stdin | REST API |
|---|---|---|---|
| **Setup** | Bot token + server + permissions | None | Set `api_port` in `config.yaml` |
| **Best for** | Notifications, mobile access, multi-channel, scheduled job alerts | Quick local testing | External scripts, pollers, cross-agent wrappers |
| **How** | See [Discord setup](#discord-setup) | Runs automatically when no `DISCORD_TOKEN` is configured | Loopback HTTP on `127.0.0.1:<api_port>` (see [docs/events.md](docs/events.md)) |

Discord is the primary interface. If no Discord token is set, the agent falls back to **stdin mode** — type prompts directly in the terminal — which is handy for a first run. The REST API is a loopback-only event intake for automation, not a chat UI.

## Discord setup

The recommended interface for day-to-day use.

Use Discord's [Developer Portal](https://discord.com/developers/applications):

1. **General Information:** Set app/bot name and basic metadata.
2. **Installation:** Set `Install Link` to `None`, then save.
3. **OAuth2 → URL Generator:**
   - Check `bot`
   - Select permissions: `View Channels`, `Send Messages`, `Send Messages in Threads`, `Read Message History`, `Add Reactions`, `Attach Files`
4. **Bot tab:**
   - Disable `Public Bot`
   - Enable `Message Content Intent`
5. **Bot tab → Reset Token:**
   - Copy token immediately (it won't be shown again)
   - Set in `.env`: `DISCORD_TOKEN=<your_discord_bot_token>`
6. Use the generated OAuth2 bot invite URL to add the bot to your server.

Reference docs:
- [Getting started](https://docs.discord.com/developers/quick-start/getting-started)
- [OAuth2](https://docs.discord.com/developers/topics/oauth2)
- [Permissions](https://docs.discord.com/developers/topics/permissions)
- [Gateway + intents](https://docs.discord.com/developers/events/gateway)

Where this is configured in kynetic-agents:
- Token env var name: `config.yaml` → `discord_token_env` (default `DISCORD_TOKEN`)
- Bot allowlist: `config.yaml` → `always_respond_bot_ids`

## `config.yaml` reference

```yaml
model: MiniMax-M2.5
model_max_retries: 6
journal_entries_in_prompt: 90
discord_messages_in_prompt: 10
discord_token_env: DISCORD_TOKEN
always_respond_bot_ids: []
api_port: 0
folders:
  state: rw
  skills: rw
  blocks: ro
  scripts: ro
  logs: ro
```

| Key | Purpose |
|---|---|
| `model` | Model name or `provider:model` |
| `model_max_retries` | Provider retry attempts for transient model/API failures |
| `journal_entries_in_prompt` | Journal entries included in each prompt |
| `discord_messages_in_prompt` | Recent Discord messages in each prompt |
| `discord_token_env` | Env var name for Discord token |
| `always_respond_bot_ids` | Bot author IDs the agent responds to |
| `api_port` | Loopback REST API port (`0` disables it) |
| `folders` | Map of folder names to access mode (`rw` or `ro`) |
| `mcp_servers` | List of MCP server configs (see below) |

### Folders

The `folders` key controls which directories the agent can see and whether it can write to them. Each entry maps a folder name to an access mode:

- `rw` — read-write (agent can read and modify files)
- `ro` — read-only (agent can read but not modify files)

Folders are created automatically on startup. Add custom folders to give your agent access to additional directories:

```yaml
folders:
  state: rw
  skills: rw
  scripts: ro
  logs: ro
  research: ro       # custom read-only folder
  data: rw           # custom read-write folder
```

#### External directories

Folder paths can be relative to the agent's home directory. Use `../` to give an agent read-only access to a sibling directory:

```yaml
folders:
  state: rw
  skills: rw
  scripts: ro
  logs: ro
  "../cybernetics-research": ro   # sibling directory, read-only
```

If the agent lives at `~/jester/`, this resolves to `~/cybernetics-research/`. The agent can read files in that directory but can't modify them.

This is useful for giving an agent access to shared resources — research repos, documentation, datasets — without copying them into the agent's home directory. The directory is created on startup if it doesn't exist.

### MCP Servers

Add [MCP](https://modelcontextprotocol.io/) servers to give your agent access to external tools. Servers run as subprocesses and their tools appear alongside built-in tools.

```yaml
mcp_servers:
  - name: brave-search
    command: npx
    args: ["-y", "@anthropic/mcp-server-brave-search"]
    env:
      BRAVE_API_KEY: "${BRAVE_API_KEY}"
  - name: github
    command: npx
    args: ["-y", "@anthropic/mcp-server-github"]
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
```

Each server entry requires:
- `name` — unique identifier (used to namespace tools as `mcp_<name>_<tool>`)
- `command` — executable to run (e.g., `npx`, `python`, `node`)
- `args` — command arguments
- `env` — (optional) environment variables; `${VAR}` references are expanded from the process environment

Servers start on app launch. If a server fails to start, it's skipped — other servers and the rest of the app continue normally. Works with any model (MiniMax, Kimi, Claude, etc.).

Related files:
- `scheduler.yaml` — cron/time-of-day jobs
- `blocks/*.yaml` — memory blocks in prompt context
- `checkpoint.md` — post-journal reflection prompt
- `skills/` — user-editable local skills

Runtime behavior:
- Git sync (`git add -A` → commit → push) runs automatically after each processed turn.
- New agent homes include a twice-daily prediction-review job (09:00 and 21:00 UTC).

## Tests

```bash
uv run pytest -q
```

Discord test coverage:
- Unit tests with mocked boundaries: `tests/test_discord.py`
- Live integration tests: `tests/test_discord_live.py`

Live test env vars:
- `DISCORD_TOKEN` (required for live connect test)
- `DISCORD_TEST_CHANNEL_ID` (optional; enables live send-message test)
