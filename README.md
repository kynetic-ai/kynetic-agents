# kynetic-agents

> **Forked from [open-strix](https://github.com/open-strix/open-strix).** This project extends open-strix with first-class support for multi-agent interaction via Discord using a **hub-and-spoke topology**: a central hub agent coordinates a fleet of spoke agents, each living in its own Discord channel, communicating through mentions and configurable `home_channels` routing.

[![PyPI version](https://img.shields.io/pypi/v/kynetic-agents.svg)](https://pypi.org/project/kynetic-agents/)

A persistent AI companion framework that lives in Discord, remembers everything, and gets better over time. Designed for running multiple agents together.

```bash
uvx kynetic-agents setup --home my-agent --github
cd my-agent
uv run kynetic-agents
```

Two commands. You have an agent. Connect it to Discord and start talking.

## What is this?

kynetic-agents is an opinionated framework for building long-running AI agents. Not chatbots — *companions*. Agents that develop personality through conversation, maintain memory across sessions, schedule their own work, and learn from their mistakes.

It runs on cheap models (MiniMax M2.5, ~$0.01/message), talks to you over Discord, and stores everything in git. No vector databases, no cloud services, no enterprise pricing. Just files, memory blocks, and a git history you can actually read.

**Hub-and-spoke multi-agent topology:** Each agent owns one or more Discord channels (`home_channels`). A hub agent coordinates the fleet — spokes communicate back to the hub by @mentioning it in their own channels. The hub has read access to spoke channels and picks up those messages automatically.

**How you interact with it:** You talk to agents on Discord. They talk back using tools (`send_message`, `react`). They have scheduled jobs that fire even when you're not around. Over time, they develop interests, track your projects, and start doing useful things without being asked.

## Why this exists

Most agent frameworks optimize for tool-calling pipelines or enterprise orchestration. kynetic-agents optimizes for a different thing: **agents that know you and get better over time**, and that can work together without getting in each other's way.

Three design bets:

- **Focused.** Small core, everything else is skills — markdown files the agent reads and follows. Add capabilities by dropping a file in `skills/`, or let the agent discover and install them at runtime.
- **Cheap.** Defaults to MiniMax M2.5 via the Anthropic-compatible API. Pennies per message. Run it on a $5/month VPS.
- **Stable.** Ships with built-in skills for self-diagnosis — prediction calibration loops, event introspection, onboarding that fades into regular operation. The agent can read its own logs, check whether its predictions were right, and notice when it's drifting.

## How it works

### The home repo

When you run `uvx kynetic-agents setup`, it creates a directory — the agent's *home*. Everything the agent knows lives here:

```
blocks/          # YAML memory blocks — identity, goals, patterns. In every prompt.
state/           # Markdown files — projects, notes, research. Read on demand.
skills/          # Markdown skill files. Drop one in, agent picks it up.
logs/
  events.jsonl   # Every tool call, error, and event. The agent can read this.
  chat-history.jsonl # Append-only chat transcript.
  journal.jsonl  # Agent's own log — what happened, what it predicted.
scheduler.yaml   # Cron jobs the agent manages itself.
config.yaml      # Model, Discord config, prompt tuning.
```

Everything except logs is committed to git after every turn. The git history *is* the audit trail.

### Multi-agent routing with `home_channels`

Each agent can be configured with a list of channels it should respond in:

```yaml
home_channels:
  - my-agent-channel
  - 1234567890123456789   # channel IDs also work
```

When `home_channels` is set, the agent only processes messages in those channels — unless it is directly @mentioned or an `always_respond_bot_ids` bot explicitly addresses it by name or ID. When `home_channels` is empty, the agent responds in all channels (original behavior).

For hub-and-spoke deployments:
- Give each spoke agent its own `home_channels` entry
- The hub agent reads all spoke channels (via Discord permissions) and picks up spoke→hub messages via `@mention`
- Spokes communicate to the hub through their `collaboration.md` onboarding docs

### Memory

Two layers:

- **Blocks** (`blocks/*.yaml`) — short text that appears in every prompt. Identity, communication style, current focus, relationships.
- **Files** (`state/`) — longer content the agent reads when relevant. Research notes, project tracking, world context.

No embeddings, no vector search. Just files and git.

### Skills

A skill is a markdown file in `skills/` with a YAML header. The agent sees all skills in its prompt and invokes them by name.

```yaml
---
name: my-skill
description: What this skill does and when to use it.
---
# Instructions for the agent
...
```

Built-in skills:

| Skill | Purpose |
|-------|---------|
| **onboarding** | Walks the agent through establishing identity, goals, and schedules |
| **memory** | How to maintain and organize memory blocks and state files |
| **skill-creator** | Create new skills from repeated workflows |
| **prediction-review** | Calibration loops — revisit past predictions against ground truth |
| **introspection** | Self-diagnosis from event logs and behavioral patterns |
| **five-whys** | Root-cause analysis when predictions or behaviors go wrong |

### Scheduling

The agent has tools to create, modify, and remove its own scheduled jobs. Jobs are cron expressions stored in `scheduler.yaml`. When a job fires, it sends a prompt to the agent — even if no human is around.

### Events API

Every tool call, incoming message, error, and scheduler trigger is logged to `logs/events.jsonl`. The agent can read its own event log — and the introspection skill teaches it how.

When `api_port` is set in `config.yaml`, a loopback REST API accepts events from external scripts.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and a Discord bot token.

```bash
uvx kynetic-agents setup --home my-agent --github
cd my-agent
# Edit .env with your API key and DISCORD_TOKEN
uv run kynetic-agents
```

The setup command handles everything: directory structure, git init, GitHub repo creation (with `--github`), service files for your OS, and a walkthrough for model/Discord configuration.

See [SETUP.md](SETUP.md) for detailed instructions on environment variables, model configuration, Discord setup, and deployment options.

## Upgrading

```bash
uv add -U kynetic-agents
```

## Configuration

`config.yaml`:

```yaml
model: MiniMax-M2.5
model_max_retries: 6
journal_entries_in_prompt: 90
discord_messages_in_prompt: 10
discord_token_env: DISCORD_TOKEN
always_respond_bot_ids: []
home_channels: []
api_port: 0
```

Models use the Anthropic-compatible API format. MiniMax M2.5 and Kimi K2.5 both work out of the box. Any model with an Anthropic-compatible endpoint will work — set `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` in `.env`.

## Tests

```bash
uv run pytest -q
```

## Safety

Agent file writes are limited to `state/` and `skills/`. Reads use repository scope. Built-in skills are read-only.

There is no sandboxing. Agents have full shell access. This is intentionally simple and should not be treated as a security boundary.

## License

MIT. See `LICENSE`.
