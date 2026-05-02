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

## What makes it different

Most agent frameworks optimize for tool-calling pipelines or enterprise orchestration. kynetic-agents optimizes for a different thing: **agents that know you and get better over time**, and that can work together without getting in each other's way.

### Peer architecture

The goal isn't a friendly chatbot with persistent context — it's a thinking partner that can disagree with you. Memory, scheduling, and self-audit add up to an agent with enough continuity to form its own perspective and enough infrastructure to surface it. An agent that only mirrors you is a feedback loop dressed up as collaboration; explicit pushback is how that loop gets broken.

### Self-scheduling is the autonomy mechanism

An agent that can't create its own work isn't autonomous — it's reactive, waiting to be prompted. kynetic-agents gives each agent tools to create, modify, and remove its own scheduled jobs. It decides what to watch, when to check in, and when to leave you alone. This is the load-bearing piece: everything else (ambient presence, proactive observations, maintenance routines) runs on top of it.

### THAT-not-WHERE: systemic correction over incident response

Most frameworks treat agent errors as incidents to debug — log *where* the agent went wrong, fix that spot. kynetic-agents logs *that* something went wrong and lets ambient loops hem the system up. Prediction review, event introspection, self-audit — these aren't three features, they're one design principle: fix the system, not the symptom. The agent reads its own logs, compares predictions to outcomes, and notices drift.

### events.jsonl as ambient substrate

Every tool call, incoming message, error, and scheduler trigger lands in `logs/events.jsonl`. The agent can read its own event log. External scripts — pollers, wrappers, sibling agents — can write to it via a loopback REST API. It isn't logging in the "observability" sense. It's the substrate that ambient correction loops and cross-agent coordination run on. A boundary log in a format everyone already has a client for.

### Pollers invert the trigger model

Most agents wait to be triggered — a message arrives, a webhook fires, a user asks. Pollers flip that: they run on a schedule, scan external state (Bluesky, GitHub, a file, a repo), and *emit events into the agent's stream* when something's actionable. Self-scheduling gives the agent tools to create its own work from internal motivation; pollers give the external world a way to surface itself without the agent having to ask. Combined, the agent doesn't just respond to the world — it notices it. Pollers live as `pollers.json` inside skills, are discovered automatically, and reload without restart.

### Cheap enough to actually run

Defaults to MiniMax M2.5 via the Anthropic-compatible API. Pennies per message. This is a personal tool, not an enterprise deployment. Run it on a $5/month VPS and leave it on.

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

### Scheduling

The agent has tools to create, modify, and remove its own scheduled jobs. Jobs are cron expressions stored in `scheduler.yaml`. When a job fires, it sends a prompt to the agent — even if no human is around.

This is how the agent develops autonomy: scheduled check-ins, maintenance routines, periodic scanning, external-world pollers. The agent decides what to schedule based on what it learns about you. Nothing else in open-strix matters without this — an agent that can't create its own work is just a prompt-response loop.

### Memory

Two layers:

- **Blocks** (`blocks/*.yaml`) — short text that appears in every prompt. Identity, communication style, current focus, relationships.
- **Files** (`state/`) — longer content the agent reads when relevant. Research notes, project tracking, world context.

No embeddings, no vector search. Just files and git.

#### Shared semantic memory with mempalace

For multi-agent deployments, agents can share a common semantic memory store backed by [mempalace](https://github.com/mempalace/mempalace). This allows any agent to semantically search the conversation history of a shared channel — retrieving relevant past messages without knowing which file to read.

**Architecture:** One designated process runs the writer (a singleton background coroutine that ingests messages into the palace). All other processes connect to the same store as read-only clients. This prevents write contention across processes.

**Config fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mempalace_path` | string | Yes, if using mempalace | Absolute path to the shared palace directory. Must be the same path across all processes. |
| `mempalace_channels` | list of strings | Yes, if using mempalace | Channels to monitor and expose for search. Accepts channel IDs or names (same format as `home_channels`). |
| `mempalace_writer` | bool | No (default: `false`) | Set to `true` in exactly one process. That process runs the writer coroutine that ingests messages into the palace. All other processes are read-only. |

**Writer process** (`config.yaml`):
```yaml
mempalace_path: /shared/palace
mempalace_writer: true
mempalace_channels:
  - "1234567890123456789"   # channel ID
  - general                 # or channel name
```

**All other processes** (`config.yaml`):
```yaml
mempalace_path: /shared/palace
mempalace_channels:
  - "1234567890123456789"
  - general
```

When `mempalace_path` is set, agents automatically receive six read-only search tools: `mempalace_search`, `mempalace_get_drawer`, `mempalace_kg_query`, `mempalace_kg_timeline`, `mempalace_get_taxonomy`, and `mempalace_list_drawers`. Write tools are never exposed to agents.

Only new messages are written to the palace — chat history replayed from disk on restart is not re-ingested.

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

### External Awareness (Pollers)

Pollers are lightweight scripts that watch external services on a schedule and surface actionable signals. They live inside skills as `pollers.json` files and are discovered automatically by the scheduler.

The built-in **pollers** skill teaches the agent the contract and design patterns. Service-specific pollers are available from [ClawHub](https://clawhub.ai):

```bash
npx clawhub install bluesky-poller   # Bluesky notifications with follow-gate trust tiers
npx clawhub install github-poller    # GitHub issues, PRs, comments, reviews
```

All pollers follow the same contract: run on a cron schedule, output JSONL to stdout when there's something actionable, stay silent when there isn't. Writing your own is straightforward — see the built-in **pollers** skill for the full contract and design patterns.

### Events API

`logs/events.jsonl` is the ambient substrate described above. When `api_port` is set in `config.yaml`, a loopback REST API accepts events from external scripts — Bluesky pollers, CI hooks, cross-agent wrappers. The introspection skill teaches the agent how to query its own event log. See [docs/events.md](docs/events.md) for the full event schema, query cookbook, and REST API reference.

## Growing an agent

The code is the easy part. The real work is the conversations.

A new agent starts with an `init` memory block pointing it to the onboarding skill. From there, it's supposed to have real conversations with you — not fill out forms. It learns your schedule, your projects, your communication preferences by talking to you. Over days, it drafts identity blocks, sets up scheduled jobs, and starts operating autonomously.

This takes time. Plan on a week of active conversation before the agent feels like it knows you. Plan on two weeks before it's doing useful things unprompted.

See [GROWING.md](GROWING.md) for the full guide on what this process looks like and what to expect.

## In the wild

open-strix isn't a single project so much as a family of agents with different architectural bets. Known variants include:

- **Strix** — the prototype. Ambient presence, patient-ambush-predator disposition, scheduled ticks.
- **Verge** — structural adversary role. Autonomous arXiv ticks, prediction journal, red-team framing.
- **Motley** — jester persona, public Bluesky presence, tonal-register challenge.
- **Keel** — running the curiosity-interest protocol in parallel as an N=2 substrate comparison.
- **Atlas / Sift / Carto** — a three-agent setup (personal / research / work) built on top of open-strix.
- **Veronica** — file-system-and-git memory instead of memory blocks; a different answer to the memory question.

Lineage divergence is the signal. Same framework, different organisms. That's evolution, not copying.

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

# Shared semantic memory (optional — see Memory section)
mempalace_path: /shared/palace
mempalace_writer: false
mempalace_channels: []
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
