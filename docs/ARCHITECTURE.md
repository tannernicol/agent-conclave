# Conclave Architecture

Conclave is a V2 orchestration engine that produces a **versioned consensus decision** ("pope") using **local context** and a **Claude↔Codex feedback loop**.

## Goals
- Stable, inspectable role routing (no hardcoded providers)
- Local-first execution (no paid APIs by default)
- Claude↔Codex iterative agreement before final answer
- Versioned decisions with a **latest consensus pointer**
- Clear UI signals for deliberation and consensus

## Core components
- **Model Registry**: capability cards + telemetry + health (includes CLI-based Claude/Codex)
- **Planner**: deterministic role assignment by capability, latency, and reliability
- **Pipeline**: router → reasoner ↔ critic (multi-round) → summarizer
- **Store**: versioned run history + `latest.json`
- **Context**: homelab-search RAG + local NAS index

## Data flow
1. **Preflight**: ping models (20s max), update telemetry
2. **Route**: detect domain (tax/health/bounty/general)
3. **Retrieve**: query RAG collections + NAS index
4. **Deliberate**: Codex drafts, Claude critiques, Codex revises (repeat until agreement or max rounds)
5. **Summarize**: final decision + confidence + pope title
6. **Persist**: run JSON + update latest pointer

## Storage layout
Base directory: `/home/tanner/.conclave`

```
conclave/
├── models/
│   ├── registry.json
│   ├── benchmarks.jsonl
│   └── health.json
├── runs/
│   └── <run_id>/run.json
│   └── <run_id>/audit.jsonl
├── latest.json
└── index/
    └── nas_index.db
```

## Scheduling
- Weekly reconcile via systemd timer (`conclave-reconcile.timer`)
- Manual re-run via CLI or UI always surfaces the latest pope
- Topic schedules can be rendered into systemd timers via `conclave schedule apply`
- When using per-topic timers, disable the legacy `conclave-reconcile.timer` to avoid duplicate runs

## Context sources
- **homelab-search** (rag.tannner.com) for RAG collections
- **NAS index** from an allowlist (with explicit exclusions)
- **MCP registry** detected from `~/.mcp.json` (logged; no tool calls by default)
- **User inputs** stored as Markdown in `~/.conclave/inputs` and treated as high-signal evidence
- **Health dashboard** pages (health.tannner.com) and **Money API** (money.tannner.com) can be ingested as evidence

### RAG discovery
- Pulls collection catalog from rag.tannner.com
- Filters empty collections
- Adds domain-specific collections by keyword patterns

## UI states
- **Status pill** shows run state
- **White smoke** when consensus is reached
- **Latest pope** always shown in the main panel

## Audit trail
- Each pipeline run writes `audit.jsonl` with step-by-step decisions:
  - role assignment + rationale
  - routing + collections
  - retrieval samples
  - model invocations (role, model id, latency, ok/error)
  - deliberation rounds + agreement
  - settlement and reconciliation
  - quality checks and evidence statistics

## Security & privacy
- Local-first inference, no paid API tokens by default
- Claude/Codex via CLI login (no API key required)
- Explicit exclude patterns avoid secrets and media
