# Conclave Architecture

Conclave is a V2 orchestration engine that produces a **versioned consensus decision** ("pope") using **local Ollama models** and **homelab context**.

## Goals
- Stable, inspectable role routing (no hardcoded providers)
- Local-first execution (no paid APIs by default)
- Versioned decisions with a **latest consensus pointer**
- Clear UI signals for deliberation and consensus

## Core components
- **Model Registry**: capability cards + telemetry + health
- **Planner**: deterministic role assignment by capability, latency, and reliability
- **Pipeline**: router → reasoner → critic → summarizer
- **Store**: versioned run history + `latest.json`
- **Context**: homelab-search RAG + local NAS index

## Data flow
1. **Preflight**: ping models (20s max), update telemetry
2. **Route**: detect domain (tax/health/bounty/general)
3. **Retrieve**: query RAG collections + NAS index
4. **Deliberate**: reasoner proposes, critic challenges
5. **Summarize**: final decision + confidence + pope title
6. **Persist**: run JSON + update latest pointer

## Storage layout
Base directory: `/mnt/nas/Homelab/BugBounty/conclave`

```
conclave/
├── models/
│   ├── registry.json
│   ├── benchmarks.jsonl
│   └── health.json
├── runs/
│   └── <run_id>/run.json
├── latest.json
└── index/
    └── nas_index.db
```

## Scheduling
- Weekly reconcile via systemd timer (`conclave-reconcile.timer`)
- Manual re-run via CLI or UI always surfaces the latest pope

## Context sources
- **homelab-search** (rag.tannner.com) for RAG collections
- **NAS index** from an allowlist (with explicit exclusions)

## UI states
- **Locked doors** during deliberation
- **White smoke** when consensus is reached
- **Latest pope** always shown in the main panel

## Security & privacy
- No paid API tokens by default
- Explicit exclude patterns avoid secrets and media
- Local-only inference (Ollama)
