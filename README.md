# Conclave -- Multi-Agent Consensus Engine

[![CI](https://github.com/tannernicol/agent-conclave/actions/workflows/ci.yml/badge.svg)](https://github.com/tannernicol/agent-conclave/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Orchestrate multiple LLMs into auditable, high-confidence decisions through structured deliberation.**

Conclave is a model orchestration engine that routes questions through a multi-round Reasoner/Critic deliberation loop, uses simulated annealing to optimize across iterations, and produces versioned consensus decisions backed by a full audit trail.

## At a Glance

- Multi-model consensus engine for technical and high-stakes decisions
- Structured reasoner/critic rounds with configurable convergence rules
- Audit-first design with replayable decision traces
- Local-first and provider-agnostic model routing

```
                          +------------------+
                          |   Query Input    |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |   Domain Router  |  keyword-based routing
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |              |              |
              +-----v----+  +-----v----+  +------v-----+
              | RAG/Index |  | On-Demand|  | User Input |
              | Retrieval |  | Sources  |  | & Artifacts|
              +-----+----+  +-----+----+  +------+-----+
                    |              |              |
                    +--------------+--------------+
                                   |
                          +--------v---------+
                          |  Quality Gate    |  evidence thresholds
                          +--------+---------+
                                   |
                   +---------------v---------------+
                   |      Simulated Annealing      |
                   |  +-------------------------+  |
                   |  |   Deliberation Loop     |  |
                   |  |                         |  |
                   |  |  Reasoner --> Critic    |  |
                   |  |     ^           |       |  |
                   |  |     +-----------+       |  |
                   |  |   (repeat until agree)  |  |
                   |  +-------------------------+  |
                   |  Re-route, perturb, rescore   |
                   +---------------+---------------+
                                   |
                          +--------v---------+
                          |   Panel Voting   |  weighted agreement
                          +--------+---------+
                                   |
                          +--------v---------+
                          |   Reconciliation |  compare to prior run
                          +--------+---------+
                                   |
                          +--------v---------+
                          |   Consensus      |  versioned decision
                          +------------------+
```

## What Makes Conclave Different

Most multi-model systems do a single pass: ask N models, pick the best answer. Conclave goes further:

- **Simulated Annealing Optimization** -- Runs multiple deliberation iterations with decreasing temperature, accepting worse solutions early (exploration) and converging on the best late (exploitation). Not just one-shot.

- **Self-Organizing Role Assignment** -- Models are assigned to roles (reasoner, critic, summarizer) based on capability scoring, latency, cost, and domain-specific calibration. Models can self-report their strengths.

- **Cross-Run Reconciliation** -- Each decision is compared against previous consensus on the same topic. Stability is tracked; changes are justified. Decisions improve over time.

- **Panel Voting with Weighted Agreement** -- Multiple models vote on each deliberation round. Agreement is measured with configurable thresholds, weighted by model reliability.

- **Multi-Round Reasoner/Critic Loop** -- Not just "generate and rank." The reasoner drafts, the critic challenges, the reasoner revises -- iterating until genuine agreement or max rounds.

- **Domain-Aware Quality Gates** -- Evidence requirements scale with domain risk. High-stakes domains demand more evidence. Insufficient evidence returns an honest "needs more data" instead of hallucinating.

- **Full Audit Trail** -- Every run produces `audit.jsonl` with step-by-step decisions: routing, role assignment, retrieval, each deliberation round, and final settlement.

## Quick Start

```bash
# Clone and install
git clone https://github.com/tannernicol/agent-conclave.git
cd agent-conclave
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure your models (edit config/default.yaml)
# At minimum, you need Ollama running locally:
# ollama pull qwen2.5-coder:7b

# Run the API + UI
python -m conclave.server
# Open http://localhost:8099

# Or use the CLI
conclave run --query "What is the best programming language for systems programming?"
```

## CLI Usage

```bash
# List available models
conclave models list

# Plan role assignments for a query
conclave plan --query "Compare Rust and Go for backend services"

# Run a full consensus pipeline
conclave run --query "Should we use microservices or a monolith?" --progress

# Write output to markdown
conclave run --query "Best practices for API design" --output-md /tmp/conclave.md

# View latest run
conclave runs latest

# Reconcile a scheduled topic
conclave reconcile --topic weekly-review

# Health check
conclave health

# Validate routing
conclave validate --routing

# Run evaluation suite
conclave eval --limit 5
```

## Example Output (Deliberation Trace)

```
[conclave] run_id=2026-02-05T10-30-00-abc123
 preflight start
 calibration ollama:qwen2.5-coder:7b ok
 requirements start
 requirements ok
 route done reasoner->codex critic->claude summarizer->claude
 retrieve done rag=12 evidence=8
 quality ok evidence_count=8 max_signal=0.92
 deliberate start
 deliberate round=1 agreement=false disagreements=3
 deliberate round=2 agreement=false disagreements=1
 deliberate round=3 agreement=true
 annealing iteration=1 score=0.85 accepted=true
 annealing iteration=2 score=0.91 accepted=true
 annealing done best_score=0.91
 settlement complete confidence=high
```

## Configuration

Config is layered: `config/default.yaml` (defaults) + `~/.config/conclave/config.yaml` (overrides).

Key sections:

| Section | Purpose |
|---------|---------|
| `models.cards` | Model capability cards (Ollama, CLI, API) |
| `planner` | Role assignment weights and preferences |
| `deliberation` | Multi-round loop settings, panel voting |
| `annealing` | Simulated annealing schedule and parameters |
| `quality` | Evidence thresholds and domain risk multipliers |
| `routing` | Domain keyword routing and validation |
| `rag` | RAG retrieval settings |
| `verification` | On-demand source fetching |

See [docs/CONFIG.md](docs/CONFIG.md) for full reference.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture guide.

## Project Structure

```
conclave/
  pipeline.py        # Core consensus engine
  cli.py             # CLI interface
  server.py          # FastAPI web UI
  config.py          # Configuration loader
  store.py           # Versioned decision storage
  audit.py           # Audit trail logging
  rag.py             # RAG retrieval client
  sources.py         # External source fetchers
  verification.py    # On-demand evidence fetching
  mcp.py             # MCP server discovery
  mcp_bridge.py      # MCP tool bridge
  scheduler.py       # Systemd timer generation
  quality_audit.py   # RAG/MCP health audits
  domains/           # Domain-specific prompt hints
  models/
    registry.py      # Model capability registry
    planner.py       # Role assignment + calibration
    ollama.py        # Ollama client
    cli.py           # CLI model client
    gemini.py        # Gemini API client
config/
  default.yaml       # Default configuration
prompts/
  roles/             # Role prompt templates
tests/               # Test suite
examples/            # Demo scripts
```

## Extending Conclave

### Add a New Domain

1. Add domain keywords to `routing.domain_keywords` in config
2. Add domain instructions to `conclave/domains/__init__.py`
3. Optionally add quality overrides in `quality.domain_overrides`

### Add a New Model

Add a card to `models.cards` in config:

```yaml
models:
  cards:
    - id: ollama:llama3:8b
      provider: ollama
      kind: local
      model_label: llama3:8b
      capabilities:
        text_reasoning: true
        code_generation: low
        json_reliability: medium
      perf_baseline:
        p50_latency_ms: 800
        tokens_per_sec: 45
```

### Custom Quality Gates

Override evidence thresholds per domain:

```yaml
quality:
  domain_overrides:
    my_domain:
      min_evidence: 5
      min_non_user_evidence: 3
      min_strong_evidence: 2
```

## Tests

```bash
python -m pytest tests/ -v
```

## Public Hygiene

Before publishing docs, logs, or examples:

```bash
python scripts/redact.py --self-check
```

Reference:

- [Security Policy](SECURITY.md)
- [Public Scope](docs/public-scope.md)
- [Redaction Policy](docs/redaction-policy.md)

## License

MIT
