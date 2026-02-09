# Conclave Architecture

Conclave is a multi-agent consensus engine that produces **versioned decisions** through **structured deliberation** and **simulated annealing optimization**.

## Design Goals

- Stable, inspectable role routing (no hardcoded providers)
- Local-first execution (no paid APIs by default)
- Multi-round Reasoner/Critic iterative agreement
- Simulated annealing to explore diverse solution paths
- Versioned decisions with cross-run reconciliation
- Full audit trail for every decision

## Core Components

| Component | Purpose |
|-----------|---------|
| **Model Registry** | Capability cards, telemetry, health status for each model |
| **Planner** | Deterministic role assignment by capability, latency, cost, and domain calibration |
| **Pipeline** | Orchestrates the full decision flow: route -> retrieve -> deliberate -> settle |
| **Store** | Versioned run history with `latest.json` pointer |
| **RAG Client** | Retrieves evidence from configured collections |
| **Quality Gate** | Enforces minimum evidence thresholds before producing answers |
| **Audit Log** | Step-by-step decision trace (`audit.jsonl` per run) |

## Data Flow

### 1. Preflight
- Ping models (20s max), update telemetry
- Optional: model self-report capabilities

### 2. Route
- Detect domain from query keywords (configurable keyword map)
- Select RAG collections based on domain
- Assign models to roles (router, reasoner, critic, summarizer) via the Planner
- Compose the panel of voting models

### 3. Retrieve
- Query RAG collections for evidence
- Fetch on-demand sources (HTTP, MCP, search)
- Load user input files if provided
- Score and filter evidence by signal strength

### 4. Quality Check
- Count evidence items, check domain-specific thresholds
- If insufficient: return honest "needs more data" response
- If sufficient: proceed to deliberation

### 5. Deliberate (with Simulated Annealing)

The annealing loop runs multiple deliberation iterations:

```
for iteration in 1..max_iterations:
    temperature = anneal(T_start, T_end, iteration)
    noise = anneal(noise_start, noise_end, iteration)

    # Re-route with noise (explore different model assignments)
    candidate_route = route_query(noise=noise)

    # Run deliberation loop
    for round in 1..max_rounds:
        reasoner_draft = reasoner.generate(query, context, feedback)
        critic_feedback = critic.review(draft)
        if critic agrees: break

    # Panel voting (all configured models vote)
    panel_votes = [model.review(draft) for model in panel]
    agreement = weighted_agreement(panel_votes)

    # Annealing acceptance
    score = deliberation_quality_score(agreement, disagreements, convergence)
    if score > current or random() < exp(delta / temperature):
        accept(candidate)
```

Each deliberation round within an iteration:
- **Reasoner** drafts or revises a response
- **Critic** challenges with specific disagreements
- **Reasoner** addresses each disagreement in next round
- **Panel** votes on the current draft
- Loop continues until agreement or max rounds

### 6. Reconcile
- Compare current answer to previous consensus on the same topic
- Calculate text similarity (Jaccard on word trigrams)
- Track whether the decision changed and why

### 7. Settle
- Produce final consensus with confidence level
- Write run JSON, audit log, and optional markdown report
- Update latest pointer

## Storage Layout

```
~/.conclave/
  models/
    registry.json         # Model capability cards
    benchmarks.jsonl      # Performance telemetry
    health.json           # Current model health
    model_calibration.json # Domain-specific calibration scores
  runs/
    <run_id>/
      run.json            # Full run data + consensus
      audit.jsonl         # Step-by-step audit trail
  latest.json             # Pointer to most recent run
  evals/                  # Evaluation results
  audits/                 # RAG/MCP health audits
```

## Model Calibration

Models can be calibrated per-domain. The calibration system:

1. Asks each model to self-rate expertise across domains (1-10)
2. Caches scores with a 7-day TTL
3. Uses scores to bias role assignment toward domain experts
4. Updates scores based on deliberation outcomes (winner gets +0.1, others -0.05)

Calibration domains: `general`, `code_review`, `security`, `creative`, `research`

## Quality Scoring

Deliberation quality is scored 0-1 based on:
- Agreement status (40% weight)
- Weighted panel agreement ratio (25%)
- Disagreement count (20%)
- Convergence speed (10%)
- Stop reason penalty (5%)

This score drives annealing acceptance decisions.

## Diversity Checks

An optional third model can be injected as a "diversity critic" with annealed probability:
- Higher chance early in deliberation (exploration)
- Lower chance later (convergence)
- Boosted on disagreement rounds

## Scheduling

Topic schedules can be rendered into systemd timers:

```bash
conclave schedule apply --enable
```

Each topic gets a service + timer pair for automated periodic re-evaluation.

## Security and Privacy

- Local-first inference; no paid API tokens required by default
- CLI models use local login sessions (no API keys)
- Explicit exclude patterns prevent indexing secrets and media
- All data stays on your machine
