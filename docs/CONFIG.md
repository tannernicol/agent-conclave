# Conclave Configuration

Config is layered:
1. `config/default.yaml` (shipped defaults)
2. `~/.config/conclave/config.yaml` (optional user overrides)
3. Environment variables (highest priority)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `CONCLAVE_HOST` | Server bind address |
| `CONCLAVE_PORT` | Server port |
| `CONCLAVE_DATA_DIR` | Base directory for runs and indexes |
| `CONCLAVE_RAG_URL` | RAG server base URL |
| `CONCLAVE_MCP_CONFIG` | Path to MCP config file |
| `CONCLAVE_STRICT` | Enable/disable strict quality mode |
| `CONCLAVE_CALIBRATION` | Enable/disable model calibration |
| `CONCLAVE_RUN_TIMEOUT` | Pipeline run timeout in seconds |
| `CONCLAVE_CLI_TIMEOUT` | CLI model call timeout in seconds |

## Key Sections

### Server
```yaml
server:
  host: 127.0.0.1
  port: 8099
```

### Data Directory
```yaml
data_dir: ~/.conclave
```

### Models
Each model is defined as a capability card:

```yaml
models:
  cards:
    - id: ollama:qwen2.5-coder:7b
      provider: ollama
      kind: local
      model_label: qwen2.5-coder:7b
      capabilities:
        text_reasoning: true
        code_generation: true
        code_review: low
        tool_use: false
        json_reliability: low
      constraints:
        max_context_tokens: 8192
        supports_system_prompt: true
      cost:
        usd_per_1m_input_tokens: 0
        usd_per_1m_output_tokens: 0
      perf_baseline:
        p50_latency_ms: 900
        tokens_per_sec: 40
```

CLI-based models use `command`, `prompt_mode`, and `timeout_seconds`:

```yaml
    - id: cli:claude
      provider: cli
      kind: external
      command: ["claude", "--print", "--model", "sonnet", "--output-format", "text"]
      prompt_mode: arg
      timeout_seconds: 300
```

### Planner
Controls how models are assigned to roles:

```yaml
planner:
  prefer_local: false
  prefer_best: true
  self_organize:
    enabled: true
    budget:
      enabled: true
      total_tokens: 120000
      cost_weight_boost: 0.35
  role_overrides:
    reasoner: cli:codex
    critic: cli:claude
  weights:
    latency: 0.35
    reliability: 0.25
    cost: 0.2
    affinity: 0.2
  role_affinity:
    reasoner: {speed: 0.3, reasoning: 1.0, json: 0.5}
    critic: {speed: 0.2, reasoning: 1.0, json: 0.8}
```

### RAG
```yaml
rag:
  base_url: http://localhost:8091
  use_server_collections: true
  skip_empty_collections: true
  max_results_per_collection: 8
  min_score: 0.2
  min_snippet_len: 40
```

#### Domain Allowlists
Restrict which RAG collections are used per domain:
```yaml
rag:
  enforce_allowlist: true
  domain_allowlist:
    code_review: [code-docs, api-docs]
    research: [papers, notes]
```

#### Dynamic Patterns
Auto-include collections by keyword:
```yaml
rag:
  dynamic_patterns:
    security:
      - "security"
      - "audit"
      - "vulnerability"
```

### Routing
Keyword-based domain detection:

```yaml
routing:
  domain_priority:
    - code_review
    - research
    - creative
    - general
  domain_keywords:
    code_review:
      - code review
      - architecture
      - debugging
      - refactor
    research:
      - research
      - analysis
      - compare
      - evaluate
    creative:
      - brainstorm
      - creative
      - story
      - design
```

### Routing Validation
Automated regression tests for routing:

```yaml
routing_validation:
  enabled: true
  cases:
    - query: "review this code for bugs"
      expect_domain: code_review
    - query: "brainstorm marketing ideas"
      expect_domain: creative
```

### Deliberation
Multi-round Reasoner/Critic loop:

```yaml
deliberation:
  max_rounds: 5
  min_intelligent_models: 2
  require_agreement: true
  stop_on_repeat_disagreements: true
  stability_rounds: 2
  max_draft_chars: 4000
  max_feedback_chars: 4000
  resolver:
    enabled: true
    max_disagreements: 6
  panel:
    enabled: true
    model_ids: [cli:codex, cli:claude, ollama:qwen2.5-coder:7b]
    require_all: false
    min_agree_ratio: 0.6
```

### Annealing
Simulated annealing optimization:

```yaml
annealing:
  enabled: true
  max_iterations: 4
  stable_rounds: 2
  schedule: linear          # linear or exp
  temperature_start: 1.2
  temperature_end: 0.3
  noise_start: 0.18
  noise_end: 0.05
  accept_worse_min_prob: 0.05
  shuffle_panel: true
  perturb_prompt: false
  content_convergence: true
  similarity_threshold: 0.85
```

### Diversity Check
Optional third-model diversity injection:

```yaml
diversity_check:
  enabled: true
  model_ids: [cli:gemini]
  trigger: disagreement_or_random
  max_calls: 1
  annealing:
    start_prob: 0.15
    end_prob: 0.02
```

### Quality
Evidence thresholds and domain risk:

```yaml
quality:
  strict: true
  min_evidence: 2
  min_strong_evidence: 0
  min_content_evidence: 1
  min_non_user_evidence: 1
  domain_risk_multipliers:
    security: 2.5
    research: 1.5
    general: 1.0
  domain_overrides:
    general:
      min_evidence: 1
      min_non_user_evidence: 0
      allow_user_only: true
    security:
      min_evidence: 3
      min_non_user_evidence: 2
      min_strong_evidence: 1
```

### Calibration
Model health pings:

```yaml
calibration:
  enabled: true
  providers: [ollama, cli]
  max_seconds: 20
  ping_prompt: "Return only: OK"
```

### Self-Report
Models describe their own capabilities:

```yaml
self_report:
  enabled: true
  providers: [ollama, cli]
  max_seconds: 30
  ttl_seconds: 86400
```

### Required Models
Cancel runs if mandatory models are unavailable:

```yaml
required_models:
  enabled: true
  models:
    - cli:codex
    - cli:claude
```

### Evaluation Suite
Compare Conclave vs baseline:

```yaml
evaluation:
  enabled: true
  baseline_model: cli:codex
  judge_model: cli:claude
  cases:
    - id: code_review_example
      query: "Review this API design for issues"
      output_type: report
```

### Topics (Scheduled Reconciliation)
```yaml
topics:
  - id: weekly-review
    query: "Weekly project status and priorities"
    schedule: weekly
    output_type: report
    output_md: ~/reports/weekly.md
```

### Token Budget (CLI)
Override planner token budget per run:

```bash
conclave run --query "..." --token-budget-total 50000
conclave run --query "..." --token-budget-remaining 25000
```

### Verification (On-Demand Sources)
Fetch additional evidence per domain:

```yaml
verification:
  search:
    base_url: http://127.0.0.1:8095
    endpoint: /search
    timeout: 10
  on_demand:
    research:
      - type: search
        title: Research papers
        collection: web-search
        query_suffix: "academic papers research"
```
