# Conclave Configuration

Config is layered:
1) `config/default.yaml`
2) `~/.config/conclave/config.yaml` (optional override)

Key sections:
- `server`: host/port
- `data_dir`: base directory for runs and indexes
- `models`: capability cards and defaults
- `planner`: role weights + preferences
- `rag`: homelab-search endpoint + dynamic collection discovery
- `index`: NAS allowlist + exclude patterns + auto_build/refresh
- `quality`: evidence gating + signal thresholds for high-fidelity answers
- `mcp`: detected from `~/.mcp.json` and logged (no tool calls by default)
- `topics`: scheduled re-run topics

### RAG options
- `use_server_collections`: discover collections from rag.tannner.com
- `skip_empty_collections`: avoid empty collections (file_count == 0)
- `max_results_per_collection`: cap results per collection
- `prefer_non_pdf`: deprioritize PDF-heavy results
- `dynamic_patterns`: include extra collections by keyword
- `trust_explicit_collections`: when user passes collections, skip catalog lookup

### Topics scheduling
Each topic can include:
- `schedule`: systemd `OnCalendar` string (e.g. `weekly`, `daily`, `Mon *-*-* 03:00:00`)
- `enabled`: boolean (default true)

Use `conclave schedule apply --enable` to create and enable user-level timers.
Use `--disable-legacy` to disable the legacy `conclave-reconcile.timer`.
Schedules are validated with `systemd-analyze calendar` unless `--no-validate` is set.

### Quality options
- `strict`: if true, Conclave will refuse low-evidence answers
- `min_evidence`: minimum number of evidence items to proceed
- `low_signal_threshold`: minimum signal score for confidence
- `high_signal_threshold`: score to allow high confidence
- `high_evidence_min`: minimum evidence items required for high confidence
- `pdf_ratio_limit`: if too PDF-heavy, downshift confidence
- `off_domain_ratio_limit`: if evidence is mostly outside target collections, downshift confidence
- `strict_exit_code`: exit code returned by CLI when strict mode blocks a response
- `domain_paths`: optional path globs used to mark NAS items as on-domain by topic
- `trust_explicit_collections`: when true, skip catalog lookup for explicit collections

### Sources
- `sources.health_dashboard_url`: base URL for health.tannner.com (local health dashboard)
- `sources.health_pages`: pages to scrape for health context
- `sources.money_api_url`: base URL for money.tannner.com
- `sources.money_endpoints`: API endpoints to include as evidence
