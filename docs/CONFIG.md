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
- `mcp`: detected from `~/.mcp.json` and logged (no tool calls by default)
- `topics`: scheduled re-run topics

### RAG options
- `use_server_collections`: discover collections from rag.tannner.com
- `skip_empty_collections`: avoid empty collections (file_count == 0)
- `max_results_per_collection`: cap results per collection
- `prefer_non_pdf`: deprioritize PDF-heavy results
- `dynamic_patterns`: include extra collections by keyword
