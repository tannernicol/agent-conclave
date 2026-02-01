# Conclave Configuration

Config is layered:
1) `config/default.yaml`
2) `~/.config/conclave/config.yaml` (optional override)

Key sections:
- `server`: host/port
- `data_dir`: base directory for runs and indexes
- `models`: capability cards and defaults
- `planner`: role weights + preferences
- `rag`: homelab-search endpoint + default collections
- `index`: NAS allowlist + exclude patterns + auto_build/refresh
- `topics`: scheduled re-run topics
