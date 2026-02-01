# Conclave

Model orchestration engine for Tanner's homelab. Conclave routes questions to local models, retrieves context from NAS/RAG, and produces versioned consensus decisions.

## Highlights
- Role-based planning (router, reasoner, critic, summarizer)
- Local Ollama only (no paid APIs by default)
- Versioned decisions with a "latest pope" consensus
- White-smoke UI when consensus is reached
- Weekly reconciliation via systemd timer

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Run the API + UI
python -m conclave.server
```

Open: http://localhost:8099

## CLI

```bash
python -m conclave.cli models list
python -m conclave.cli plan --query "Should I rebalance my tax lots?"
python -m conclave.cli run --query "Explain my latest health labs" --collection health-rag
python -m conclave.cli run --query "What did I file last year?" --collection tax-rag
python -m conclave.cli runs latest
python -m conclave.cli reconcile --topic tax-checkup
python -m conclave.cli schedule list
python -m conclave.cli schedule apply --enable
```

## Configuration
- Default config: `config/default.yaml`
- Local override: `~/.config/conclave/config.yaml`

## Data locations
- Conclave state: `/home/tanner/.conclave`
- Model registry: `/home/tanner/.conclave/models/registry.json`
- Benchmarks: `/home/tanner/.conclave/models/benchmarks.jsonl`
- Runs: `/home/tanner/.conclave/runs/`

To move storage to NAS, override `data_dir` + model paths in `~/.config/conclave/config.yaml`.

## Systemd (optional)

```bash
mkdir -p ~/.config/systemd/user
cp systemd/conclave.service ~/.config/systemd/user/
cp systemd/conclave-reconcile.* ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now conclave.service
systemctl --user enable --now conclave-reconcile.timer
```

## Notes
- Conclave can query homelab-search (`rag.tannner.com`) when available.
- Use `config/default.yaml` to tune roles, RAG collections, and index paths.
- Bounty can invoke Conclave by calling `conclave run --query ...` from its pipeline.
- Each run writes `audit.jsonl` with routing, role assignments, and disagreements.
- Use `conclave schedule apply` to materialize systemd timers from topic schedules.
- Add `--disable-legacy` to turn off the legacy `conclave-reconcile.timer`.
- NAS index auto-build is off by default; run `conclave index` to build it.
- MCP servers are detected from `~/.mcp.json` and logged into the audit trail.
- RAG collections are discovered from `rag.tannner.com` and expanded by domain patterns.

## Tests (smoke + planner)

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Set `CONCLAVE_SMOKE_STRICT=1` (or `CI=1`) to require a live Ollama smoke run.
