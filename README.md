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
```

## Configuration
- Default config: `config/default.yaml`
- Local override: `~/.config/conclave/config.yaml`

## Data locations
- Conclave state: `/mnt/nas/Homelab/BugBounty/conclave`
- Model registry: `/mnt/nas/Homelab/BugBounty/conclave/models/registry.json`
- Benchmarks: `/mnt/nas/Homelab/BugBounty/conclave/models/benchmarks.jsonl`
- Runs: `/mnt/nas/Homelab/BugBounty/conclave/runs/`

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

## Tests (smoke + planner)

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Set `CONCLAVE_SMOKE_STRICT=1` (or `CI=1`) to require a live Ollama smoke run.
