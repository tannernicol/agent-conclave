# Contributing

Thanks for your interest in Agent Conclave.

## Getting Started

```bash
git clone https://github.com/tannernicol/agent-conclave.git
cd agent-conclave
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Verification

Run these before opening a PR:

```bash
ruff check conclave tests          # lint
python -m pytest tests/ -q         # unit tests
python -m compileall conclave      # syntax check
conclave validate --config         # config schema validation
node --test tests/ui_state.test.js # JS unit tests (optional, requires Node)
```

## Running the Dashboard

```bash
conclave health          # verify models are reachable
uvicorn conclave.server:app --port 8099  # start the web UI
```

## Pull Requests

1. Fork the repo and create a focused branch.
2. Include tests for behavior changes.
3. Run the verification commands above before opening a PR.
4. Keep PRs scoped to one logical change.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
