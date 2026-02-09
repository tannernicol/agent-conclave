# Contributing

Thanks for contributing to Agent Conclave.

## Workflow

1. Fork the repository and create a focused branch.
2. Implement your change with tests/docs updates.
3. Run local validation before opening a PR.
4. Open a PR with a concise summary and risk notes.

## Local Validation

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
pre-commit run --all-files
python scripts/redact.py --self-check
```

## Pull Request Expectations

- Keep each PR scoped to one logical change.
- Include tests for behavior changes.
- Update docs and examples when interfaces change.
- Do not include real credentials, internal domains, private IPs, or personal data.
