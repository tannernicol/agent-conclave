# Threat Model

## Scope

This model covers Agent Conclave public repository behavior and documented workflows.

## Assets

- Source code and release artifacts
- Configuration templates and example workflows
- CI pipelines and generated reports

## Trust Boundaries

- Developer workstation
- CI runners
- Local runtime environment
- Optional model/data providers

## Threats Considered

- Secret leakage through docs, logs, or examples
- Supply-chain risk in dependencies and GitHub Actions
- Unsafe defaults or bypassed guardrails
- Incomplete access boundaries in automation scripts

## Controls

- Redaction checks in CI and local workflows
- Pre-commit hygiene checks
- CodeQL and dependency audits
- Branch protection plus review requirements
- SBOM generation for release artifacts

## Out of Scope

- Private internal deployments not represented in this repo
- Unauthorized offensive testing
- Proprietary datasets and private network topology
