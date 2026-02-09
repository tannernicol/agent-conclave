# Release Policy

## Cadence

- Target one tagged release per month.
- Ship patch releases as needed for security or correctness.

## Versioning

- Follow semantic versioning (MAJOR.MINOR.PATCH).
- MAJOR: breaking API or workflow changes.
- MINOR: backward-compatible features.
- PATCH: bug fixes and non-breaking maintenance.

## Changelog Discipline

- Keep an Unreleased section updated continuously.
- Each PR touching behavior should include a changelog note.
- On release, move Unreleased notes into a dated version entry.

## Release Gate

- CI, hygiene, security, and SBOM workflows must pass.
- Threat model and hardening checklist reviewed for impactful changes.
- Public artifacts must pass redaction checks.
