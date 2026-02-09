# Sanitized Workflow Example

This workflow demonstrates Agent Conclave using fully synthetic identifiers and data.

## Goal

Show practical usage for engineers without exposing personal, internal, or sensitive context.

## Scenario

- Environment: team-lab
- Target service: service-a
- Region: region-1
- Data class: synthetic only

## Steps

1. Run setup and baseline checks.
2. Execute the core workflow.
3. Capture deterministic output artifacts.
4. Validate against policy and hygiene gates.
5. Share the redacted summary with reviewers.

## Commands

- python scripts/demo.py --config config/local.yaml
- python scripts/benchmark_synthetic.py --format markdown
- python scripts/redact.py --self-check

## Expected Output

- No secrets, keys, personal details, or internal hostnames.
- Deterministic output suitable for documentation and demos.
- Output safe for public OSS examples and employer review.
