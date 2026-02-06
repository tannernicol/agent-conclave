# Use Cases

## Security Incident Triage
Use when you need a defensible AI-assisted decision for a high-impact incident.

Inputs:
- Incident summary
- Evidence bundle
- Policy constraints

Outputs:
- Consensus decision
- Audit trail
- Model response breakdown

Notes:
- Require human approval for action steps.

## Policy-Gated Tool Approvals
Use when you want model advice but must enforce a strict policy gate.

Inputs:
- Tool request
- Policy rules
- Context snapshot

Outputs:
- Allow/deny decision
- Policy hit list
- Decision artifact

Notes:
- Keep logs for compliance.

## Model Regression Reviews
Use when comparing model updates for consistency and drift.

Inputs:
- Evaluation prompts
- Baseline outputs
- New model outputs

Outputs:
- Diff report
- Consensus delta
- Risk summary

Notes:
- Document model versions in the artifact.
