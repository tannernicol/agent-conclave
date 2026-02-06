# Architecture Overview

Agent Conclave orchestrates multiple model responses and aggregates them into a defensible, auditable output.

## Components
- Adapter layer for hosted and local models
- Orchestrator for parallel model calls
- Consensus engine for scoring and aggregation
- Policy and guardrail layer
- Audit logger for reproducible results

## Data Flow
1. Request enters orchestrator with a policy context
2. Adapters generate candidate responses
3. Consensus engine scores and aggregates
4. Guardrails filter or require escalation
5. Final output + audit bundle is stored
