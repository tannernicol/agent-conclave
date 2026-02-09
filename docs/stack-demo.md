# Private Stack Demo (Cross-Repo)

This walkthrough shows how the four public repos can work together in a private on-prem workflow:

1. Helm Suite provisions and operates the private runtime.
2. Grimoire indexes approved security knowledge for local retrieval.
3. Agent Conclave orchestrates multi-model review and consensus.
4. Ollama Says continuously validates local model defenses.

## Flow

1. Bring up local stack and health checks in Helm Suite.
2. Populate Grimoire with approved public security sources.
3. Run Agent Conclave for architecture and security review prompts.
4. Run Ollama Says suite before promoting model changes.
5. Publish sanitized benchmark and audit artifacts.

## Repositories

- Helm Suite: https://github.com/tannernicol/helm-suite
- Grimoire: https://github.com/tannernicol/grimoire
- Agent Conclave: https://github.com/tannernicol/agent-conclave
- Ollama Says: https://github.com/tannernicol/ollama-says

## Guardrails

- Use synthetic example data in public demos.
- Never publish internal hostnames, credentials, or personal datasets.
- Keep private findings and proprietary workflows in private repos.
