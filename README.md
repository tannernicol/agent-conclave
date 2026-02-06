<div align="center">
  <img src="logo.svg" width="96" height="96" alt="Agent Conclave logo" />
  <h1>Agent Conclave</h1>
  <p><strong>Auditable multi-model consensus for security-sensitive AI decisions</strong></p>
  <p>
    <a href="https://tannner.com">tannner.com</a> ·
    <a href="https://github.com/tannernicol/agent-conclave">GitHub</a>
  </p>
</div>

---

## What it does

Agent Conclave turns multiple LLM outputs into a single, traceable decision artifact. It orchestrates model-agnostic adapters with configurable consensus strategies, policy hooks, and guardrails so that security teams can cross-check findings before humans see the signal. Every decision is reproducible and ships with a full audit trail.

## Key features

- Model-agnostic adapters for local, hosted, and self-hosted LLMs
- Configurable consensus strategies with deterministic replay
- Policy hooks and guardrails for sensitive workflows
- Full audit trails with replayable traces

## Stack

- Python

## Getting started

```bash
git clone https://github.com/tannernicol/agent-conclave.git
cd agent-conclave
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/example.yaml config/local.yaml
python scripts/demo.py --config config/local.yaml
```

## Author

**Tanner Nicol** — Principal Security Infrastructure Engineer
[tannner.com](https://tannner.com) · [GitHub](https://github.com/tannernicol) · [LinkedIn](https://linkedin.com/in/tanner-nicol-60b21126)
