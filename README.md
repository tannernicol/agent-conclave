<div align="center">
  <img src="logo.svg" width="96" height="96" alt="Agent Conclave logo" />
  <h1>Agent Conclave</h1>
  <p><strong>Multi-model consensus engine — local agents self-organize and iterate until an agreed best solution is reached</strong></p>
  <p>
    <a href="https://tannner.com">tannner.com</a> ·
    <a href="https://github.com/tannernicol/agent-conclave">GitHub</a>
  </p>
</div>

---

<p align="center">
  <img src="docs/demo.png" alt="Conclave demo" width="700" />
</p>

## The Problem

You're making a high-stakes decision — shipping a deploy, choosing an architecture, reviewing a security change. You ask Claude. Then you ask GPT. Then you ask Llama. You copy-paste between tabs, mentally diff the answers, and try to synthesize a verdict. Every time.

## The Solution

Conclave automates multi-model deliberation. Send one query, get structured consensus from N models that see each other's reasoning and iterate until they converge. Simulated annealing controls exploration vs. exploitation. Every decision produces a replayable audit trail.

**One query in, one verdict out. No more copy-paste consensus.**

## Key Features

- **Model-agnostic adapters** for local, hosted, and self-hosted LLMs (Ollama, OpenAI, Anthropic)
- **Multi-round deliberation** — models cross-validate each other's reasoning
- **Configurable consensus strategies** with deterministic replay and simulated annealing
- **Policy hooks and guardrails** for sensitive workflows
- **Full audit trails** with replayable traces — every decision is reproducible

## Quick Start

```bash
git clone https://github.com/tannernicol/agent-conclave.git
cd agent-conclave
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config/example.yaml config/local.yaml
# Edit config/local.yaml with your model endpoints

python scripts/demo.py --config config/local.yaml
```

```
$ conclave run --panel claude,gpt4,llama \
    "Is this deployment safe to ship?"
→ Panel: 3 models, 12 reasoning chains
→ Verdict: HOLD (2-1) — claude flagged auth regression
```

## How It Works

1. **Fan-out** — sends the same prompt to N models in parallel
2. **Score** — each response is scored against configurable rubrics
3. **Iterate** — models see each other's responses and refine (simulated annealing controls exploration)
4. **Converge** — stops when consensus threshold is met or max rounds reached
5. **Audit** — full decision trace written to JSON for reproducibility

## Requirements

- Python 3.10+
- At least one LLM provider (Ollama for local, or OpenAI/Anthropic API keys)

## Author

**Tanner Nicol** — [tannner.com](https://tannner.com) · [GitHub](https://github.com/tannernicol) · [LinkedIn](https://linkedin.com/in/tanner-nicol-60b21126)

## License

MIT — see [LICENSE](LICENSE).
