# Good First Issues

These starter tasks are designed for first-time contributors.

## 1) Add CLI smoke test for help output
Acceptance criteria:
- Adds a test that validates command help output.
- Test runs in CI and passes on Python 3.10/3.11/3.12.

## 2) Expand sanitized example inputs
Acceptance criteria:
- Adds one new synthetic example workflow under examples.
- Includes expected output and redaction-safe sample logs.

## 3) Improve docs cross-links
Acceptance criteria:
- Adds missing links between README and docs guides.
- No broken relative links.

## 4) Add troubleshooting section
Acceptance criteria:
- Documents three common failure modes and fixes.
- Includes minimal reproducible commands.

## 5) Add benchmark export format test
Acceptance criteria:
- Tests scripts/benchmark_synthetic.py for markdown and json output.
- Ensures deterministic output shape.
