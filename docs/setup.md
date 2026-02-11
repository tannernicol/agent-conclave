# Setup Guide

## Prerequisites
- Python 3.10+
- `pip` or `uv`

## Step 1: Environment
1. Create a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`

## Step 2: Configuration
1. Copy `config/example.yaml` to `config/local.yaml`.
2. Fill in model endpoints and API keys.
3. Keep secrets out of git.

## Step 3: Dry Run
Run the demo harness to verify wiring:
```
python scripts/demo.py --config config/local.yaml
```

Expected output: a summary showing multiple model responses and a consensus decision.
