# Examples

## Quick Start

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp config/example.yaml config/local.yaml
python scripts/demo.py --config config/local.yaml
```

## Example Output

```
{
  "models": [
    "local-ollama"
  ],
  "responses": [
    "response from local-ollama"
  ],
  "consensus": "response from local-ollama"
}
```
