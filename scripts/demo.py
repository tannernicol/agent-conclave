#!/usr/bin/env python3
import argparse
import json
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    models = cfg.get("models", [])
    responses = [f"response from {m.get('name')}" for m in models]
    consensus = responses[0] if responses else "no models configured"

    payload = {
        "models": [m.get("name") for m in models],
        "responses": responses,
        "consensus": consensus,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
