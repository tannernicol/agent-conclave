#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

ALLOWLIST_DOMAINS = {
    "github.com",
    "shields.io",
    "example.com",
    "example.internal",
}

ALLOWLIST_IPS = {
    "10.0.0.0",
}

PATTERNS = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("domain", re.compile(r"\b[a-zA-Z0-9-]+\.(com|net|org|io|ai|dev|app|co)\b")),
]

REPLACEMENTS = {
    "email": "user@example.com",
    "ip": "10.0.0.0",
    "github_token": "REDACTED_TOKEN",
    "aws_access_key": "REDACTED_AWS_KEY",
    "domain": "example.internal",
}


def is_allowed_domain(value: str) -> bool:
    return value.lower() in ALLOWLIST_DOMAINS


def scan_text(text):
    hits = []
    for name, rx in PATTERNS:
        for m in rx.finditer(text):
            value = m.group(0)
            if name == "domain" and is_allowed_domain(value):
                continue
            if name == "email":
                domain = value.split("@")[-1].lower()
                if is_allowed_domain(domain):
                    continue
            if name == "ip" and value in ALLOWLIST_IPS:
                continue
            hits.append((name, value))
    return hits


def redact_text(text):
    for name, rx in PATTERNS:
        text = rx.sub(REPLACEMENTS[name], text)
    return text


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if ".git" in path.parts:
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf"}:
            continue
        yield path


def self_check(root: Path) -> int:
    problems = []
    for path in iter_files(root):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        hits = scan_text(text)
        if hits:
            problems.append((path, hits[:3]))
    if problems:
        for path, hits in problems:
            print(f"{path}: {hits}")
        return 1
    print("redaction check: OK")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--in", dest="input")
    parser.add_argument("--out", dest="output")
    args = parser.parse_args()

    root = Path(".")
    if args.self_check:
        raise SystemExit(self_check(root))

    if not args.input or not args.output:
        parser.error("--in and --out are required unless --self-check")
    text = Path(args.input).read_text()
    Path(args.output).write_text(redact_text(text))


if __name__ == "__main__":
    main()
