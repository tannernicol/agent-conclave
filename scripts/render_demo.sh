#!/usr/bin/env bash
set -euo pipefail

if ! command -v vhs >/dev/null 2>&1; then
  echo "vhs not found. Install: https://github.com/charmbracelet/vhs" >&2
  exit 1
fi

vhs demo.tape
