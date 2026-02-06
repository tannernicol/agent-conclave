#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   RSYNC_SOURCE=../internal-docs ./scripts/sync_sanitized.sh

if [[ -z "${RSYNC_SOURCE:-}" ]]; then
  echo "Set RSYNC_SOURCE to the internal source directory" >&2
  exit 1
fi

TARGET_DIR="./docs"
rsync -av --delete "$RSYNC_SOURCE/" "$TARGET_DIR/"

# Redaction check
python3 scripts/redact.py --self-check

# Reminder
echo "Sync complete. Review diffs before commit."
