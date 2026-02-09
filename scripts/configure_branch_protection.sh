#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required"
  exit 1
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <owner/repo> <branch> [required-check-context ...]"
  echo "Example: $0 tannernicol/agent-conclave main \"CI / test (3.10)\" \"Hygiene / pre_commit\""
  exit 1
fi

owner_repo="$1"
branch="$2"
shift 2

payload="$({
python - "$@" <<'PY'
import json, sys
contexts = sys.argv[1:]
required = None
if contexts:
    required = {
        "strict": True,
        "checks": [{"context": c} for c in contexts],
    }
payload = {
    "required_status_checks": required,
    "enforce_admins": True,
    "required_pull_request_reviews": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews": True,
        "require_code_owner_reviews": True,
        "require_last_push_approval": False,
    },
    "restrictions": None,
    "required_linear_history": True,
    "allow_force_pushes": False,
    "allow_deletions": False,
    "block_creations": False,
    "required_conversation_resolution": True,
    "lock_branch": False,
    "allow_fork_syncing": True,
}
print(json.dumps(payload))
PY
})"

gh api --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/${owner_repo}/branches/${branch}/protection" \
  --input - <<<"${payload}" >/dev/null

echo "Branch protection applied for ${owner_repo}:${branch}"
