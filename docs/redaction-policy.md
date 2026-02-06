# Redaction Policy

This repo is private until all sensitive details are scrubbed.

## Remove or Replace
- Domains and hostnames (replace with `example.internal`)
- IP addresses (replace with `10.0.0.0/24`)
- Usernames and emails (replace with `user@example.com`)
- Tokens, API keys, secrets (remove entirely)
- Internal paths and service names (use placeholders)

## Verification
- Run `scripts/redact.py --self-check`
- Manually review diffs before pushing
