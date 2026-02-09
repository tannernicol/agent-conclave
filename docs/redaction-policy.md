# Redaction Policy

Sanitize examples, logs, screenshots, and reports before publishing.

## Always Redact

- Emails and usernames
- Credentials, API keys, and tokens
- Internal domains, hostnames, and private IPs
- Any customer or proprietary incident data

## Safe Replacements

- Email: `user@example.com`
- Domain: `example.internal`
- IP: `10.0.0.0`
- Secret: `REDACTED`

## Verification

Run:

```bash
python scripts/redact.py --self-check
```
