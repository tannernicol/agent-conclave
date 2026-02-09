# Production Hardening Checklist

## Baseline

- [ ] Pin dependency versions for deployment artifacts
- [ ] Enforce trusted branch protection policy
- [ ] Enable GitHub secret scanning and Dependabot alerts
- [ ] Store runtime secrets outside repository files

## Runtime

- [ ] Run with least-privilege service accounts
- [ ] Restrict network egress to required endpoints only
- [ ] Enable structured logs with redaction filters
- [ ] Set resource limits (CPU, memory, timeout)

## Operations

- [ ] Define incident contact and escalation path
- [ ] Test backup and restore process quarterly
- [ ] Document rollback steps for each release
- [ ] Validate disaster recovery runbook annually

## Security Validation

- [ ] Run CI checks before every merge
- [ ] Re-run redaction checks before publishing docs/screenshots
- [ ] Review threat model per release
- [ ] Track unresolved high-risk findings to closure
