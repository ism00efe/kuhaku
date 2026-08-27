# Security Policy

## Supported versions

kuhaku is pre-1.0 (currently 0.1.0, alpha). Only the latest release on PyPI receives
security fixes -- there is no backport policy yet.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability. Use GitHub's private security
advisory form instead: go to the
[Security tab](https://github.com/ism00efe/kuhaku/security/advisories/new) on this
repository and open a new draft advisory.

Include what you'd include in a bug report: the affected version, a reproduction, and
the impact you believe it has. There is no fixed SLA at this stage of the project, but
reports are read and acknowledged.

## Scope

This applies to `kuhaku` itself -- the code in this repository, published as the
`kuhaku` package on PyPI. It does not cover:

- vulnerabilities in a dependency (report those upstream, to the dependency's own
  project);
- a deployment-specific misconfiguration (e.g. an application built on `kuhaku` that
  disables a security control kuhaku itself has no switch for -- see AGENTS.md's "no
  off switch" rule);
- issues in `_superseded/` or any other content explicitly marked as historical/removed.

## What "security-relevant" means here

kuhaku's own security surface is PII sanitization (`core/sanitization.py`), the
prompt-injection input guard (`core/security/guard.py`), document-level access
filtering (`tools/rag/retriever.py`'s `access_tags`/`AuthContext` enforcement), and the
audit trail (`core/security/audit.py`). A report about any of these silently failing,
being bypassable, or leaking data it should have masked/blocked is exactly what this
policy is for.
