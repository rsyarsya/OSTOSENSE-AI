# Security

## Reporting a vulnerability

Do not include credentials, private sensor recordings, or participant information
in a public issue. If GitHub private vulnerability reporting is available, use
the repository's Security tab. Otherwise, ask the
[maintainer](https://github.com/rsyarsya) for a private reporting channel without
posting exploit details or sensitive data.

Include the affected commit, a minimal synthetic reproduction, expected impact,
and any proposed fix. No response-time guarantee or supported long-term release
line is currently offered.

## Service boundary

The HTTP adapter is an experimental backend service. It requires a bearer token,
binds to loopback by default, limits request size, and disables synthetic and
experimental prediction modes unless explicitly enabled.

A deployment must provide TLS or an appropriately isolated network, device/session
authorization, request freshness checks, and access control outside this stateless
adapter. The service does not store patient identities or operate a database.

Keep tokens out of browser/mobile clients and Git. Do not substitute fabricated
readings when transport or sensors fail. Experimental AI output must remain
distinct from direct contact detection and connection status.

## Data handling

Private raw recordings, participant records, credentials, and environment files
are excluded from this repository. Synthetic fixtures are provided for testing.
If sensitive material is accidentally published, notify the maintainer privately
and revoke any exposed credentials; deleting a file in a later commit does not
remove it from repository history.

For model limitations, see [Research notes](docs/research.md). For intellectual
property and reuse permissions, see [REUSE.md](REUSE.md).
