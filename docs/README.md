# Documentation

Start with the [repository overview](../README.md). These guides describe the
implemented AI component and distinguish it from the broader wearable project.

## Guides

| Guide | Purpose |
|---|---|
| [Architecture](architecture.md) | Components, signal processing, model, and system boundaries |
| [Python package](../ai/README.md) | Modules, APIs, and individual pipeline commands |
| [Reproducibility](reproducibility.md) | Public experiment, expected artifacts, and verification |
| [Research notes](research.md) | Dataset provenance, evaluation methods, limitations, and citation |
| [HTTP service](ai-service-integration-v0.1.md) | Installation, configuration, endpoints, and raw-window processing |
| [C++ kernels](../firmware/README.md) | Host compilation and inference parity |
| [Contributing](../CONTRIBUTING.md) | Development environment, checks, and change review |

## Specifications

Machine-readable contracts are in [ai/contracts](../ai/contracts/README.md).
Version numbers identify file formats, not clinical validation levels.

| Reference | Status |
|---|---|
| [CSV data contract v1.1](ai-data-contract-v1.1.md) | Implemented schema and logger |
| [Runtime output v0.2](ai-software-integration-contract-v0.2.md) | Current application contract |
| [Label rulebook v0.3](ai-label-rulebook-v0.3.md) | Label structure specified; real-data boundaries remain unset |
| [Pipeline configuration](../ai/configs/README.md) | Versioned settings and provenance |
| [Real sensor inventory](real-pilot-data-audit-v0.1.md) | Unlabeled characterization data; raw files private |

The current references are in English. Original Indonesian specifications are
retained in the archive for technical history.

## Evidence and history

- [September 2026 evaluation](ai-evaluation/2026-09-09/README.md): the original
  publication bundle, including tables, figures, and its recorded checks.
  Reports and figure labels retain their original language. Historical check
  results describe that snapshot, not the current branch.
- [Archived design and integration notes](archive/README.md): earlier proposals,
  superseded contracts, and dated internal audits. These are background records,
  not current deployment instructions.

The current source, configuration, schemas, and tests take precedence over
historical design notes when describing executable behavior.
