# OSTOSENSE AI

[![Quality](https://github.com/rsyarsya/OSTOSENSE-AI/actions/workflows/quality.yml/badge.svg)](https://github.com/rsyarsya/OSTOSENSE-AI/actions/workflows/quality.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB)](ai/pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C)](firmware/README.md)

**Reproducible ordinal classification for an experimental ostomy monitoring system.**

OSTOSENSE AI turns capacitive sensor windows into four ordered classes:
**Safe, Monitor, Caution, and Urgent**. It provides a Python data pipeline,
an inspectable ordinal logistic regression model, portable C++ inference,
and a typed HTTP interface for application integration.

Developed as part of the OSTOSENSE PKM-KC project at Universitas Gadjah Mada,
this repository contains the AI and data-processing component. The wearable
electronics, patient application, and clinical dashboard are separate systems.

> **Research prototype.** The bundled model is trained on synthetic data.
> Reported classification scores are internal synthetic validation results.
> Performance on real users and clinical benefit have not been established.

## What is included

- **Auditable data processing:** CSV contracts, quality checks, deterministic
  windowing, and five explicitly defined capacitive features.
- **Reproducible modeling:** synthetic fixtures, group-separated partitions,
  ordinal regression, validation metrics, and parameter export with provenance.
- **Portable inference:** dependency-free Python prediction and C++17 kernels,
  tested against shared reference vectors on a host computer.
- **Application interfaces:** versioned JSON Schemas, TypeScript types, a CLI,
  and an optional authenticated FastAPI service.
- **Research evidence:** published predictions, metrics, sensor summaries,
  and a clear account of which results can be reproduced from public inputs.

## Quick start

Use Python 3.11 or 3.12. Run commands from the repository root.

```bash
git clone https://github.com/rsyarsya/OSTOSENSE-AI.git
cd OSTOSENSE-AI
python -m venv .venv
```

Activate the environment:

| Shell | Command |
|---|---|
| Linux / macOS | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

Then install and run the bundled model:

```bash
python -m pip install -e ./ai
python -m ostosense_ai.bundled_model
```

The command verifies model integrity and prints four synthetic examples, one
for each class. No training dependencies, private data, or server are needed.
The distribution is named `ostosense-contract`; its Python imports are
`ostosense_ai` and `ostosense_contract`.

## Reproduce the synthetic experiment

```bash
python -m pip install -e "./ai[pipeline]"
python scripts/reproduce.py --output outputs/synthetic-run
```

This runs generation, feature extraction, labeling, training, validation, and
C++ parameter export. It refuses an existing output directory. See the
[reproduction guide](docs/reproducibility.md) for expected outputs and results.

### Published validation results

All three models below use the same **38 synthetic validation windows from
two sessions**, with 76 development windows. Windows overlap within sessions;
they are not 38 independent participants or experiments.

| Model | Correct / total | Accuracy | Macro F1 | Quadratic weighted kappa |
|---|---:|---:|---:|---:|
| Ordinal logistic regression | 36 / 38 | 94.74% | 0.9529 | 0.9733 |
| Logistic regression | 36 / 38 | 94.74% | 0.9529 | 0.9733 |
| Decision tree | 37 / 38 | 97.37% | 0.9775 | 0.9865 |

OLR is retained for its ordered output and compact parameterization, not because
it outperforms the alternatives. The two OLR errors are Monitor windows predicted
as Safe. These results exercise the pipeline; they do not measure real-world
leakage detection. [Methods and limitations](docs/research.md) ·
[Source metrics](docs/ai-evaluation/2026-09-09/tables/model_metrics.csv)

Separately, 11 real sensor sessions contain 21,887 readings, 2,184 one-second
summaries, and 94 unlabeled windows. They support sensor characterization, not
classifier accuracy. Raw recordings are not distributed.

## Documentation

| Task | Start here |
|---|---|
| Understand the implementation | [Architecture and feature definitions](docs/architecture.md) |
| Run or extend the pipeline | [Python package guide](ai/README.md) |
| Reproduce and interpret results | [Reproduction](docs/reproducibility.md) · [Research notes](docs/research.md) |
| Integrate the model into an application | [HTTP service](docs/ai-service-integration-v0.1.md) · [Runtime contract](docs/ai-software-integration-contract-v0.2.md) |
| Build the C++ kernels | [C++ guide](firmware/README.md) |
| Contribute or run quality checks | [Contributing](CONTRIBUTING.md) |
| Browse specifications and historical records | [Documentation index](docs/README.md) |

## Repository layout

| Directory | Contents |
|---|---|
| `ai/src/ostosense_ai/` | Features, modeling, inference, quality checks, and service |
| `ai/src/ostosense_contract/` | CSV schemas and logger |
| `ai/contracts/` | JSON Schemas, TypeScript definitions, and example payloads |
| `ai/configs/` | Versioned experiment configuration |
| `ai/tests/` | Unit, integration, and cross-language parity tests |
| `firmware/` | Portable C++ headers and host tests |
| `scripts/` | Reproduction, verification, and packaging tools |
| `docs/` | Guides, specifications, research evidence, and historical context |

## Citation and reuse

For research that references this implementation, cite the repository and the
exact commit used. Machine-readable metadata is provided in [CITATION.cff](CITATION.cff).
No DOI or associated peer-reviewed paper is claimed.

An open-source license has not been granted for this repository. Review the
[intellectual property and reuse notice](REUSE.md) before reusing or distributing
the code, model, or figures. Dependencies retain their own licenses.
