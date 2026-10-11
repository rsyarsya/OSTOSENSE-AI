# Contributing

OSTOSENSE AI is a research prototype. Contributions should make its behavior
easier to reproduce, inspect, and integrate. Review [REUSE.md](REUSE.md) for the
current intellectual-property status before proposing external code reuse.

## Development setup

Use Python 3.11 or 3.12, Git, and a C++17 compiler. From the repository root,
create and activate a virtual environment as shown in the [README](README.md),
then install the complete development stack:

```bash
python -m pip install -c ai/requirements-serve.lock -e "./ai[pipeline,quality,serve,serve-test]"
python scripts/verify.py
```

The verifier uses the active Python interpreter and `g++` from PATH. To select
another GCC/Clang-compatible compiler:

```bash
export CXX=/path/to/g++
python scripts/verify.py
```

On Windows PowerShell:

```powershell
$env:CXX = "C:\msys64\ucrt64\bin\g++.exe"
python scripts/verify.py
```

The Windows verifier adds the compiler directory to the child process PATH
for MinGW runtime DLLs. The C++ CMake build is described in
[firmware/README.md](firmware/README.md).

## Quality checks

The full verifier runs:

1. Python tests, including HTTP behavior and Python–C++ parity.
2. Ruff correctness checks and Pyright.
3. Wheel/source builds and model loading from an isolated wheel installation.
4. Three standalone C++ host tests with compiler warnings treated as errors.
5. Repository hygiene and whitespace checks.

For focused work:

```bash
python -W error -m unittest discover -s ai/tests
python -m ruff check ai/src ai/tests scripts
python scripts/verify.py --cpp-only
python scripts/check_published_metrics.py
```

Minimal environments may skip tests requiring optional packages or a compiler.
Install the full extras for a complete verification. Container startup is checked
separately in CI.

## Change expectations

- Explain the behavior being changed, the reason, and relevant validation.
- Preserve public payload fields and state meanings. Breaking contract changes
  require a new schema version and migration notes.
- Keep synthetic and real-data provenance explicit. Never promote a synthetic
  result to a real-world performance claim.
- Add tests for changed behavior and failure cases. Use synthetic fixtures, not
  participant records or private sensor files.
- Keep sample identifiers, sessions, bags, and sensors separated across model
  partitions. Real label boundaries require a documented experiment decision.
- Do not commit credentials, private recordings, generated model headers,
  virtual environments, or local build output.
- Record user-visible changes under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).

Write public documentation in plain English. Retain exact schema identifiers
where interoperability requires them, and explain them when first introduced.
Historical research records may retain their original language.

## Reporting problems

For reproducible bugs, include the commit, platform, Python version, command,
and a minimal synthetic input. Use the issue templates to distinguish a bug
from a documentation or research question. Follow [SECURITY.md](SECURITY.md)
for vulnerabilities; do not post secrets or participant data.
