# Portable C++ kernels

This directory provides C++17 headers for the CSV contract, capacitive feature
extraction, and ordinal inference. They are host-tested components intended to
support embedded integration; a complete ESP32 firmware application is not
included.

| Header | Responsibility |
|---|---|
| `include/ostosense/data_contract.hpp` | Shared data and quality definitions |
| `include/ostosense/capacitive_features.hpp` | Five capacitive window features |
| `include/ostosense/ordinal_inference.hpp` | Scaled ordinal logistic inference |

## Build and test

From the repository root, using Python and a GCC/Clang-compatible C++17 compiler:

```bash
python scripts/verify.py --cpp-only
```

Alternatively, use CMake and CTest:

```bash
cmake -S firmware -B build/firmware
cmake --build build/firmware --config Release
ctest --test-dir build/firmware -C Release --output-on-failure
```

CMake selects the installed compiler; no external C++ libraries are required.

## Model export and parity

```bash
python scripts/reproduce.py --output outputs/synthetic-run
```

The `edge/` output contains generated parameters and golden vectors. Keep these
run-specific files outside tracked headers. Python tests reconstruct all 38
synthetic validation windows and compare feature values, probabilities, and
classes with the C++ implementation. Probability/feature comparisons use
tolerances; class agreement is exact for these vectors.

Host tests do not establish hardware timing, memory consumption, sensor
calibration, or on-device reliability.
