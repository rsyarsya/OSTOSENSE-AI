# Model card: OSTOSENSE ordinal classifier

## Identity

| Field | Value |
|---|---|
| Model ID | `engineering-v0.3` |
| Artifact format | `0.1.0` |
| Family | StandardScaler + mord.LogisticAT |
| Task | Four-class ordinal classification of capacitive sensor windows |
| Training origin | Synthetic fixture v0.3 |
| Development set | 76 windows; four eligible sessions |
| Validation set | 38 windows; two separate sessions |
| Model SHA-256 | `ceb2922841431a38c274e0ebd5c350873a28a95f40a6f6aab2194a1603b93cef` |

## Inputs and outputs

Input is a vector of five features from a single capacitive channel:
mean delta, last delta, slope per second, population variance, and range.
Delta means raw reading minus the session baseline. Windows are 120 seconds
long with a 10-second stride.

The model stores five coefficients, three ordered thresholds, and scaler
parameters. Output classes are `Safe`, `Monitor`, `Caution`, and `Urgent`.
They do not represent calibrated patient probabilities, leakage countdowns,
fluid volume, or skin condition.

## Intended use

- Inspecting and reproducing the synthetic experiment.
- Testing Python and C++ numerical implementations.
- Exercising application payloads and interfaces with synthetic examples.
- Operator-confirmed real-sensor integration experiments with explicit
  `UNVALIDATED` status.

The real-input path has not demonstrated equivalent signal scaling or
predictive validity. A class may remain constant under a shifted sensor
distribution. Operator confirmation does not establish calibration.

## Evaluation

On the 38 synthetic validation windows, 36 predictions match the labels:
accuracy 94.74%, Macro F1 0.9529, quadratic weighted kappa 0.9733.
Two Monitor windows are predicted as Safe. Support is 21/8/6/3 in class order.
Overlapping windows from two sessions do not establish independent-user
performance.

The model was not trained on the private P001–P007 recordings. No real-data
classification accuracy, clinical benefit, or patient notification performance
has been established. Experimental outputs must not trigger patient care
decisions.

## Provenance and artifacts

- `ordinal_model.json`: scaler, coefficients, thresholds, and feature order.
- `training_manifest.json`: source hashes, dependency versions, and fit policy.
- `examples.json`: four actual synthetic validation inputs and calculated outputs.
- `bundle.json`: hashes checked by the loader.

Hash checks establish asset integrity, not model validity. Rebuild into a new
directory and compare results before replacing a bundle. In a repository clone,
see `docs/reproducibility.md` for commands and `REUSE.md` for reuse terms.
