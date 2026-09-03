# ostosense_ai configs

This directory holds versioned, human-readable configuration for implemented
and planned OSTOSENSE AI pipeline stages:

- deterministic synthetic-generation seeds and scenario parameters,
- feature/window parameters (window length `W`, stride `S`),
- ordinal model hyperparameters (L2 strength) and parameter-export settings,
- labeling boundary configuration (`boundary_config_version`).

Production labeling boundaries (B1/B2/B3) remain `PILOT_PENDING` under the
structure-locked rulebook (`docs/ai-label-rulebook-v0.3.md`) and must stay
unset until derived from Tier 1 Development data. Numeric fixtures are allowed
only under `ai/tests/fixtures/`, marked `ENGINEERING_TEST_ONLY`, and guarded for
`SYNTHETIC_PIPELINE_TEST_ONLY` input.

Dependency decision for later batches:

- training model: `mord.LogisticAT` (all-threshold ordinal logistic, L2),
- evaluation library: `scikit-learn`,
- weighted agreement metric: quadratic weighted Cohen's kappa,
- edge parity target: exported class labels, not bit-identical floating point.

This decision does not lock any clinical or bench label boundary.

## Files

- `synthetic-v0.2.json` — config for the `ostosense_ai.synthetic`
  pipeline-test generator. It holds only synthetic signal/timing parameters,
  each classified in a `provenance` map (`CONTRACT_DERIVED`,
  `ENGINEERING_TEST_ONLY`, `LITERATURE_VERIFIED`, `PILOT_PENDING`). The
  generator emits a `manifest.json` sidecar marked `SYNTHETIC_PIPELINE_TEST_ONLY`
  and contains no label boundaries. Config v0.2 also records synthetic
  bag/sensor groups and independent visual-observation delays.
- `synthetic-v0.3.json` — extends `synthetic-v0.2.json` by preserving all raw
  records for the original nine sessions in order under the same seed, then
  appending two `validation`-partition sessions (`safe-validation`,
  `gradual-validation`). After filtering v0.3 to those nine session IDs, their
  CSV records are byte-identical in value and order; the complete v0.2 and v0.3
  files are not byte-identical because v0.3 contains the appended sessions.
  for grouped validation evaluation. Used with the
  `ai/tests/fixtures/ostosense-evaluation-v0.1/` manifests, which partition the
  nine original sessions as `development` and the two appended sessions as
  `validation` (no session/bag/sensor crosses partitions).
- `features-v0.1.json` — DRAFT working config for the `ostosense_ai.features`
  extractor. It fixes the working window convention (`(t-W,t]`, `W`=120 s,
  stride 10 s, 1 Hz, 200 ms jitter tolerance) and the exact five capacitive
  feature names, citing Data Collection Protocol v0.1 as the working source. It
  contains no B1/B2/B3 values, labels, clinical thresholds, or model
  hyperparameters.
- `training-v0.1.json` — `ENGINEERING_TEST_ONLY` config for the
  `ostosense_ai.training` ordinal trainer. It fixes the canonical trainer
  (`sklearn.preprocessing.StandardScaler` + `mord.LogisticAT`, `alpha=1.0`,
  `max_iter=10000`, `uniform_window` weighting), accepts only
  `SYNTHETIC_PIPELINE_TEST_ONLY` data, fits on `development` and rejects
  `final_test`. It has no random seed (the operation has no random step) and is
  strictly validated (missing/unknown keys, wrong types, and unsupported values
  are rejected).
- `raw-qc-v0.1.json` — `PROPOSED_PILOT_SETTING` config for the
  `ostosense_ai.raw_qc` real-data intake/QC gate. It fixes the provisional
  timing/baseline QC thresholds (`expected_interval_ms`, `jitter_tolerance_ms`,
  `unmarked_gap_threshold_ms`, `minimum_pre_injection_dry_s`, `baseline_window_s`)
  and the accepted `contract_version`/`protocol_version`, with a `provenance`
  map tagging every numeric setting `DRAFT_PROTOCOL_V0.1` and a `warning` that
  these are provisional engineering QC settings, not clinical thresholds,
  validated OSTOSENSE performance criteria, or label boundaries. It is strictly
  validated (missing/unknown keys, wrong types, unsupported versions, nonpositive
  values, and internally inconsistent timing are rejected). These values do not
  set or imply production B1/B2/B3, which remain `PILOT_PENDING`.
- `real-pilot-v0.1.json` — deterministic preparation config for the current
  flat P001-P007 ESP32 logger CSVs. It records the exact source-file inventory,
  known scenario/position metadata, 10 Hz to 1 Hz median aggregation, a clearly
  provisional 20-second per-session baseline, `(t-W,t]` windows (`W=120 s`,
  stride `10 s`), and channel roles. `Kap_4`, `Kap_5`, and `Kap_7` are feature
  candidates; `Res_15` and `Res_16` are retained only for descriptive
  correlation and the separate fail-safe path. P006 is excluded from the
  operational correlation summary because it is a sensor-fault test. The
  resulting windows remain unlabeled and must not be used for classifier
  fitting or performance claims.
