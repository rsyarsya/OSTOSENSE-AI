export type CanonicalFeatureOrderV01 = readonly [
  "cap_delta_mean",
  "cap_delta_last",
  "cap_delta_slope_per_s",
  "cap_delta_variance",
  "cap_delta_range",
];

export type CanonicalFeatureValuesV01 = readonly [
  number,
  number,
  number,
  number,
  number,
];

type AiFeatureInputBaseV01 = {
  feature_input_version: "0.1.0";
  source_window_end_ms: number;
  feature_basis: "RAW_MINUS_SESSION_BASELINE";
  feature_order: CanonicalFeatureOrderV01;
  features: CanonicalFeatureValuesV01;
};

export type AiFeatureInputV01 = AiFeatureInputBaseV01 &
  (
    | {
        data_source: "SYNTHETIC_FIXTURE";
        model_input_channel: "SYNTHETIC_CAPACITIVE";
      }
    | {
        data_source: "REAL_SENSOR";
        model_input_channel: "Kap_7";
      }
  );
