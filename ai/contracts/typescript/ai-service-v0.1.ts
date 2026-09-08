import type { AiRuntimeOutputV02 } from './ai-runtime-output-v0.2';
import type { AiFeatureInputV01 } from './ai-feature-input-v0.1';

export type RawWindowSample = {
  sample_no: number;
  elapsed_ms: number;
  kap_7_raw: number;
  kap_7_status: string;
};

export type AiRawWindowInputV01 = {
  request_version: '0.1.0';
  device_id: string;
  session_id: string;
  acquisition_id: string;
  operator_confirmed: boolean;
  source_window_end_ms: number;
  baseline_samples: RawWindowSample[];
  window_samples: RawWindowSample[];
};

// Runtime JSON Schema validation is still required; TS types do not validate network data.
export type AiServiceResponseV01 = {
  service_response_version: '0.1.0';
  ai: AiRuntimeOutputV02;
  diagnostics: {
    availability_reason: string;
    quality_verification: 'UNKNOWN' | 'OPERATOR_ONLY' | 'AUTOMATIC' | 'TEST_ONLY';
    baseline_value: number | null;
  };
  feature_input: AiFeatureInputV01 | null;
};
