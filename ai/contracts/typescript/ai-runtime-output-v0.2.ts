export type RiskPrediction =
  | { risk_class: "Safe"; risk_class_index: 0 }
  | { risk_class: "Monitor"; risk_class_index: 1 }
  | { risk_class: "Caution"; risk_class_index: 2 }
  | { risk_class: "Urgent"; risk_class_index: 3 };

export type AiRuntimeUnavailableV02 = {
  runtime_output_version: "0.2.0";
  mode: "LIVE";
  data_source: "NONE";
  model_status: "UNAVAILABLE";
  prediction_available: false;
  risk_class: null;
  risk_class_index: null;
  source_window_end_ms: null;
  model_input_channel: null;
  model_artifact_version: null;
  model_artifact_sha256: null;
  evidence_scope: "NO_PREDICTION";
  warning: "No usable AI prediction is available; no OSTOSENSE risk class was produced.";
};

type AiRuntimePredictionBaseV02 = {
  runtime_output_version: "0.2.0";
  prediction_available: true;
  source_window_end_ms: number;
  model_artifact_version: "0.1.0";
  model_artifact_sha256: string;
  warning: string;
};

export type AiRuntimeEngineeringTestV02 = AiRuntimePredictionBaseV02 &
  RiskPrediction & {
    mode: "ENGINEERING_TEST";
    data_source: "SYNTHETIC_FIXTURE";
    model_status: "TEST_ONLY";
    model_input_channel: "SYNTHETIC_CAPACITIVE";
    evidence_scope: "PIPELINE_MECHANICS_ONLY";
    warning: "ENGINEERING_TEST_ONLY synthetic result; not a patient-risk assessment or clinical output.";
  };

export type AiRuntimeLiveExperimentalV02 = AiRuntimePredictionBaseV02 &
  RiskPrediction & {
    mode: "LIVE_EXPERIMENTAL";
    data_source: "REAL_SENSOR";
    model_status: "UNVALIDATED";
    model_input_channel: "Kap_7";
    evidence_scope: "EXPERIMENTAL_UNVALIDATED";
    warning: "UNVALIDATED experimental class from a synthetic-trained model applied to real Kap_7 sensor features; not for patient notification or clinical action.";
  };

export type AiRuntimeOutputV02 =
  | AiRuntimeUnavailableV02
  | AiRuntimeEngineeringTestV02
  | AiRuntimeLiveExperimentalV02;
