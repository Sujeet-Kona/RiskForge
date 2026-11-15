export type Decision = "ALLOW" | "REVIEW" | "BLOCK";

export interface HistoryTransaction {
  TransactionDT: number;
  TransactionAmt: number;
  card1?: number | null;
  addr1?: number | null;
  addr2?: number | null;
}

export interface ScoreRequest {
  transaction: {
    TransactionID: string;
    TransactionDT: number;
    TransactionAmt: number;
    card1: number;
    addr1?: number | null;
    addr2?: number | null;
    ProductCD?: string | null;
    card6?: string | null;
    P_emaildomain?: string | null;
    R_emaildomain?: string | null;
  };
  history: HistoryTransaction[];
  top_k: number;
}

export interface ReasonCode {
  code: string;
  title: string;
  type: string;
  description: string;
  feature: string;
  contribution: number;
  direction: string;
  n_transformed_columns: number;
}

export interface ScoreResponse {
  transaction_id: string;
  risk_probability: number;
  decision: Decision;
  reason_codes: ReasonCode[];
  risk_decreasing_signals: ReasonCode[];
  decision_reason_code: { code: string; title: string; type: string; description: string };
  decision_explanation: string;
  model_detail: {
    base_value: number;
    raw_margin: number;
    raw_model_probability: number;
    attribution_space: string;
    attribution_space_note: string;
    total_risk_increasing: number;
    total_risk_decreasing: number;
  };
  policy_detail: { review_threshold: number; block_threshold: number };
  model_version: string;
  feature_version: string;
  calibration_method: string;
  policy_version: string;
  feature_count: number;
  history_transactions_used: number;
}

export interface AuditDecisionSummary {
  id: number | null;
  transaction_id: string;
  risk_probability: number;
  decision: Decision;
  model_version: string;
  feature_version: string;
  calibration_method: string;
  policy_version: string;
  created_at: string | null;
}

export interface AuditDecisionListResponse {
  items: AuditDecisionSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface ArtifactSummary {
  n_features?: number | null;
  [key: string]: unknown;
}

export interface HealthResponse {
  status: "ok" | "unavailable";
  model_version?: string | null;
  feature_version?: string | null;
  calibration_method?: string | null;
  policy_version?: string | null;
  database?: "ready" | "unavailable" | null;
  artifact?: ArtifactSummary | null;
}

export interface ApiError {
  error?: { code?: string; message?: string };
  detail?: unknown;
}

export type ReviewStatus = "PENDING" | "APPROVED" | "REJECTED";
export type ReviewOutcome = "APPROVED" | "REJECTED";

export interface ReviewCase {
  case_id: number;
  decision_id: number;
  status: ReviewStatus;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
  resolved_by: string | null;
  resolution_comment: string | null;
  transaction_id: string;
  risk_probability: number;
  decision: Decision;
  reason_codes: ReasonCode[];
  risk_decreasing_signals: ReasonCode[];
  decision_reason_code: { code: string; title: string; type: string; description: string };
  decision_explanation: string;
  model_detail: {
    base_value: number;
    raw_margin: number;
    raw_model_probability: number;
    attribution_space: string;
    attribution_space_note: string;
    total_risk_increasing: number;
    total_risk_decreasing: number;
  };
  policy_detail: { review_threshold: number; block_threshold: number };
  scored_inputs: Record<string, unknown>;
  model_version: string;
  feature_version: string;
  calibration_method: string;
  policy_version: string;
}

export interface ReviewQueueResponse {
  items: ReviewCase[];
  total: number;
  limit: number;
  offset: number;
}

export interface ReviewResolutionResponse {
  case: ReviewCase;
}
