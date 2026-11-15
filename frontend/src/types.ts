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
  policy_version: string;
  feature_count: number;
  history_transactions_used: number;
}

export interface HealthResponse {
  status: "ok" | "unavailable";
  model_version?: string | null;
  feature_version?: string | null;
  policy_version?: string | null;
  database?: "ready" | "unavailable" | null;
  artifact?: Record<string, unknown> | null;
}

export interface ApiError {
  error?: { code?: string; message?: string };
  detail?: unknown;
}