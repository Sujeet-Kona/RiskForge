import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import App from "./App";
import {
  getDecisionHistory,
  getHealth,
  getReviewQueue,
  resolveReviewCase,
  scoreTransaction,
} from "./api";
import type {
  AuditDecisionSummary,
  HealthResponse,
  ReasonCode,
  ReviewCase,
  ScoreResponse,
} from "./types";

// The whole API surface is mocked: these tests exercise rendering, state
// transitions and accessibility, not the request wiring, which src/api.test.ts
// covers against a stubbed fetch. Every identifier below is synthetic — none of
// it describes the frozen production artifact.
vi.mock("./api", () => ({
  getHealth: vi.fn(),
  getDecisionHistory: vi.fn(),
  getReviewQueue: vi.fn(),
  resolveReviewCase: vi.fn(),
  scoreTransaction: vi.fn(),
}));

const HEALTHY: HealthResponse = {
  status: "ok",
  model_version: "model-test",
  feature_version: "features-test",
  calibration_method: "isotonic",
  policy_version: "policy-test",
  database: "ready",
  artifact: { n_features: 28 },
};

const RISK_INCREASING: ReasonCode = {
  code: "amount_spike",
  title: "Amount far above this card's history",
  type: "numeric",
  description: "TransactionAmt is well above the card's recent median.",
  feature: "transaction_amount_vs_card_median",
  contribution: 0.412,
  direction: "risk_increasing",
  n_transformed_columns: 28,
};

const RISK_DECREASING: ReasonCode = {
  code: "address_history",
  title: "Address seen in prior transactions",
  type: "behavioral",
  description: "The address pair has recent history.",
  feature: "address_transaction_count_24h",
  contribution: -0.108,
  direction: "risk_decreasing",
  n_transformed_columns: 28,
};

const SCORE_RESPONSE: ScoreResponse = {
  transaction_id: "RF-TEST-1",
  risk_probability: 0.1834,
  decision: "BLOCK",
  reason_codes: [RISK_INCREASING],
  risk_decreasing_signals: [RISK_DECREASING],
  decision_reason_code: {
    code: "above_block_threshold",
    title: "Block",
    type: "policy",
    description: "Calibrated probability is at or above the block threshold.",
  },
  decision_explanation: "Calibrated probability 18.34% is above the 11.0% block threshold.",
  model_detail: {
    base_value: -2.1,
    raw_margin: -0.42,
    raw_model_probability: 0.396,
    attribution_space: "raw",
    attribution_space_note: "TreeSHAP values are computed in margin space.",
    total_risk_increasing: 0.412,
    total_risk_decreasing: -0.108,
  },
  policy_detail: { review_threshold: 0.075, block_threshold: 0.11 },
  model_version: "model-test",
  feature_version: "features-test",
  calibration_method: "isotonic",
  policy_version: "policy-test",
  feature_count: 28,
  history_transactions_used: 1,
};

const HISTORY_ROW: AuditDecisionSummary = {
  id: 42,
  transaction_id: "RF-TEST-9",
  risk_probability: 0.032,
  decision: "ALLOW",
  model_version: "model-test",
  feature_version: "features-test",
  calibration_method: "isotonic",
  policy_version: "policy-test",
  created_at: "2026-01-04T09:12:00Z",
};

const REVIEW_CASE: ReviewCase = {
  case_id: 4,
  decision_id: 99,
  status: "PENDING",
  created_at: "2026-01-04T09:12:00Z",
  updated_at: "2026-01-04T09:12:00Z",
  resolved_at: null,
  resolved_by: null,
  resolution_comment: null,
  transaction_id: "RF-TEST-4",
  risk_probability: 0.0912,
  decision: "REVIEW",
  reason_codes: [RISK_INCREASING],
  risk_decreasing_signals: [],
  decision_reason_code: {
    code: "between_thresholds",
    title: "Review",
    type: "policy",
    description: "Calibrated probability sits between the review and block thresholds.",
  },
  decision_explanation: "Calibrated probability 9.12% is above the 7.5% review threshold.",
  model_detail: SCORE_RESPONSE.model_detail,
  policy_detail: SCORE_RESPONSE.policy_detail,
  scored_inputs: { TransactionAmt: 245.5, ProductCD: "W" },
  model_version: "model-test",
  feature_version: "features-test",
  calibration_method: "isotonic",
  policy_version: "policy-test",
};

const EMPTY_HISTORY = { items: [], total: 0, limit: 8, offset: 0 };
const EMPTY_QUEUE = { items: [], total: 0, limit: 20, offset: 0 };

function scoringForm(): HTMLFormElement {
  return screen.getByRole("form", { name: "Score a transaction" });
}

function resultRegion(): HTMLElement {
  return screen.getByRole("region", { name: "Decision result" });
}

function reviewRegion(): HTMLElement {
  return screen.getByRole("region", { name: "Manual review queue" });
}

async function renderReady() {
  render(<App />);
  await screen.findByText("Ready");
}

function enterReviewerKey(key: string) {
  fireEvent.change(screen.getByLabelText(/reviewer api key/i), {
    target: { value: key },
  });
}

async function loadReviewQueue() {
  await renderReady();
  await enterReviewerKey("reviewer-secret");
  fireEvent.click(screen.getByRole("button", { name: "Load pending cases" }));
  await screen.findByText("RF-TEST-4");
}

beforeEach(() => {
  // The api module is mocked once for the whole file, so its call history would
  // otherwise carry over from the previous test.
  vi.clearAllMocks();
  vi.mocked(getHealth).mockResolvedValue(HEALTHY);
  vi.mocked(getDecisionHistory).mockResolvedValue(EMPTY_HISTORY);
  vi.mocked(getReviewQueue).mockResolvedValue(EMPTY_QUEUE);
});

describe("system status", () => {
  it("reports the feature contract size from the API rather than a constant", async () => {
    vi.mocked(getHealth).mockResolvedValue({
      ...HEALTHY,
      artifact: { n_features: 31 },
    });

    await renderReady();

    const card = screen.getByText("FEATURE CONTRACT").closest(".metric-card");
    expect(card).not.toBeNull();
    expect(within(card as HTMLElement).getByText("31")).toBeInTheDocument();
    expect(screen.queryByText("28")).not.toBeInTheDocument();
  });

  it("shows an unknown feature contract when the API cannot be reached", async () => {
    vi.mocked(getHealth).mockRejectedValue(new Error("connection refused"));

    render(<App />);

    expect(await screen.findByText("Offline")).toBeInTheDocument();
    expect(screen.getByText("API unavailable")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("connection refused");

    const card = screen.getByText("FEATURE CONTRACT").closest(".metric-card");
    expect(within(card as HTMLElement).getByText("—")).toBeInTheDocument();
  });

  it("disables scoring until the artifact and the audit store are both ready", async () => {
    vi.mocked(getHealth).mockResolvedValue({ ...HEALTHY, database: "unavailable" });

    render(<App />);

    expect(await screen.findByText("Audit store unavailable")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /run risk assessment/i }),
    ).toBeDisabled();
    expect(screen.getByText(/scoring is disabled/i)).toBeInTheDocument();
  });
});

describe("decision rendering", () => {
  it("starts in the empty state", async () => {
    await renderReady();

    expect(screen.getByText("Awaiting assessment")).toBeInTheDocument();
    expect(within(resultRegion()).getByRole("status")).toBeEmptyDOMElement();
  });

  it("renders the decision, calibrated probability, thresholds and both signal directions", async () => {
    vi.mocked(scoreTransaction).mockResolvedValue(SCORE_RESPONSE);
    await renderReady();

    fireEvent.submit(scoringForm());

    expect(await screen.findByText("Block")).toBeInTheDocument();
    expect(screen.getByText("18.34%")).toBeInTheDocument();
    expect(screen.getByText("Review 7.5%")).toBeInTheDocument();
    expect(screen.getByText("Block 11.0%")).toBeInTheDocument();
    expect(
      screen.getByText("Amount far above this card's history"),
    ).toBeInTheDocument();
    expect(screen.getByText("0.412")).toBeInTheDocument();
    expect(screen.getByText("Address seen in prior transactions")).toBeInTheDocument();
    expect(screen.getByText("-0.108")).toBeInTheDocument();
    expect(
      screen.getByText("Transaction RF-TEST-1 · 28 features · features-test"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Awaiting assessment")).not.toBeInTheDocument();
  });

  it("shows the scoring state while the request is in flight and clears it after", async () => {
    let release: (value: ScoreResponse) => void = () => {};
    vi.mocked(scoreTransaction).mockImplementation(
      () => new Promise<ScoreResponse>((resolve) => { release = resolve; }),
    );
    await renderReady();

    fireEvent.submit(scoringForm());

    const busy = await screen.findByRole("button", { name: /scoring transaction/i });
    expect(busy).toBeDisabled();
    expect(within(resultRegion()).getByRole("status")).toHaveTextContent(
      "Scoring transaction.",
    );

    release(SCORE_RESPONSE);

    expect(await screen.findByText("Block")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /run risk assessment/i }),
    ).toBeEnabled();
  });

  it("reuses one idempotency key for retries and rotates it when a field changes", async () => {
    vi.mocked(scoreTransaction).mockResolvedValue(SCORE_RESPONSE);
    await renderReady();

    fireEvent.submit(scoringForm());
    await screen.findByText("Block");
    fireEvent.submit(scoringForm());
    await waitFor(() => expect(scoreTransaction).toHaveBeenCalledTimes(2));

    const [first, second] = vi.mocked(scoreTransaction).mock.calls;
    expect(first[1]).toBe(second[1]);

    fireEvent.change(screen.getByLabelText(/transaction amount/i), {
      target: { value: "900.00" },
    });
    fireEvent.submit(scoringForm());
    await waitFor(() => expect(scoreTransaction).toHaveBeenCalledTimes(3));

    expect(vi.mocked(scoreTransaction).mock.calls[2][1]).not.toBe(first[1]);
  });

  it("rejects a history payload that is not an array before calling the API", async () => {
    await renderReady();

    fireEvent.change(screen.getByLabelText(/prior transaction history/i), {
      target: { value: '{"TransactionDT": 1}' },
    });
    fireEvent.submit(scoringForm());

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "History must be a JSON array.",
    );
    expect(scoreTransaction).not.toHaveBeenCalled();
  });
});

describe("error states", () => {
  it("reports a scoring failure in an assertive live region", async () => {
    vi.mocked(scoreTransaction).mockRejectedValue(new Error("Artifact unavailable"));
    await renderReady();

    fireEvent.submit(scoringForm());

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Artifact unavailable");
    expect(screen.queryByText("Block")).not.toBeInTheDocument();
  });

  it("shows the point-in-time rejection reason returned by a 422", async () => {
    vi.mocked(scoreTransaction).mockRejectedValue(
      new Error(
        "history: Value error, history row 0 has TransactionDT=100 which is not " +
          "strictly less than transaction.TransactionDT=100.",
      ),
    );
    await renderReady();

    fireEvent.submit(scoringForm());

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "strictly less than transaction.TransactionDT=100",
    );
  });

  it("reports an audit history failure without disturbing the rest of the page", async () => {
    vi.mocked(getDecisionHistory).mockRejectedValue(new Error("audit store offline"));

    await renderReady();

    const history = screen.getByRole("region", { name: "Recent decisions" });
    expect(await within(history).findByRole("alert")).toHaveTextContent(
      "audit store offline",
    );
    expect(screen.getByText("Ready")).toBeInTheDocument();
  });
});

describe("audit history", () => {
  it("shows the empty state when nothing has been persisted", async () => {
    await renderReady();

    const history = screen.getByRole("region", { name: "Recent decisions" });
    expect(within(history).getByRole("status")).toHaveTextContent(
      "No persisted decisions yet.",
    );
  });

  it("renders each persisted decision with its outcome, risk and model identity", async () => {
    vi.mocked(getDecisionHistory).mockResolvedValue({
      items: [HISTORY_ROW],
      total: 1,
      limit: 8,
      offset: 0,
    });

    await renderReady();

    const history = screen.getByRole("region", { name: "Recent decisions" });
    expect(await within(history).findByText("RF-TEST-9")).toBeInTheDocument();
    expect(within(history).getByText("ALLOW")).toBeInTheDocument();
    expect(within(history).getByText("3.20%")).toBeInTheDocument();
    expect(within(history).getByText("Audit #42")).toBeInTheDocument();
    expect(
      within(history).getByText("isotonic · policy-test · features-test"),
    ).toBeInTheDocument();
    expect(within(history).getAllByRole("columnheader")).toHaveLength(5);
  });

  it("reloads on demand", async () => {
    await renderReady();

    fireEvent.click(screen.getByRole("button", { name: "Refresh decision history" }));

    await waitFor(() => expect(getDecisionHistory).toHaveBeenCalledTimes(2));
  });
});

describe("review queue", () => {
  it("will not call the API without a reviewer key", async () => {
    await renderReady();

    expect(screen.getByRole("button", { name: "Load pending cases" })).toBeDisabled();

    enterReviewerKey("reviewer-secret");

    const load = screen.getByRole("button", { name: "Load pending cases" });
    expect(load).toBeEnabled();
    fireEvent.click(load);

    await waitFor(() =>
      expect(getReviewQueue).toHaveBeenCalledWith("reviewer-secret"),
    );
  });

  it("reports an unauthorised key from the API", async () => {
    vi.mocked(getReviewQueue).mockRejectedValue(
      new Error("Valid reviewer credentials are required."),
    );
    await renderReady();
    enterReviewerKey("wrong-key");

    fireEvent.click(screen.getByRole("button", { name: "Load pending cases" }));

    const region = reviewRegion();
    expect(await within(region).findByRole("alert")).toHaveTextContent(
      "Valid reviewer credentials are required.",
    );
  });

  it("keeps the automated REVIEW decision visible on each pending case", async () => {
    vi.mocked(getReviewQueue).mockResolvedValue({
      items: [REVIEW_CASE],
      total: 1,
      limit: 20,
      offset: 0,
    });

    await loadReviewQueue();

    const card = screen.getByRole("article", { name: "RF-TEST-4" });
    expect(within(card).getByText("CASE #4 · DECISION #99")).toBeInTheDocument();
    expect(within(card).getByText("9.12%")).toBeInTheDocument();
    expect(within(card).getByText("REVIEW")).toBeInTheDocument();
    expect(within(card).getByText("245.50")).toBeInTheDocument();
    expect(
      within(card).getByText("Amount far above this card's history"),
    ).toBeInTheDocument();
  });

  it("records an approval with the reviewer note and reports it politely", async () => {
    vi.mocked(getReviewQueue).mockResolvedValue({
      items: [REVIEW_CASE],
      total: 1,
      limit: 20,
      offset: 0,
    });
    vi.mocked(resolveReviewCase).mockResolvedValue({
      case: { ...REVIEW_CASE, status: "APPROVED", resolved_by: "reviewer-01" },
    });

    await loadReviewQueue();

    fireEvent.change(screen.getByLabelText(/reviewer notes/i), {
      target: { value: "Card holder confirmed the purchase." },
    });
    fireEvent.click(screen.getByRole("button", { name: /clear transaction/i }));

    await waitFor(() =>
      expect(resolveReviewCase).toHaveBeenCalledWith(
        4,
        "reviewer-secret",
        "APPROVED",
        "Card holder confirmed the purchase.",
      ),
    );

    const region = reviewRegion();
    expect(await within(region).findByRole("status")).toHaveTextContent(
      "RF-TEST-4 recorded as APPROVED. The original model decision remains REVIEW.",
    );
    expect(getReviewQueue).toHaveBeenCalledTimes(2);
  });

  it("records a rejection when the reviewer confirms suspected fraud", async () => {
    vi.mocked(getReviewQueue).mockResolvedValue({
      items: [REVIEW_CASE],
      total: 1,
      limit: 20,
      offset: 0,
    });
    vi.mocked(resolveReviewCase).mockResolvedValue({
      case: { ...REVIEW_CASE, status: "REJECTED" },
    });

    await loadReviewQueue();

    fireEvent.click(screen.getByRole("button", { name: /confirm suspected fraud/i }));

    await waitFor(() =>
      expect(resolveReviewCase).toHaveBeenCalledWith(4, "reviewer-secret", "REJECTED", ""),
    );
  });

  it("surfaces a conflict when the case was already resolved elsewhere", async () => {
    vi.mocked(getReviewQueue).mockResolvedValue({
      items: [REVIEW_CASE],
      total: 1,
      limit: 20,
      offset: 0,
    });
    vi.mocked(resolveReviewCase).mockRejectedValue(
      new Error("The request conflicts with a decision or review case that has already been recorded."),
    );

    await loadReviewQueue();

    fireEvent.click(screen.getByRole("button", { name: /clear transaction/i }));

    const region = reviewRegion();
    expect(await within(region).findByRole("alert")).toHaveTextContent(
      "already been recorded",
    );
  });
});

describe("accessibility", () => {
  it("offers a skip link that targets the focusable main landmark", async () => {
    await renderReady();

    const skip = screen.getByRole("link", { name: "Skip to main content" });
    expect(skip).toHaveAttribute("href", "#overview");

    const main = screen.getByRole("main");
    expect(main).toHaveAttribute("id", "overview");
    expect(main).toHaveAttribute("tabindex", "-1");
  });

  it("announces the decision through a persistent polite live region", async () => {
    vi.mocked(scoreTransaction).mockResolvedValue(SCORE_RESPONSE);
    await renderReady();

    const status = within(resultRegion()).getByRole("status");
    expect(status).toBeEmptyDOMElement();

    fireEvent.submit(scoringForm());

    await waitFor(() =>
      expect(status).toHaveTextContent(
        "Decision Block. Calibrated fraud probability 18.34 percent.",
      ),
    );
    expect(status).toHaveClass("visually-hidden");
  });

  it("names every landmark region after its own heading", async () => {
    await renderReady();

    expect(screen.getByRole("form", { name: "Score a transaction" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Decision result" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Recent decisions" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Manual review queue" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Main navigation" })).toBeInTheDocument();
  });

  it("hides decorative icons in notices from assistive technology", async () => {
    vi.mocked(getDecisionHistory).mockRejectedValue(new Error("audit store offline"));

    await renderReady();

    const history = screen.getByRole("region", { name: "Recent decisions" });
    const alert = await within(history).findByRole("alert");
    expect(alert.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
  });
});
