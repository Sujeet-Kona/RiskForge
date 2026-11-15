import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import App from "./App";
import {
  getDecisionHistory,
  getHealth,
  getReviewQueue,
  resolveReviewCase,
  scoreTransaction,
} from "./api";
import type {
  AuditDecisionListResponse,
  AuditDecisionSummary,
  HealthResponse,
  ReasonCode,
  ReviewCase,
  ReviewQueueResponse,
  ReviewResolutionResponse,
  ScoreResponse,
} from "./types";

// All fixtures and credentials are synthetic; the entire API is mocked.
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
const PENDING_QUEUE = { ...EMPTY_QUEUE, items: [REVIEW_CASE], total: 1 };

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

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
  vi.resetAllMocks();
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Unexpected fetch in UI test"));
  vi.mocked(getHealth).mockResolvedValue(HEALTHY);
  vi.mocked(getDecisionHistory).mockResolvedValue(EMPTY_HISTORY);
  vi.mocked(getReviewQueue).mockResolvedValue(EMPTY_QUEUE);
});

describe("system status", () => {
  it.each([
    { missing: "artifact", artifact: undefined },
    { missing: "n_features", artifact: {} },
  ])("shows an unknown feature contract when $missing is missing", async ({ artifact }) => {
    vi.mocked(getHealth).mockResolvedValue({ ...HEALTHY, artifact });

    await renderReady();

    const card = screen.getByText("FEATURE CONTRACT").closest(".metric-card");
    expect(within(card as HTMLElement).getByText("—")).toBeInTheDocument();
    expect(within(card as HTMLElement).queryByText("28")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run risk assessment/i })).toBeEnabled();
  });

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
  it.each([
    { outcome: "ALLOW", label: "Allow", probability: 0.032, percent: "3.20", explanation: "Synthetic probability is below the review threshold." },
    { outcome: "REVIEW", label: "Review", probability: 0.0912, percent: "9.12", explanation: "Synthetic probability is between the review and block thresholds." },
    { outcome: "BLOCK", label: "Block", probability: 0.1834, percent: "18.34", explanation: "Synthetic probability is above the block threshold." },
  ] as const)("renders $outcome with its calibrated probability and announcement", async ({ outcome, label, probability, percent, explanation }) => {
    vi.mocked(scoreTransaction).mockResolvedValue({
      ...SCORE_RESPONSE,
      decision: outcome,
      risk_probability: probability,
      decision_explanation: explanation,
      decision_reason_code: {
        code: `synthetic_${outcome.toLowerCase()}`,
        title: label,
        type: "policy",
        description: explanation,
      },
    });
    await renderReady();
    const region = resultRegion();
    const status = within(region).getByRole("status");

    fireEvent.submit(scoringForm());

    const decision = await within(region).findByText(label);
    expect(decision.closest(".decision-banner")).toHaveClass(outcome.toLowerCase());
    expect(within(region).getByText(`${percent}%`)).toBeInTheDocument();
    expect(within(region).getByText("Review 7.5%")).toBeInTheDocument();
    expect(within(region).getByText("Block 11.0%")).toBeInTheDocument();
    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toHaveTextContent(`Decision ${label}. Calibrated fraud probability ${percent} percent. ${explanation}`);
  });

  it("renders empty signals without retaining a previous assessment's signals", async () => {
    vi.mocked(scoreTransaction)
      .mockResolvedValueOnce(SCORE_RESPONSE)
      .mockResolvedValueOnce({
        ...SCORE_RESPONSE,
        reason_codes: [],
        risk_decreasing_signals: [],
        model_detail: {
          ...SCORE_RESPONSE.model_detail,
          total_risk_increasing: 0,
          total_risk_decreasing: 0,
        },
      });
    await renderReady();
    fireEvent.submit(scoringForm());
    await within(resultRegion()).findByText(RISK_INCREASING.title);
    expect(within(resultRegion()).getByText(RISK_DECREASING.title)).toBeInTheDocument();

    fireEvent.submit(scoringForm());

    expect(await within(resultRegion()).findByText("No risk-increasing signals returned.")).toBeInTheDocument();
    expect(within(resultRegion()).queryByText(RISK_INCREASING.title)).not.toBeInTheDocument();
    expect(within(resultRegion()).queryByText(RISK_DECREASING.title)).not.toBeInTheDocument();
    expect(within(resultRegion()).queryByRole("heading", { name: "Risk-reducing signals" })).not.toBeInTheDocument();
    expect(within(resultRegion()).getByText("Block")).toBeInTheDocument();
    expect(within(resultRegion()).getByText("18.34%")).toBeInTheDocument();
  });

  it("keeps separate features with the same reason code stable across result updates", async () => {
    const consoleError = vi.spyOn(console, "error");
    const secondSignal = {
      ...RISK_INCREASING,
      title: "Synthetic second amount signal",
      feature: "amount_log1p",
    };
    vi.mocked(scoreTransaction)
      .mockResolvedValueOnce({
        ...SCORE_RESPONSE,
        reason_codes: [RISK_INCREASING, secondSignal],
      })
      .mockResolvedValueOnce({
        ...SCORE_RESPONSE,
        reason_codes: [secondSignal],
      });
    await renderReady();
    fireEvent.submit(scoringForm());
    await within(resultRegion()).findByText(secondSignal.title);
    expect(within(resultRegion()).getAllByText("0.412")).toHaveLength(2);

    fireEvent.submit(scoringForm());
    await within(resultRegion()).findByText(secondSignal.title);

    expect(within(resultRegion()).queryByText(RISK_INCREASING.title)).not.toBeInTheDocument();
    expect(within(resultRegion()).getAllByText("0.412")).toHaveLength(1);
    expect(consoleError).not.toHaveBeenCalled();
  });

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

  it("rotates the idempotency key when only prior history changes", async () => {
    vi.mocked(scoreTransaction).mockResolvedValue(SCORE_RESPONSE);
    await renderReady();
    fireEvent.submit(scoringForm());
    await within(resultRegion()).findByText("Block");
    const [firstPayload, firstKey] = vi.mocked(scoreTransaction).mock.calls[0];
    const syntheticHistory = [{ TransactionDT: 10, TransactionAmt: 4.2 }];

    fireEvent.change(screen.getByLabelText(/prior transaction history/i), {
      target: { value: JSON.stringify(syntheticHistory) },
    });
    expect(screen.getByText("1 records")).toBeInTheDocument();
    fireEvent.submit(scoringForm());
    await within(resultRegion()).findByText("Block");

    expect(scoreTransaction).toHaveBeenCalledTimes(2);
    const [secondPayload, secondKey] = vi.mocked(scoreTransaction).mock.calls[1];
    expect(firstPayload.history).toEqual([]);
    expect(secondPayload).toEqual({ ...firstPayload, history: syntheticHistory });
    expect(firstKey).toEqual(expect.any(String));
    expect(firstKey).not.toBe("");
    expect(secondKey).toEqual(expect.any(String));
    expect(secondKey).not.toBe("");
    expect(secondKey).not.toBe(firstKey);
  });

  it("reuses the same payload and idempotency key after a network failure", async () => {
    vi.mocked(scoreTransaction)
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(SCORE_RESPONSE);
    await renderReady();
    fireEvent.submit(scoringForm());
    expect(await within(scoringForm()).findByRole("alert")).toHaveTextContent("Failed to fetch");
    expect(screen.getByRole("button", { name: /run risk assessment/i })).toBeEnabled();
    expect(getDecisionHistory).toHaveBeenCalledTimes(1);

    fireEvent.submit(scoringForm());
    await within(resultRegion()).findByText("Block");

    expect(scoreTransaction).toHaveBeenCalledTimes(2);
    const [first, retry] = vi.mocked(scoreTransaction).mock.calls;
    expect(first[1]).toEqual(expect.any(String));
    expect(first[1]).not.toBe("");
    expect(retry).toEqual(first);
    expect(within(scoringForm()).queryByRole("alert")).not.toBeInTheDocument();
    expect(getDecisionHistory).toHaveBeenCalledTimes(2);
  });

  it("rejects invalid history JSON locally without making a scoring request", async () => {
    await renderReady();
    fireEvent.change(screen.getByLabelText(/prior transaction history/i), {
      target: { value: '[{"TransactionDT": 1}' },
    });

    fireEvent.submit(scoringForm());

    expect(await within(scoringForm()).findByRole("alert")).toHaveTextContent(/JSON/i);
    expect(scoreTransaction).not.toHaveBeenCalled();
    expect(within(resultRegion()).getByText("Awaiting assessment")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run risk assessment/i })).toBeEnabled();
  });

  it("treats blank history as an empty array", async () => {
    vi.mocked(scoreTransaction).mockResolvedValue({ ...SCORE_RESPONSE, history_transactions_used: 0 });
    await renderReady();
    fireEvent.change(screen.getByLabelText(/prior transaction history/i), {
      target: { value: " \n\t " },
    });

    fireEvent.submit(scoringForm());
    await within(resultRegion()).findByText("Block");

    expect(scoreTransaction).toHaveBeenCalledTimes(1);
    expect(scoreTransaction).toHaveBeenCalledWith(
      expect.objectContaining({ history: [] }),
      expect.any(String),
    );
    expect(screen.getByText("0 records")).toBeInTheDocument();
    expect(within(scoringForm()).queryByRole("alert")).not.toBeInTheDocument();
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
  it("keeps its status mounted through loading, failure and a successful retry without claiming an empty audit", async () => {
    const pending = deferred<AuditDecisionListResponse>();
    vi.mocked(getDecisionHistory).mockReturnValueOnce(pending.promise);
    await renderReady();
    const region = screen.getByRole("region", { name: "Recent decisions" });
    const status = within(region).getByRole("status");
    const refresh = within(region).getByRole("button", { name: "Refresh decision history" });
    expect(status).toHaveTextContent("Loading recorded decisions…");
    expect(refresh).toBeDisabled();
    expect(refresh).toHaveAttribute("aria-busy", "true");
    expect(within(region).queryByText(/No persisted decisions yet/)).not.toBeInTheDocument();
    fireEvent.click(refresh);
    expect(getDecisionHistory).toHaveBeenCalledTimes(1);

    await act(async () => { pending.reject(new Error("Synthetic audit unavailable")); });

    expect(within(region).getByRole("alert")).toHaveTextContent("Synthetic audit unavailable");
    expect(within(region).getByText("Decision history could not be loaded.")).toBeInTheDocument();
    expect(within(region).queryByText(/No persisted decisions yet/)).not.toBeInTheDocument();
    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toBeEmptyDOMElement();
    expect(refresh).toBeEnabled();
    expect(refresh).toHaveAttribute("aria-busy", "false");

    const retry = deferred<AuditDecisionListResponse>();
    vi.mocked(getDecisionHistory).mockReturnValueOnce(retry.promise);
    fireEvent.click(refresh);
    expect(status).toHaveTextContent("Loading recorded decisions…");
    await act(async () => {
      retry.resolve({ ...EMPTY_HISTORY, items: [HISTORY_ROW], total: 1 });
    });

    expect(getDecisionHistory).toHaveBeenCalledTimes(2);
    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toHaveTextContent("1 recent decisions loaded.");
    expect(within(region).getByText(HISTORY_ROW.transaction_id)).toBeInTheDocument();
    expect(within(region).queryByRole("alert")).not.toBeInTheDocument();
    expect(refresh).toBeEnabled();
  });

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
  it("distinguishes an authenticated empty queue from a queue that has not been loaded", async () => {
    await renderReady();
    const region = reviewRegion();
    const status = within(region).getByRole("status");
    expect(status).toBeEmptyDOMElement();
    expect(within(region).getByText("No pending cases loaded. Authenticate to retrieve the review queue.")).toBeInTheDocument();
    expect(within(region).queryByText("No pending review cases.")).not.toBeInTheDocument();
    expect(getReviewQueue).not.toHaveBeenCalled();

    enterReviewerKey("reviewer-secret");
    expect(status).toBeEmptyDOMElement();
    fireEvent.click(within(region).getByRole("button", { name: "Load pending cases" }));
    await waitFor(() => expect(status).toHaveTextContent("0 pending review cases loaded."));

    expect(within(region).getByRole("status")).toBe(status);
    expect(within(region).getByText("No pending review cases.")).toBeInTheDocument();
    expect(within(region).queryByText(/Authenticate to retrieve/)).not.toBeInTheDocument();
    expect(within(region).queryByRole("alert")).not.toBeInTheDocument();
    expect(getReviewQueue).toHaveBeenCalledExactlyOnceWith("reviewer-secret");

    enterReviewerKey("synthetic-other-key");
    expect(status).toBeEmptyDOMElement();
    expect(within(region).queryByText("No pending review cases.")).not.toBeInTheDocument();
    expect(within(region).getByText(/Authenticate to retrieve/)).toBeInTheDocument();
    expect(getReviewQueue).toHaveBeenCalledTimes(1);
  });

  it("clears old cases and disables credentials and duplicate loads while the queue is loading", async () => {
    const pending = deferred<ReviewQueueResponse>();
    vi.mocked(getReviewQueue)
      .mockResolvedValueOnce(PENDING_QUEUE)
      .mockReturnValueOnce(pending.promise);
    await loadReviewQueue();
    const region = reviewRegion();
    const status = within(region).getByRole("status");
    const key = within(region).getByLabelText(/reviewer api key/i);
    expect(status).toHaveTextContent("1 pending review cases loaded.");

    fireEvent.click(within(region).getByRole("button", { name: "Load pending cases" }));

    const load = within(region).getByRole("button", { name: "Loading…" });
    expect(load).toBeDisabled();
    expect(load).toHaveAttribute("aria-busy", "true");
    expect(key).toBeDisabled();
    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toHaveTextContent("Loading review cases…");
    expect(within(region).queryByRole("article")).not.toBeInTheDocument();
    expect(within(region).queryByText("No pending review cases.")).not.toBeInTheDocument();
    expect(within(region).queryByText(/Authenticate to retrieve/)).not.toBeInTheDocument();
    fireEvent.click(load);
    expect(getReviewQueue).toHaveBeenCalledTimes(2);

    await act(async () => { pending.resolve(PENDING_QUEUE); });

    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toHaveTextContent("1 pending review cases loaded.");
    expect(within(region).getByRole("article", { name: REVIEW_CASE.transaction_id })).toBeInTheDocument();
    expect(key).toBeEnabled();
    expect(key).toHaveValue("reviewer-secret");
    expect(load).toHaveTextContent("Load pending cases");
    expect(load).toBeEnabled();
    expect(load).toHaveAttribute("aria-busy", "false");
  });

  it("locks every resolution button and credential control until resolution and both refreshes finish", async () => {
    const secondCase: ReviewCase = { ...REVIEW_CASE, case_id: 5, decision_id: 100, transaction_id: "RF-TEST-5" };
    const resolution = deferred<ReviewResolutionResponse>();
    const queueRefresh = deferred<ReviewQueueResponse>();
    const historyRefresh = deferred<AuditDecisionListResponse>();
    vi.mocked(getReviewQueue)
      .mockResolvedValueOnce({ ...EMPTY_QUEUE, items: [REVIEW_CASE, secondCase], total: 2 })
      .mockReturnValueOnce(queueRefresh.promise);
    vi.mocked(getDecisionHistory)
      .mockResolvedValueOnce(EMPTY_HISTORY)
      .mockReturnValueOnce(historyRefresh.promise);
    vi.mocked(resolveReviewCase).mockReturnValueOnce(resolution.promise);
    await loadReviewQueue();
    const region = reviewRegion();
    const status = within(region).getByRole("status");
    const key = within(region).getByLabelText(/reviewer api key/i);
    const load = within(region).getByRole("button", { name: "Load pending cases" });
    const actions = within(region).getAllByRole("button", { name: /clear transaction|confirm suspected fraud/i });
    expect(actions).toHaveLength(4);

    fireEvent.click(actions[0]);

    expect(status).toHaveTextContent("Recording review outcome…");
    for (const button of actions) expect(button).toBeDisabled();
    expect(actions[0]).toHaveAttribute("aria-busy", "true");
    expect(actions[1]).toHaveAttribute("aria-busy", "true");
    expect(actions[2]).toHaveAttribute("aria-busy", "false");
    expect(key).toBeDisabled();
    expect(load).toBeDisabled();
    fireEvent.click(actions[1]);
    fireEvent.click(actions[3]);
    fireEvent.click(load);
    expect(resolveReviewCase).toHaveBeenCalledExactlyOnceWith(4, "reviewer-secret", "APPROVED", "");
    expect(getReviewQueue).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolution.resolve({ case: { ...REVIEW_CASE, status: "APPROVED" } });
    });

    expect(getReviewQueue).toHaveBeenCalledTimes(2);
    expect(status).toHaveTextContent("Loading review cases…");
    expect(key).toBeDisabled();
    expect(load).toBeDisabled();
    expect(within(region).queryByRole("article")).not.toBeInTheDocument();
    expect(getDecisionHistory).toHaveBeenCalledTimes(1);

    await act(async () => {
      queueRefresh.resolve({ ...EMPTY_QUEUE, items: [secondCase], total: 1 });
    });

    expect(getDecisionHistory).toHaveBeenCalledTimes(2);
    expect(status).toHaveTextContent("Recording review outcome…");
    expect(key).toBeDisabled();
    expect(load).toBeDisabled();
    const remainingActions = within(region).getAllByRole("button", { name: /clear transaction|confirm suspected fraud/i });
    expect(remainingActions).toHaveLength(2);
    for (const button of remainingActions) expect(button).toBeDisabled();

    await act(async () => { historyRefresh.resolve(EMPTY_HISTORY); });

    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toHaveTextContent("RF-TEST-4 recorded as APPROVED. The original model decision remains REVIEW.");
    expect(key).toBeEnabled();
    expect(load).toBeEnabled();
    for (const button of remainingActions) expect(button).toBeEnabled();
    expect(within(region).queryByRole("article", { name: REVIEW_CASE.transaction_id })).not.toBeInTheDocument();
    expect(within(region).queryByRole("alert")).not.toBeInTheDocument();
  });

  it("preserves resolution success alongside a queue refresh failure", async () => {
    vi.mocked(getReviewQueue)
      .mockResolvedValueOnce(PENDING_QUEUE)
      .mockRejectedValueOnce(new Error("Synthetic queue refresh unavailable"));
    vi.mocked(resolveReviewCase).mockResolvedValueOnce({ case: { ...REVIEW_CASE, status: "APPROVED" } });
    await loadReviewQueue();
    const region = reviewRegion();
    const status = within(region).getByRole("status");

    fireEvent.click(within(region).getByRole("button", { name: /clear transaction/i }));
    await waitFor(() => expect(status).toHaveTextContent("RF-TEST-4 recorded as APPROVED. The original model decision remains REVIEW."));

    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toHaveClass("success");
    expect(within(region).getByRole("alert")).toHaveTextContent("Synthetic queue refresh unavailable");
    expect(within(region).getByText("Review queue could not be loaded.")).toBeInTheDocument();
    expect(within(region).queryByText("No pending review cases.")).not.toBeInTheDocument();
    expect(within(region).queryByRole("article")).not.toBeInTheDocument();
    expect(within(region).getByLabelText(/reviewer api key/i)).toBeEnabled();
    expect(within(region).getByRole("button", { name: "Load pending cases" })).toBeEnabled();
    expect(resolveReviewCase).toHaveBeenCalledExactlyOnceWith(4, "reviewer-secret", "APPROVED", "");
    expect(getReviewQueue).toHaveBeenCalledTimes(2);
    expect(getDecisionHistory).toHaveBeenCalledTimes(2);
  });

  it("clears prior cases and notes when the reviewer key changes", async () => {
    vi.mocked(getReviewQueue).mockResolvedValue(PENDING_QUEUE);
    vi.mocked(resolveReviewCase).mockResolvedValueOnce({ case: { ...REVIEW_CASE, status: "REJECTED" } });
    await loadReviewQueue();
    const region = reviewRegion();
    const status = within(region).getByRole("status");
    fireEvent.change(within(region).getByLabelText(/reviewer notes/i), {
      target: { value: "Synthetic private note for the first reviewer." },
    });

    enterReviewerKey("synthetic-second-reviewer");

    expect(within(region).queryByRole("article")).not.toBeInTheDocument();
    expect(within(region).queryByLabelText(/reviewer notes/i)).not.toBeInTheDocument();
    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toBeEmptyDOMElement();
    expect(within(region).getByText(/Authenticate to retrieve/)).toBeInTheDocument();
    expect(getReviewQueue).toHaveBeenCalledTimes(1);
    fireEvent.click(within(region).getByRole("button", { name: "Load pending cases" }));
    await within(region).findByRole("article", { name: REVIEW_CASE.transaction_id });

    expect(getReviewQueue).toHaveBeenNthCalledWith(2, "synthetic-second-reviewer");
    expect(within(region).getByLabelText(/reviewer notes/i)).toHaveValue("");
    vi.mocked(getReviewQueue).mockResolvedValueOnce(EMPTY_QUEUE);
    fireEvent.click(within(region).getByRole("button", { name: /confirm suspected fraud/i }));
    await waitFor(() => expect(status).toHaveTextContent("RF-TEST-4 recorded as REJECTED."));
    expect(resolveReviewCase).toHaveBeenCalledExactlyOnceWith(4, "synthetic-second-reviewer", "REJECTED", "");
  });

  it("clears prior success and error notices when the reviewer key is removed", async () => {
    vi.mocked(getReviewQueue)
      .mockResolvedValueOnce(PENDING_QUEUE)
      .mockRejectedValueOnce(new Error("Synthetic queue refresh unavailable"));
    vi.mocked(resolveReviewCase).mockResolvedValueOnce({ case: { ...REVIEW_CASE, status: "APPROVED" } });
    await loadReviewQueue();
    const region = reviewRegion();
    const status = within(region).getByRole("status");
    fireEvent.click(within(region).getByRole("button", { name: /clear transaction/i }));
    await waitFor(() => expect(status).toHaveTextContent("RF-TEST-4 recorded as APPROVED."));
    expect(within(region).getByRole("alert")).toHaveTextContent("Synthetic queue refresh unavailable");

    enterReviewerKey("");

    expect(within(region).getByRole("status")).toBe(status);
    expect(status).toBeEmptyDOMElement();
    expect(status).not.toHaveClass("success");
    expect(within(region).queryByRole("alert")).not.toBeInTheDocument();
    expect(within(region).queryByText("Review queue could not be loaded.")).not.toBeInTheDocument();
    expect(within(region).getByText(/Authenticate to retrieve/)).toBeInTheDocument();
    expect(within(region).getByRole("button", { name: "Load pending cases" })).toBeDisabled();
    expect(getReviewQueue).toHaveBeenCalledTimes(2);
  });

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
