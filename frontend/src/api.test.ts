import { describe, expect, it, vi } from "vitest";
import {
  getDecisionHistory,
  getHealth,
  getReviewQueue,
  resolveReviewCase,
  scoreTransaction,
} from "./api";
import type { HealthResponse, ScoreRequest } from "./types";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubFetch(response: Response) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue(response);
}

function callOf(fetchMock: ReturnType<typeof stubFetch>, index = 0) {
  const [url, init] = fetchMock.mock.calls[index] as [string, RequestInit];
  return { url, init, headers: init.headers as Record<string, string> };
}

const HEALTH: HealthResponse = {
  status: "ok",
  model_version: "model-test",
  feature_version: "features-test",
  calibration_method: "isotonic",
  policy_version: "policy-test",
  database: "ready",
  artifact: { n_features: 28 },
};

const SCORE_REQUEST: ScoreRequest = {
  transaction: {
    TransactionID: "RF-TEST-1",
    TransactionDT: 100,
    TransactionAmt: 12.5,
    card1: 13926,
    addr1: null,
    addr2: null,
    ProductCD: "W",
    card6: "debit",
    P_emaildomain: null,
    R_emaildomain: null,
  },
  history: [{ TransactionDT: 10, TransactionAmt: 4.2 }],
  top_k: 3,
};

describe("request wiring", () => {
  it("reads health from the default API base", async () => {
    const fetchMock = stubFetch(jsonResponse(200, HEALTH));

    await expect(getHealth()).resolves.toEqual(HEALTH);

    const { url, headers } = callOf(fetchMock);
    expect(url).toBe("/api/health");
    expect(headers.Accept).toBe("application/json");
  });

  it("pages the audit history with the documented default limit", async () => {
    const fetchMock = stubFetch(
      jsonResponse(200, { items: [], total: 0, limit: 8, offset: 0 }),
    );

    await getDecisionHistory();

    expect(callOf(fetchMock).url).toBe("/api/v1/audit/decisions?limit=8&offset=0");
  });

  it("sends the idempotency key and a JSON content type when scoring", async () => {
    const fetchMock = stubFetch(jsonResponse(200, {}));

    await scoreTransaction(SCORE_REQUEST, "key-abc");

    const { url, init, headers } = callOf(fetchMock);
    expect(url).toBe("/api/v1/risk/score");
    expect(init.method).toBe("POST");
    expect(headers["Idempotency-Key"]).toBe("key-abc");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body as string)).toEqual(SCORE_REQUEST);
  });

  it("sends the reviewer key as a header rather than in the query string", async () => {
    const fetchMock = stubFetch(
      jsonResponse(200, { items: [], total: 0, limit: 20, offset: 0 }),
    );

    await getReviewQueue("reviewer-secret");

    const { url, headers } = callOf(fetchMock);
    expect(url).toBe("/api/v1/reviews?status=PENDING");
    expect(headers["X-Reviewer-API-Key"]).toBe("reviewer-secret");
    expect(url).not.toContain("reviewer-secret");
  });

  it("turns an empty reviewer note into an explicit null", async () => {
    const fetchMock = stubFetch(jsonResponse(200, { case: {} }));

    await resolveReviewCase(7, "reviewer-secret", "REJECTED", "");

    const { url, init, headers } = callOf(fetchMock);
    expect(url).toBe("/api/v1/reviews/7/resolve");
    expect(init.method).toBe("POST");
    expect(headers["X-Reviewer-API-Key"]).toBe("reviewer-secret");
    expect(JSON.parse(init.body as string)).toEqual({
      outcome: "REJECTED",
      comment: null,
    });
  });

  it("passes a non-empty reviewer note through unchanged", async () => {
    const fetchMock = stubFetch(jsonResponse(200, { case: {} }));

    await resolveReviewCase(7, "reviewer-secret", "APPROVED", "Card holder confirmed.");

    expect(JSON.parse(callOf(fetchMock).init.body as string)).toEqual({
      outcome: "APPROVED",
      comment: "Card holder confirmed.",
    });
  });
});

describe("error reporting", () => {
  it("prefers the application error envelope", async () => {
    stubFetch(
      jsonResponse(401, {
        error: {
          code: "unauthenticated",
          message: "Valid reviewer credentials are required.",
        },
      }),
    );

    await expect(getReviewQueue("wrong")).rejects.toThrow(
      "Valid reviewer credentials are required.",
    );
  });

  it("reports a 409 conflict from the envelope, not the status code", async () => {
    stubFetch(
      jsonResponse(409, {
        error: {
          code: "conflict",
          message:
            "The request conflicts with a decision or review case that has already been recorded.",
        },
      }),
    );

    await expect(resolveReviewCase(1, "k", "APPROVED", "")).rejects.toThrow(
      /already been recorded/,
    );
  });

  it("surfaces a plain string detail", async () => {
    stubFetch(jsonResponse(405, { detail: "Method Not Allowed" }));

    await expect(getHealth()).rejects.toThrow("Method Not Allowed");
  });

  it("surfaces the field and reason from a 422 validation body", async () => {
    stubFetch(
      jsonResponse(422, {
        detail: [
          {
            type: "value_error",
            loc: ["body", "history", 0, "TransactionDT"],
            msg: "Value error, history row 0 has TransactionDT=100 which is not strictly less than transaction.TransactionDT=100.",
            input: { TransactionDT: 100 },
          },
        ],
      }),
    );

    await expect(scoreTransaction(SCORE_REQUEST, "k")).rejects.toThrow(
      "TransactionDT: Value error, history row 0 has TransactionDT=100 which is not strictly less than transaction.TransactionDT=100.",
    );
  });

  it("joins every validation failure rather than showing only the first", async () => {
    stubFetch(
      jsonResponse(422, {
        detail: [
          { loc: ["body", "transaction", "TransactionAmt"], msg: "Input should be a valid number" },
          { loc: ["body", "top_k"], msg: "Input should be a valid integer" },
        ],
      }),
    );

    await expect(scoreTransaction(SCORE_REQUEST, "k")).rejects.toThrow(
      "TransactionAmt: Input should be a valid number; top_k: Input should be a valid integer",
    );
  });

  it("falls back to the status code when a body carries no usable message", async () => {
    stubFetch(jsonResponse(422, { detail: [] }));

    await expect(getHealth()).rejects.toThrow("Request failed (422)");
  });

  it("falls back to the status code when the body is not JSON at all", async () => {
    stubFetch(
      new Response("<html><body>502 Bad Gateway</body></html>", {
        status: 502,
        headers: { "Content-Type": "text/html" },
      }),
    );

    await expect(getHealth()).rejects.toThrow("Request failed (502)");
  });

  it("lets a transport failure propagate untouched", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(getHealth()).rejects.toThrow("Failed to fetch");
  });
});
