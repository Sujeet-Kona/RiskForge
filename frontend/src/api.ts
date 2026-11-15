import type {
  ApiError,
  HealthResponse,
  ReviewQueueResponse,
  ReviewResolutionResponse,
  ReviewOutcome,
  ScoreRequest,
  ScoreResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

// FastAPI validation errors retain their own detail array instead of the application envelope.
function validationMessage(detail: unknown): string | null {
  if (!Array.isArray(detail)) return null;

  const reasons = detail.flatMap((entry) => {
    if (typeof entry !== "object" || entry === null) return [];
    const { loc, msg } = entry as { loc?: unknown; msg?: unknown };
    if (typeof msg !== "string") return [];
    const field = Array.isArray(loc) ? loc[loc.length - 1] : undefined;
    return [typeof field === "string" ? `${field}: ${msg}` : msg];
  });

  return reasons.length > 0 ? reasons.join("; ") : null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ApiError | null;
    const detail = body?.detail;
    throw new Error(
      body?.error?.message ??
        (typeof detail === "string" ? detail : validationMessage(detail)) ??
        `Request failed (${response.status})`,
    );
  }

  return (await response.json()) as T;
}

export function getHealth() {
  return request<HealthResponse>("/health");
}

export function getDecisionHistory(limit = 8) {
  return request<import("./types").AuditDecisionListResponse>(
    `/v1/audit/decisions?limit=${limit}&offset=0`,
  );
}

export function scoreTransaction(payload: ScoreRequest, idempotencyKey: string) {
  return request<ScoreResponse>("/v1/risk/score", {
    method: "POST",
    headers: { "Idempotency-Key": idempotencyKey },
    body: JSON.stringify(payload),
  });
}

export function getReviewQueue(apiKey: string) {
  return request<ReviewQueueResponse>("/v1/reviews?status=PENDING", {
    headers: { "X-Reviewer-API-Key": apiKey },
  });
}

export function resolveReviewCase(
  caseId: number,
  apiKey: string,
  outcome: ReviewOutcome,
  comment: string,
) {
  return request<ReviewResolutionResponse>(`/v1/reviews/${caseId}/resolve`, {
    method: "POST",
    headers: { "X-Reviewer-API-Key": apiKey },
    body: JSON.stringify({ outcome, comment: comment || null }),
  });
}
