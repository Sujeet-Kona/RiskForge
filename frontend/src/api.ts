import type { ApiError, HealthResponse, ScoreRequest, ScoreResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

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
    const body = (await response.json().catch(() => ({}))) as ApiError;
    throw new Error(
      body.error?.message ??
        (typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`),
    );
  }

  return (await response.json()) as T;
}

export function getHealth() {
  return request<HealthResponse>("/health");
}

export function scoreTransaction(payload: ScoreRequest) {
  return request<ScoreResponse>("/v1/risk/score", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}