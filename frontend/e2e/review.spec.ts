import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { test as base, expect, type Page } from "@playwright/test";

const root = fileURLToPath(new URL("../..", import.meta.url));
const key = process.env.RISKFORGE_REVIEWER_API_KEY!;
const reviewer = process.env.RISKFORGE_REVIEWER_ID!;

interface Snapshot {
  total: number;
  cases: Array<{
    id: number;
    status: string;
    resolved_by: string | null;
    resolution_comment: string | null;
    resolved_at: string | null;
    decision: { transaction_id: string; [key: string]: unknown };
  }>;
  events: Array<{ review_case_id: number; action: string; reviewer_id: string; comment: string | null }>;
}

function fixture(action: "reset" | "seed" | "snapshot") {
  return execFileSync("python", ["scripts/e2e_review_fixture.py", action], {
    cwd: root,
    env: process.env,
    encoding: "utf8",
  });
}

function snapshot(): Snapshot {
  return JSON.parse(fixture("snapshot"));
}

const test = base.extend<{ seeded: Snapshot }>({
  seeded: [async ({ page }, use) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    fixture("reset");
    try {
      fixture("seed");
      await use(snapshot());
      expect(pageErrors).toEqual([]);
    } finally {
      fixture("reset");
    }
  }, { auto: true }],
});

async function authenticate(page: Page) {
  await page.goto("/");
  await page.getByLabel("Reviewer API key held in memory only").fill(key);
  await page.getByRole("button", { name: "Load pending cases" }).click();
  await expect(page.getByRole("status").filter({ hasText: "2 pending review cases loaded." })).toBeVisible();
}

function card(page: Page, name = "E2E-APPROVE") {
  return page.getByRole("article", { name, exact: true });
}

function assertResolution(before: Snapshot, transaction: string, outcome: string, comment: string | null) {
  const after = snapshot();
  const original = before.cases.find((item) => item.decision.transaction_id === transaction)!;
  const resolved = after.cases.find((item) => item.id === original.id)!;
  expect(resolved.decision).toEqual(original.decision);
  expect(resolved.status).toBe(outcome);
  expect(resolved.resolved_by).toBe(reviewer);
  expect(resolved.resolution_comment).toBe(comment);
  expect(resolved.resolved_at).not.toBeNull();
  expect(after.events).toEqual([{
    review_case_id: original.id, action: outcome, reviewer_id: reviewer, comment,
  }]);
  expect(after.cases.filter((item) => item.status === "PENDING")).toHaveLength(1);
}

test("unauthenticated queue access is blocked", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Load pending cases" })).toBeDisabled();
  await expect(page.getByText("No pending cases loaded. Authenticate to retrieve the review queue.")).toBeVisible();
  expect((await request.get("/api/v1/reviews")).status()).toBe(401);
  expect((await request.post("/api/v1/reviews/1/resolve", { data: { outcome: "APPROVED" } })).status()).toBe(401);
  await expect(page.getByRole("region", { name: "Manual review queue" }).getByRole("article")).toHaveCount(0);
});

test("wrong reviewer key is rejected", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Reviewer API key held in memory only").fill("deliberately-invalid-e2e-key");
  const response = page.waitForResponse((r) => r.url().includes("/v1/reviews"));
  await page.getByRole("button", { name: "Load pending cases" }).click();
  expect((await response).status()).toBe(401);
  await expect(page.getByRole("region", { name: "Manual review queue" }).getByRole("alert")).toHaveText("Valid reviewer credentials are required.");
  await expect(page.getByRole("region", { name: "Manual review queue" }).getByRole("article")).toHaveCount(0);
});

test("authenticated queue displays details and forgets credentials on reload", async ({ page }) => {
  await authenticate(page);
  await expect(page.getByRole("region", { name: "Manual review queue" }).getByRole("article")).toHaveCount(2);
  const item = card(page);
  await expect(item).toContainText("CASE #1 · DECISION #1");
  await expect(item).toContainText("Automated outcome REVIEW");
  await expect(item).toContainText("9.00%");
  await expect(item).toContainText("e2e-synthetic · isotonic · policy-001");
  await expect(item).toContainText("behavioral-v1 · W");
  await expect(item).toContainText("117.50");
  await expect(item).toContainText("Synthetic fixture: 0.09 is in the REVIEW band.");
  await expect(item).toContainText("Synthetic card velocity signal");
  const stored = await page.evaluate(() => ({ local: localStorage.length, session: sessionStorage.length }));
  expect(stored).toEqual({ local: 0, session: 0 });
  await page.reload();
  await expect(page.getByLabel("Reviewer API key held in memory only")).toBeEmpty();
  await expect(page.getByRole("button", { name: "Load pending cases" })).toBeDisabled();
  await expect(page.getByRole("region", { name: "Manual review queue" }).getByRole("article")).toHaveCount(0);
});

test("approve persists one event without changing the model decision", async ({ page, seeded }) => {
  await authenticate(page);
  await card(page).getByLabel("Reviewer notes optional").fill("Disposable E2E approval evidence");
  await card(page).getByRole("button", { name: "Clear transaction" }).click();
  await expect(page.getByRole("status").filter({ hasText: "E2E-APPROVE recorded as APPROVED. The original model decision remains REVIEW." })).toBeVisible();
  await expect(card(page)).toHaveCount(0);
  assertResolution(seeded, "E2E-APPROVE", "APPROVED", "Disposable E2E approval evidence");
});

test("reject persists a null optional comment and updates the queue", async ({ page, seeded }) => {
  await authenticate(page);
  await card(page, "E2E-REJECT").getByRole("button", { name: "Confirm suspected fraud" }).click();
  await expect(page.getByRole("status").filter({ hasText: "E2E-REJECT recorded as REJECTED. The original model decision remains REVIEW." })).toBeVisible();
  await expect(card(page, "E2E-REJECT")).toHaveCount(0);
  assertResolution(seeded, "E2E-REJECT", "REJECTED", null);
});

test("a stale second browser gets 409 and cannot resolve a case twice", async ({ page, browser, seeded }) => {
  const secondContext = await browser.newContext({ baseURL: "http://127.0.0.1:5174" });
  try {
    const second = await secondContext.newPage();
    await authenticate(page);
    await authenticate(second);
    await card(page).getByRole("button", { name: "Clear transaction" }).click();
    await expect(card(page)).toHaveCount(0);
    const response = second.waitForResponse((r) => r.url().includes("/resolve"));
    await card(second).getByRole("button", { name: "Confirm suspected fraud" }).click();
    expect((await response).status()).toBe(409);
    await expect(second.getByRole("region", { name: "Manual review queue" }).getByRole("alert")).toHaveText("The request conflicts with a decision or review case that has already been recorded.");
    assertResolution(seeded, "E2E-APPROVE", "APPROVED", null);
  } finally {
    await secondContext.close();
  }
});
