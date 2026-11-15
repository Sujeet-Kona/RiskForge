import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Vitest globals are deliberately off, so Testing Library cannot register its
// own auto-cleanup (it looks for a global afterEach). Without this, every test
// would render into the previous test's DOM.
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
