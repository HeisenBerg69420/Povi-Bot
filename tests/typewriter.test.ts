import { afterEach, describe, expect, it, vi } from "vitest";
import { typeText } from "../src/typewriter";

afterEach(() => {
  vi.useRealTimers();
});

describe("typeText", () => {
  it("writes text one character at a time", async () => {
    vi.useFakeTimers();
    const element = { textContent: "" };
    const typing = typeText(element, "Povi", { delayMs: 20 });

    expect(element.textContent).toBe("P");
    await vi.advanceTimersByTimeAsync(20);
    expect(element.textContent).toBe("Po");
    await vi.runAllTimersAsync();
    await typing;
    expect(element.textContent).toBe("Povi");
  });

  it("writes immediately when the delay is zero", async () => {
    const element = { textContent: "" };

    await typeText(element, "Hallo", { delayMs: 0 });

    expect(element.textContent).toBe("Hallo");
  });

  it("stops when aborted", async () => {
    vi.useFakeTimers();
    const element = { textContent: "" };
    const controller = new AbortController();
    const typing = typeText(element, "Povi", {
      delayMs: 20,
      signal: controller.signal,
    });

    controller.abort();
    await vi.runAllTimersAsync();
    await typing;

    expect(element.textContent).toBe("P");
  });
});
