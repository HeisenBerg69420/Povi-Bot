import { describe, expect, it } from "vitest";
import { advanceHorizontal, clamp } from "../src/movement";

describe("horizontal movement", () => {
  it("advances inside the available range", () => {
    expect(advanceHorizontal(10, 1, 4, 0, 100)).toEqual({ x: 14, direction: 1 });
  });

  it("turns around at both boundaries", () => {
    expect(advanceHorizontal(99, 1, 4, 0, 100)).toEqual({ x: 100, direction: -1 });
    expect(advanceHorizontal(1, -1, 4, 0, 100)).toEqual({ x: 0, direction: 1 });
  });

  it("clamps arbitrary values", () => {
    expect(clamp(-5, 0, 100)).toBe(0);
    expect(clamp(120, 0, 100)).toBe(100);
  });
});

