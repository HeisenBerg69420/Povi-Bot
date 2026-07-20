import { describe, expect, it } from "vitest";
import { rgbaToRgb } from "../src/vision";

describe("vision frame conversion", () => {
  it("removes alpha bytes without changing RGB channels", () => {
    expect(rgbaToRgb(new Uint8ClampedArray([10, 20, 30, 40, 50, 60, 70, 80]))).toEqual([
      10,
      20,
      30,
      50,
      60,
      70,
    ]);
  });

  it("rejects incomplete pixels", () => {
    expect(() => rgbaToRgb(new Uint8ClampedArray([10, 20, 30]))).toThrow(
      "RGBA data length must be divisible by four",
    );
  });
});
