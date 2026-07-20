import { describe, expect, it } from "vitest";
import { frameBackgroundPosition, spriteSheet } from "../src/spritesheet";

describe("sprite sheet", () => {
  it("matches the source image grid", () => {
    expect(spriteSheet.frameWidth * spriteSheet.columns).toBe(spriteSheet.width);
    expect(spriteSheet.frameHeight * spriteSheet.rows).toBe(spriteSheet.height);
  });

  it("converts a frame coordinate into a CSS background position", () => {
    expect(frameBackgroundPosition(3, 2)).toBe("-576px -416px");
  });
});

