import { describe, expect, it } from "vitest";
import { animationSequence, frameAt, nextAnimationIndex } from "../src/animation";

describe("animation sequence", () => {
  it("loops back to the first animation", () => {
    expect(nextAnimationIndex(animationSequence.length - 1)).toBe(0);
  });

  it("selects deterministic frames", () => {
    const animation = animationSequence[0];
    expect(frameAt(0, animation)).toEqual({ column: 0, row: 0 });
    expect(frameAt(animation.frameDurationMs, animation)).toEqual({
      column: 1,
      row: 0,
    });
    expect(frameAt(animation.frameDurationMs * animation.frames.length, animation)).toEqual({
      column: 0,
      row: 0,
    });
  });
});

