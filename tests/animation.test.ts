import { describe, expect, it } from "vitest";
import {
  animationSequence,
  frameAt,
  frameAtMousePosition,
  nextAnimationIndex,
} from "../src/animation";
import { SpatialState } from "../src/spatial-state";

describe("animation sequence", () => {
  it("loops back to the first animation", () => {
    expect(nextAnimationIndex(animationSequence.length - 1)).toBe(0);
  });

  it("selects deterministic frames", () => {
    const animation = animationSequence[0];
    expect(frameAt(0, animation)).toEqual({ column: 0, row: animation.row });
    expect(frameAt(animation.frameDurationMs, animation)).toEqual({
      column: 1,
      row: animation.row,
    });
    expect(frameAt(animation.frameDurationMs * animation.frames.length, animation)).toEqual({
      column: 0,
      row: animation.row,
    });
  });
});

describe("mouse-facing frame", () => {
  it("maps a full circle clockwise to 16 sprite frames", () => {
    const state = new SpatialState();
    state.updateWindowPosition({ x: 50, y: 50 });
    state.updateWindowSize({ width: 100, height: 100 });

    for (let directionIndex = 0; directionIndex < 16; directionIndex += 1) {
      const angleFromTop = (directionIndex * 2 * Math.PI) / 16;
      state.updateMousePosition({
        x: 100 + Math.sin(angleFromTop) * 50,
        y: 100 - Math.cos(angleFromTop) * 50,
      });

      expect(frameAtMousePosition(state)).toEqual({
        spriteFrame: {
          column: directionIndex % 8,
          row: directionIndex < 8 ? 9 : 10,
        },
        lookDefinitionIndex: directionIndex < 8 ? 0 : 1,
      });
    }
  });

  it("returns no frame without a direction", () => {
    const state = new SpatialState();
    expect(frameAtMousePosition(state)).toBeNull();

    state.updateWindowPosition({ x: 50, y: 50 });
    state.updateWindowSize({ width: 100, height: 100 });
    state.updateMousePosition({ x: 100, y: 100 });

    expect(frameAtMousePosition(state)).toBeNull();
  });
});
