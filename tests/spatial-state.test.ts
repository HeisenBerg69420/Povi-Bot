import { describe, expect, it } from "vitest";
import { SpatialState } from "../src/spatial-state";

describe("spatial state", () => {
  it("stores the latest mouse position", () => {
    const state = new SpatialState();

    state.updateMousePosition({ x: 320, y: 180 });

    expect(state.mousePosition).toEqual({ x: 320, y: 180 });
  });

  it("calculates the window center after position and size are known", () => {
    const state = new SpatialState();

    state.updateWindowPosition({ x: 100, y: 50 });
    expect(state.windowCenter).toBeNull();

    state.updateWindowSize({ width: 220, height: 180 });
    expect(state.windowCenter).toEqual({ x: 210, y: 140 });
  });

  it("updates the center only when the window geometry changes", () => {
    const state = new SpatialState();
    state.updateWindowPosition({ x: 100, y: 50 });
    state.updateWindowSize({ width: 220, height: 180 });

    state.updateWindowPosition({ x: 140, y: 70 });
    expect(state.windowCenter).toEqual({ x: 250, y: 160 });

    state.updateWindowSize({ width: 300, height: 200 });
    expect(state.windowCenter).toEqual({ x: 290, y: 170 });
  });

  it("does not expose mutable internal values", () => {
    const state = new SpatialState();
    state.updateMousePosition({ x: 10, y: 20 });
    const position = state.mousePosition;

    if (position) {
      position.x = 999;
    }

    expect(state.mousePosition).toEqual({ x: 10, y: 20 });
  });
});
