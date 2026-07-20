export type AnimationName = "idle" | "walk" | "wave" | "sleep";

export interface AnimationDefinition {
  name: AnimationName;
  durationMs: number;
  frameDurationMs: number;
  row: number;
  frames: readonly number[];
}

export const animationSequence: readonly AnimationDefinition[] = [
  {
    name: "idle",
    durationMs: 4_000,
    frameDurationMs: 500,
    row: 0,
    frames: [0, 1, 2, 3, 4, 5, 6],
  },
  {
    name: "walk",
    durationMs: 6_000,
    frameDurationMs: 180,
    row: 7,
    frames: [0, 1, 2, 3, 4, 5],
  },
  {
    name: "wave",
    durationMs: 2_400,
    frameDurationMs: 300,
    row: 4,
    frames: [0, 1, 2, 3, 4],
  },
  {
    name: "idle",
    durationMs: 3_000,
    frameDurationMs: 500,
    row: 3,
    frames: [0, 1, 2, 3],
  },
  {
    name: "sleep",
    durationMs: 7_000,
    frameDurationMs: 800,
    row: 5,
    frames: [0, 1, 2, 3, 4, 5, 6, 7],
  },
] as const;

export interface SpriteFrame {
  column: number;
  row: number;
}

export function nextAnimationIndex(currentIndex: number): number {
  return (currentIndex + 1) % animationSequence.length;
}

export function frameAt(
  elapsedMs: number,
  definition: AnimationDefinition,
): SpriteFrame {
  const frameIndex =
    Math.floor(elapsedMs / definition.frameDurationMs) % definition.frames.length;

  return {
    column: definition.frames[frameIndex],
    row: definition.row,
  };
}

