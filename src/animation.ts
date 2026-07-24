import type { SpatialState } from "./spatial-state";

export type AnimationName = "idle" | "walk" | "wave" | "sleep";

export type lookdDir = "right" | "left";
export type UpDownDir = "up" | "down"; // up = from up to down

export interface lookFrame {
  spriteFrame: SpriteFrame;
  lookDefinitionIndex: number;
}

export interface AnimationDefinition {
  name: AnimationName;
  durationMs: number;
  frameDurationMs: number;
  row: number;
  frames: readonly number[];
}

export interface LookDefinition {
  name: lookdDir;
  dir: UpDownDir;
  row: number;
  frames: readonly number[];
}



/*
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
*/

export const animationSequence: readonly AnimationDefinition[] = [
  {
    name: "idle",
    durationMs: 4_000,
    frameDurationMs: 500,
    row: 9,
    frames: [0, 1, 2, 3, 4, 5, 6, 7],
  },
  {
    name: "walk",
    durationMs: 6_000,
    frameDurationMs: 180,
    row: 10,
    frames: [0, 1, 2, 3, 4, 5, 6, 7],
  },
] as const;

export const lookSequence: readonly LookDefinition[] = [
  {
    name: "right",
    dir: "up",
    row: 9,
    frames: [0, 1, 2, 3, 4, 5, 6, 7],
  },
  {
    name: "left",
    dir: "down",
    row: 10,
    frames: [0, 1, 2, 3, 4, 5, 6, 7],
  },
] as const;

const LOOK_FRAMES: readonly SpriteFrame[] = lookSequence.flatMap((definition) =>
  definition.frames.map((column) => ({ column, row: definition.row })),
);
const FULL_CIRCLE_RADIANS = 2 * Math.PI;
const LOOK_SECTOR_SIZE_RADIANS = FULL_CIRCLE_RADIANS / LOOK_FRAMES.length;


//Frame r9,f0 look up
//Frame r9, f1 top right
//Frame r9, f2 top middle right
//Frame r9, f3 middle right
//Frame r9, f4 middle right turn
//Frame r9, f5 small down middle right
//Frame r9, f6 down middle right
//Frame r9, f7 down right

//Frame r10, f0 look down
//bottom up inverse of r9

export interface SpriteFrame {
  column: number;
  row: number;
}

export function nextAnimationIndex(currentIndex: number): number {
  return (currentIndex + 1) % animationSequence.length;
}

export function frameAtMousePosition(
  spatialState: SpatialState,
): lookFrame | null {
  const mousePosition = spatialState.mousePosition;
  const windowCenter = spatialState.windowCenter;

  if (!mousePosition || !windowCenter) {
    return null;
  }

  const deltaX = mousePosition.x - windowCenter.x;
  const deltaY = mousePosition.y - windowCenter.y;

  if (deltaX === 0 && deltaY === 0) {
    return null;
  }

  const angleFromPositiveXAxis = Math.atan2(deltaY, deltaX);
  const clockwiseAngleFromTop =
    (angleFromPositiveXAxis + Math.PI / 2 + FULL_CIRCLE_RADIANS) %
    FULL_CIRCLE_RADIANS;
  const directionIndex =
    Math.floor(
      (clockwiseAngleFromTop + LOOK_SECTOR_SIZE_RADIANS / 2) /
        LOOK_SECTOR_SIZE_RADIANS,
    ) % LOOK_FRAMES.length;
  const frame = LOOK_FRAMES[directionIndex];
  const index = Math.floor(directionIndex / 8);

  const ret: lookFrame = {
    spriteFrame: frame,
    lookDefinitionIndex: index,
  };

  return { ...ret };
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
