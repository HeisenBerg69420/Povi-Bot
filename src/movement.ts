export type HorizontalDirection = -1 | 1;

export interface HorizontalMovement {
  x: number;
  direction: HorizontalDirection;
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

export function advanceHorizontal(
  x: number,
  direction: HorizontalDirection,
  distance: number,
  minimumX: number,
  maximumX: number,
): HorizontalMovement {
  const proposedX = x + distance * direction;

  if (proposedX <= minimumX) {
    return { x: minimumX, direction: 1 };
  }

  if (proposedX >= maximumX) {
    return { x: maximumX, direction: -1 };
  }

  return { x: proposedX, direction };
}

