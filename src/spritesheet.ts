export const spriteSheet = {
  url: "/assets/spritesheet.webp",
  width: 1_536,
  height: 2_288,
  columns: 8,
  rows: 11,
  frameWidth: 192,
  frameHeight: 208,
} as const;

export function frameBackgroundPosition(column: number, row: number): string {
  return `${-column * spriteSheet.frameWidth}px ${-row * spriteSheet.frameHeight}px`;
}

