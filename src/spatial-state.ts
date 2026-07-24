export interface ScreenPoint {
  x: number;
  y: number;
}

export interface ScreenSize {
  width: number;
  height: number;
}

export class SpatialState {
  private mousePositionValue: ScreenPoint | null = null;
  private windowPositionValue: ScreenPoint | null = null;
  private windowSizeValue: ScreenSize | null = null;
  private windowCenterValue: ScreenPoint | null = null;

  get mousePosition(): ScreenPoint | null {
    return this.copyPoint(this.mousePositionValue);
  }

  get windowPosition(): ScreenPoint | null {
    return this.copyPoint(this.windowPositionValue);
  }

  get windowSize(): ScreenSize | null {
    if (!this.windowSizeValue) {
      return null;
    }

    return { ...this.windowSizeValue };
  }

  get windowCenter(): ScreenPoint | null {
    return this.copyPoint(this.windowCenterValue);
  }

  updateMousePosition(position: ScreenPoint): void {
    this.mousePositionValue = { ...position };
  }

  updateWindowPosition(position: ScreenPoint): void {
    this.windowPositionValue = { ...position };
    this.updateWindowCenter();
  }

  updateWindowSize(size: ScreenSize): void {
    this.windowSizeValue = { ...size };
    this.updateWindowCenter();
  }

  private updateWindowCenter(): void {
    if (!this.windowPositionValue || !this.windowSizeValue) {
      this.windowCenterValue = null;
      return;
    }

    this.windowCenterValue = {
      x: this.windowPositionValue.x + this.windowSizeValue.width / 2,
      y: this.windowPositionValue.y + this.windowSizeValue.height / 2,
    };
  }

  private copyPoint(point: ScreenPoint | null): ScreenPoint | null {
    return point ? { ...point } : null;
  }
}
