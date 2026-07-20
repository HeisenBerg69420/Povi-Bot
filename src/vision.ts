import { invoke } from "@tauri-apps/api/core";

export const defaultVisionCapture = {
  width: 320,
  height: 240,
  framesPerSecond: 8,
} as const;

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ObjectDetection {
  classId: number;
  label: string;
  confidence: number;
  boundingBox: BoundingBox;
}

export interface ActionPrediction {
  classId: number;
  label: string;
  confidence: number;
  warmedUp: boolean;
  observedFrames: number;
}

export interface VisionPrediction {
  detections: ObjectDetection[];
  action: ActionPrediction;
  frameIndex: number;
  timestampMs: number;
}

export interface VisionStatus {
  ready: boolean;
  runtime: "prototype" | "onnx" | "tflite";
  detector: string;
  actionRecognizer: string;
  recommendedWidth: number;
  recommendedHeight: number;
  temporalWindowFrames: number;
  framesSeen: number;
  note: string;
}

export interface VisionFrame {
  width: number;
  height: number;
  rgb: number[];
  timestampMs: number;
}

export interface WebcamVisionOptions {
  deviceId?: string;
  width?: number;
  height?: number;
  framesPerSecond?: number;
  onPrediction: (prediction: VisionPrediction) => void;
  onError?: (error: Error) => void;
}

export function getVisionStatus(): Promise<VisionStatus> {
  return invoke<VisionStatus>("vision_status");
}

export function resetVision(): Promise<void> {
  return invoke<void>("vision_reset");
}

export function processVisionFrame(frame: VisionFrame): Promise<VisionPrediction> {
  return invoke<VisionPrediction>("vision_process_frame", { frame });
}

export function rgbaToRgb(rgba: Uint8ClampedArray): number[] {
  if (rgba.length % 4 !== 0) {
    throw new Error("RGBA data length must be divisible by four");
  }

  const rgb = new Array<number>((rgba.length / 4) * 3);
  for (let source = 0, target = 0; source < rgba.length; source += 4) {
    rgb[target] = rgba[source];
    rgb[target + 1] = rgba[source + 1];
    rgb[target + 2] = rgba[source + 2];
    target += 3;
  }
  return rgb;
}

export class WebcamVisionStream {
  private mediaStream: MediaStream | null = null;
  private animationFrame: number | null = null;
  private processing = false;
  private lastCaptureAt = 0;
  private readonly video = document.createElement("video");
  private readonly canvas: HTMLCanvasElement;

  constructor(previewCanvas?: HTMLCanvasElement) {
    this.canvas = previewCanvas ?? document.createElement("canvas");
  }

  get running(): boolean {
    return this.mediaStream !== null;
  }

  async start(options: WebcamVisionOptions): Promise<void> {
    if (this.running) {
      throw new Error("The webcam vision stream is already running");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera capture is not available in this WebView");
    }

    const width = options.width ?? defaultVisionCapture.width;
    const height = options.height ?? defaultVisionCapture.height;
    const framesPerSecond = options.framesPerSecond ?? defaultVisionCapture.framesPerSecond;
    if (width <= 0 || height <= 0 || framesPerSecond <= 0) {
      throw new Error("Capture width, height and frame rate must be greater than zero");
    }

    await resetVision();
    this.canvas.width = width;
    this.canvas.height = height;
    this.video.muted = true;
    this.video.playsInline = true;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          deviceId: options.deviceId ? { exact: options.deviceId } : undefined,
          width: { ideal: width },
          height: { ideal: height },
          frameRate: { ideal: framesPerSecond, max: framesPerSecond },
        },
      });
      this.video.srcObject = this.mediaStream;
      await this.video.play();
    } catch (error) {
      this.stop();
      throw toError(error);
    }

    const intervalMs = 1000 / framesPerSecond;
    const capture = (now: number): void => {
      if (!this.running) {
        return;
      }

      if (!this.processing && now - this.lastCaptureAt >= intervalMs) {
        this.lastCaptureAt = now;
        this.processing = true;
        void this.captureFrame(width, height)
          .then(options.onPrediction)
          .catch((error: unknown) => options.onError?.(toError(error)))
          .finally(() => {
            this.processing = false;
          });
      }
      this.animationFrame = window.requestAnimationFrame(capture);
    };

    this.animationFrame = window.requestAnimationFrame(capture);
  }

  stop(): void {
    if (this.animationFrame !== null) {
      window.cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
    this.video.pause();
    this.video.srcObject = null;
    this.processing = false;
    this.lastCaptureAt = 0;
  }

  private async captureFrame(width: number, height: number): Promise<VisionPrediction> {
    const context = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      throw new Error("Unable to create the camera capture canvas");
    }

    context.drawImage(this.video, 0, 0, width, height);
    const rgba = context.getImageData(0, 0, width, height).data;
    return processVisionFrame({
      width,
      height,
      rgb: rgbaToRgb(rgba),
      timestampMs: Math.round(performance.timeOrigin + performance.now()),
    });
  }
}

function toError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}
