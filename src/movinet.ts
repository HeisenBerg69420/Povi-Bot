import { invoke } from "@tauri-apps/api/core";

export const movinetInputSize = {
  width: 172,
  height: 172,
} as const;

export interface MovinetFrame {
  width: number;
  height: number;
  rgb: number[];
  timestampMs: number;
}

export interface MovinetPrediction {
  label: string;
  confidence: number;
  frameIndex: number;
  warmedUp: boolean;
  timestampMs: number;
}

export interface MovinetStatus {
  ready: boolean;
  runtime: "demo" | "tflite" | "onnx";
  model: string;
  expectedWidth: number;
  expectedHeight: number;
  framesSeen: number;
  note: string;
}

export function getMovinetStatus(): Promise<MovinetStatus> {
  return invoke<MovinetStatus>("movinet_status");
}

export function resetMovinet(): Promise<void> {
  return invoke<void>("movinet_reset");
}

export function classifyMovinetFrame(
  frame: MovinetFrame,
): Promise<MovinetPrediction> {
  return invoke<MovinetPrediction>("movinet_classify_frame", { frame });
}
