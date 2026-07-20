import { ENABLE_VISION_DEBUG_VIEW } from "./config";
import {
  WebcamVisionStream,
  getVisionStatus,
  type ObjectDetection,
  type VisionPrediction,
} from "./vision";
import "./vision-debug.css";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Required element not found: ${selector}`);
  }
  return element;
}

const preview = requireElement<HTMLCanvasElement>("#camera-preview");
const placeholder = requireElement<HTMLParagraphElement>("#camera-placeholder");
const streamState = requireElement<HTMLSpanElement>("#stream-state");
const debugOutput = requireElement<HTMLPreElement>("#debug-output");
const startButton = requireElement<HTMLButtonElement>("#start-camera");
const stopButton = requireElement<HTMLButtonElement>("#stop-camera");
const visionStream = new WebcamVisionStream(preview);

function setStreamState(state: "idle" | "running" | "error", label: string): void {
  streamState.dataset.state = state;
  streamState.textContent = label;
  startButton.disabled = state === "running";
  stopButton.disabled = state !== "running";
  placeholder.hidden = state === "running";
}

function drawDetection(detection: ObjectDetection): void {
  const context = preview.getContext("2d");
  if (!context) {
    return;
  }

  const bounds = detection.boundingBox;
  const label = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
  context.strokeStyle = "#ff0000";
  context.lineWidth = 2;
  context.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
  context.font = "12px system-ui, sans-serif";
  const labelWidth = context.measureText(label).width + 10;
  const labelY = Math.max(0, bounds.y - 20);
  context.fillStyle = "rgba(0, 0, 0, 0.8)";
  context.fillRect(bounds.x, labelY, labelWidth, 20);
  context.fillStyle = "#ffffff";
  context.fillText(label, bounds.x + 5, labelY + 14);
}

function displayPrediction(prediction: VisionPrediction): void {
  prediction.detections.forEach(drawDetection);
  const action = prediction.action.warmedUp
    ? prediction.action.label
    : `${prediction.action.label} (Warm-up)`;
  const detections = prediction.detections
    .map((detection) => {
      const bounds = detection.boundingBox;
      return `${detection.label} ${(detection.confidence * 100).toFixed(0)}% [${bounds.x}, ${bounds.y}, ${bounds.width}, ${bounds.height}]`;
    })
    .join("; ");
  debugOutput.textContent = [
    `Action: ${action}`,
    `Confidence: ${(prediction.action.confidence * 100).toFixed(1)}%`,
    `Detections: ${prediction.detections.length}${detections ? ` (${detections})` : ""}`,
    `Frame: ${prediction.frameIndex}`,
  ].join("\n");
}

function displayError(error: Error): void {
  visionStream.stop();
  setStreamState("error", "Fehler");
  placeholder.hidden = false;
  placeholder.textContent = error.message;
}

async function startCamera(): Promise<void> {
  if (visionStream.running) {
    return;
  }

  setStreamState("idle", "Starte …");
  try {
    await visionStream.start({
      onPrediction: displayPrediction,
      onError: displayError,
    });
    setStreamState("running", "Live");
  } catch (error) {
    displayError(error instanceof Error ? error : new Error(String(error)));
  }
}

function stopCamera(): void {
  visionStream.stop();
  setStreamState("idle", "Pausiert");
  placeholder.textContent = "Kamera ist pausiert.";
}

startButton.addEventListener("click", () => void startCamera());
stopButton.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", () => visionStream.stop());

if (ENABLE_VISION_DEBUG_VIEW) {
  void startCamera();
} else {
  placeholder.textContent = "Debug-View deaktiviert";
  startButton.disabled = true;
}
