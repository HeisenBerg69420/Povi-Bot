import {
  PhysicalPosition,
  Window as TauriWindow,
  currentMonitor,
  getCurrentWindow,
} from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import {
  animationSequence,
  frameAt,
  frameAtMousePosition,
  lookSequence,
  nextAnimationIndex,
} from "./animation";
import { ENABLE_VISION_DEBUG_VIEW } from "./config";
import { advanceHorizontal, type HorizontalDirection } from "./movement";
import { SpatialState } from "./spatial-state";
import { frameBackgroundPosition, spriteSheet } from "./spritesheet";
import "./styles.css";

const MOUSE_POLL_INTERVAL_MS = 50;
const WINDOW_MOVE_INTERVAL_MS = 32;

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Required element not found: ${selector}`);
  }
  return element;
}

const pet = requireElement<HTMLButtonElement>("#pet");
const sprite = requireElement<HTMLSpanElement>(".pet__sprite");
const status = requireElement<HTMLSpanElement>(".pet__status");

const appWindow = getCurrentWindow();
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const spatialState = new SpatialState();
let animationIndex = 0;
let animationStartedAt = performance.now();
let direction: HorizontalDirection = 1;
let movingWindow = false;
let updatingMousePosition = false;
let lastMouseDebugState = "";

function logMouseDebugState(
  mousePosition: { x: number; y: number },
  currentName: string,
): void {
  const windowCenter = spatialState.windowCenter;
  let angleDegrees: number | null = null;

  if (windowCenter) {
    const deltaX = mousePosition.x - windowCenter.x;
    const deltaY = mousePosition.y - windowCenter.y;

    if (deltaX !== 0 || deltaY !== 0) {
      const fullCircle = 2 * Math.PI;
      const angleFromPositiveXAxis = Math.atan2(deltaY, deltaX);
      const clockwiseAngleFromTop =
        (angleFromPositiveXAxis + Math.PI / 2 + fullCircle) % fullCircle;
      angleDegrees = Math.round((clockwiseAngleFromTop * 1800) / Math.PI) / 10;
    }
  }

  const debugState = JSON.stringify({
    x: mousePosition.x,
    y: mousePosition.y,
    angleDegrees,
    currentName,
  });

  if (debugState === lastMouseDebugState) {
    return;
  }

  lastMouseDebugState = debugState;
  console.log("Mouse debug:", {
    mousePosition,
    angleDegrees,
    currentName,
  });
}

async function configureVisionDebugWindow(): Promise<void> {
  const debugWindow = await TauriWindow.getByLabel("vision-debug");
  if (!debugWindow) {
    return;
  }

  if (!ENABLE_VISION_DEBUG_VIEW) {
    await debugWindow.hide();
    return;
  }

  const [petPosition, petSize, debugSize, monitor] = await Promise.all([
    appWindow.outerPosition(),
    appWindow.outerSize(),
    debugWindow.outerSize(),
    currentMonitor(),
  ]);
  const gap = 12;
  let x = petPosition.x + petSize.width + gap;
  let y = petPosition.y;

  if (monitor) {
    const monitorRight = monitor.position.x + monitor.size.width;
    const monitorBottom = monitor.position.y + monitor.size.height;
    if (x + debugSize.width > monitorRight) {
      x = petPosition.x - debugSize.width - gap;
    }
    x = Math.max(monitor.position.x, x);
    y = Math.min(Math.max(monitor.position.y, y), monitorBottom - debugSize.height);
  }

  await debugWindow.setPosition(new PhysicalPosition(Math.round(x), Math.round(y)));
  await debugWindow.show();
}

async function initializeSpatialState(): Promise<void> {
  const [position, size] = await Promise.all([
    appWindow.outerPosition(),
    appWindow.outerSize(),
  ]);
  spatialState.updateWindowPosition(position);
  spatialState.updateWindowSize(size);

  await Promise.all([
    appWindow.onMoved(({ payload }) => {
      spatialState.updateWindowPosition(payload);
    }),
    appWindow.onResized(({ payload }) => {
      spatialState.updateWindowSize(payload);
    }),
  ]);
}

async function updateMousePosition(): Promise<void> {
  if (updatingMousePosition) {
    return;
  }

  updatingMousePosition = true;
  try {
    const [x, y] = await invoke<[number, number]>("get_mouse_position");
    spatialState.updateMousePosition({ x, y });

    const lookFrame = frameAtMousePosition(spatialState);
    if (!lookFrame) {
      logMouseDebugState({ x, y }, "none");
      return;
    }

    const definition = lookSequence[lookFrame.lookDefinitionIndex];
    const { spriteFrame } = lookFrame;
    logMouseDebugState({ x, y }, definition.name);

    sprite.style.backgroundImage = `url("${spriteSheet.url}")`;
    sprite.style.backgroundSize = `${spriteSheet.width}px ${spriteSheet.height}px`;
    sprite.style.backgroundPosition = frameBackgroundPosition(
      spriteFrame.column,
      spriteFrame.row,
    );
    pet.dataset.state = definition.name;
    pet.dataset.frame = `${spriteFrame.column},${spriteFrame.row}`;
    pet.dataset.lookDirection = definition.name;
    pet.dataset.direction = "right";
    status.textContent = definition.name;
  } finally {
    updatingMousePosition = false;
  }
}

function applyAnimation(now: number): void {
  let current = animationSequence[animationIndex];
  let elapsed = now - animationStartedAt;

  if (!reduceMotion.matches && elapsed >= current.durationMs) {
    animationIndex = nextAnimationIndex(animationIndex);
    animationStartedAt = now;
    current = animationSequence[animationIndex];
    elapsed = 0;
  }

  if (reduceMotion.matches) {
    current = animationSequence[0];
    elapsed = 0;
  }

  const frame = frameAt(elapsed, current);
  sprite.style.backgroundImage = `url("${spriteSheet.url}")`;
  sprite.style.backgroundSize = `${spriteSheet.width}px ${spriteSheet.height}px`;
  sprite.style.backgroundPosition = frameBackgroundPosition(frame.column, frame.row);
  pet.dataset.state = current.name;
  pet.dataset.frame = `${frame.column},${frame.row}`;
  pet.dataset.direction = direction === 1 ? "right" : "left";
  status.textContent = current.name;

  //requestAnimationFrame(applyAnimation);
}

async function moveWindow(): Promise<void> {
  if (
    movingWindow ||
    reduceMotion.matches ||
    animationSequence[animationIndex].name !== "walk"
  ) {
    return;
  }

  movingWindow = true;
  try {
    const position = spatialState.windowPosition;
    const size = spatialState.windowSize;
    const monitor = await currentMonitor();

    if (!position || !size || !monitor) {
      return;
    }

    const minimumX = monitor.position.x;
    const maximumX = monitor.position.x + monitor.size.width - size.width;
    const movement = advanceHorizontal(position.x, direction, 4, minimumX, maximumX);
    direction = movement.direction;
    const nextPosition = new PhysicalPosition(Math.round(movement.x), position.y);
    await appWindow.setPosition(nextPosition);
    spatialState.updateWindowPosition(nextPosition);
  } finally {
    movingWindow = false;
  }
}

pet.addEventListener("pointerdown", async (event) => {
  if (event.button === 0) {
    await appWindow.startDragging();
  }
});

pet.addEventListener("contextmenu", async (event) => {
  event.preventDefault();
  const debugWindow = await TauriWindow.getByLabel("vision-debug");
  await debugWindow?.close();
  await appWindow.close();
});

reduceMotion.addEventListener("change", () => {
  document.documentElement.classList.toggle("reduced-motion", reduceMotion.matches);
});

document.documentElement.classList.toggle("reduced-motion", reduceMotion.matches);
void initializeSpatialState().catch((error: unknown) => {
  console.error("Unable to initialize spatial state", error);
});
void configureVisionDebugWindow().catch((error: unknown) => {
  console.error("Unable to configure the vision debug window", error);
});
requestAnimationFrame(applyAnimation);
window.setInterval(() => {
  void updateMousePosition().catch((error: unknown) => {
    console.error("Unable to update mouse position", error);
  });
}, MOUSE_POLL_INTERVAL_MS);
window.setInterval(() => void moveWindow(), WINDOW_MOVE_INTERVAL_MS);
