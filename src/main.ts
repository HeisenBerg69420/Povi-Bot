import {
  PhysicalPosition,
  Window as TauriWindow,
  currentMonitor,
  getCurrentWindow,
} from "@tauri-apps/api/window";
import { animationSequence, frameAt, nextAnimationIndex } from "./animation";
import { ENABLE_VISION_DEBUG_VIEW } from "./config";
import { advanceHorizontal, type HorizontalDirection } from "./movement";
import { frameBackgroundPosition, spriteSheet } from "./spritesheet";
import "./styles.css";

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
let animationIndex = 0;
let animationStartedAt = performance.now();
let direction: HorizontalDirection = 1;
let movingWindow = false;

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

  requestAnimationFrame(applyAnimation);
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
    const [position, size, monitor] = await Promise.all([
      appWindow.outerPosition(),
      appWindow.outerSize(),
      currentMonitor(),
    ]);

    if (!monitor) {
      return;
    }

    const minimumX = monitor.position.x;
    const maximumX = monitor.position.x + monitor.size.width - size.width;
    const movement = advanceHorizontal(position.x, direction, 4, minimumX, maximumX);
    direction = movement.direction;
    await appWindow.setPosition(new PhysicalPosition(Math.round(movement.x), position.y));
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
void configureVisionDebugWindow().catch((error: unknown) => {
  console.error("Unable to configure the vision debug window", error);
});
requestAnimationFrame(applyAnimation);
window.setInterval(() => void moveWindow(), 32);
