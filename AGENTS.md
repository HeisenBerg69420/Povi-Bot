# Repository Guidelines

## Project Overview

This repository is a cross-platform transparent desktop pet built with Tauri 2,
TypeScript, Vite, and Rust. The frontend renders and moves the pet; Rust owns the
native Tauri process and local, stateful vision backends. Keep camera processing
local and opt-in. Do not present the current motion prototypes as trained semantic
ML models.

## Project Structure

- `src/main.ts`: application startup, pet-window behavior, and debug-window setup
- `src/animation.ts`: declarative animation states and frame timing
- `src/spritesheet.ts`: sprite-sheet dimensions and frame coordinates
- `src/movement.ts`: walking, dragging, viewport bounds, and reduced motion
- `src/vision.ts`: throttled webcam capture and the typed Tauri vision client
- `src/movinet.ts`: client for the legacy MoViNet-compatible motion prototype
- `src/vision-debug.ts` and `vision-debug.html`: optional camera diagnostics UI
- `src/config.ts`: development feature flags, including the vision debug window
- `public/assets/`: runtime sprite sheets and other static assets
- `src-tauri/src/main.rs`: Tauri setup, managed state, and command registration
- `src-tauri/src/vision.rs`: foreground detection plus temporal action prototype
- `src-tauri/src/movinet.rs`: legacy streaming motion-classification prototype
- `src-tauri/capabilities/`: Tauri 2 window permissions
- `tests/`: Vitest suites mirroring frontend modules
- `docs/vision-architecture.md`: vision API, privacy, and ONNX migration design
- `models/`: instructions/placeholders only; model weights stay untracked

Generated `dist/`, `node_modules/`, `src-tauri/target/`, installers, credentials,
certificates, camera recordings, and model weights must not be committed.

## Build, Test, and Development Commands

Use the repository scripts as the canonical entry points:

- `npm install`: install JavaScript and Tauri tooling
- `npm run dev`: run the Vite frontend in a browser
- `npm run tauri dev`: launch the desktop overlay with live reload
- `npm run build`: type-check TypeScript and create the frontend bundle
- `npm test`: run the frontend Vitest suites (also used by Node CI)
- `npm run test:all`: run both frontend and Rust tests locally
- `npm run test:frontend`: run only deterministic Vitest suites
- `npm run test:rust`: run Rust unit tests through the Tauri manifest
- `npm run lint`: type-check TypeScript without invoking Rust tooling
- `npm run lint:all`: type-check TypeScript, check Rust formatting, and run Clippy
- `npm run tauri build`: create the native package on the current target OS

Run `npm run test:all`, `npm run lint:all`, and `npm run build` before submitting changes.
Build and manually inspect native packages on both Windows and macOS for a release.

## TypeScript and Frontend Conventions

Use two-space indentation, semicolons, and explicit types at module boundaries.
Use `camelCase` for values/functions, `PascalCase` for types/classes, and lowercase
or kebab-case filenames. Keep animation data centralized rather than scattering
frame numbers. Keep Tauri command names and camelCase payloads aligned with their
Rust definitions. Do not start the webcam automatically: camera access requires an
explicit user action and a visible active-camera state.

## Rust and Tauri Conventions

Use standard `rustfmt`, idiomatic `snake_case`, and descriptive `Result<_, String>`
errors at the command boundary. Keep platform/native integration in `src-tauri/`.
Register new commands in `src-tauri/src/main.rs` and grant only the minimum Tauri
capabilities required. Stateful frame processors live behind Tauri managed
`Mutex` state; validate input before mutating that state. Preserve stream
invariants: fixed dimensions until reset, nondecreasing timestamps, bounded
temporal buffers, and a complete reset path. Keep pure processing logic separate
from `#[tauri::command]` wrappers so it remains unit-testable without a window.

## Testing Guidelines

Frontend tests use Vitest and belong in `tests/*.test.ts`. Mock time, randomness,
browser media APIs, and Tauri calls so suites are deterministic and never require
a real camera or desktop session.

Rust unit tests live beside private implementation details in `#[cfg(test)]`
modules. Cover success paths, malformed RGB buffers, zero/overflowing dimensions,
dimension changes, timestamp ordering, warm-up thresholds, reset behavior, and
state remaining unchanged after rejected input. Prefer small synthetic RGB frames
and exact assertions on labels, indices, bounds, and error messages. New native
modules should include tests in the same change.

Manual checks still cover behavior automated tests cannot reliably prove:
transparency, click/drag handling, always-on-top behavior, multiple displays,
right-click shutdown, reduced motion, camera consent/indicator behavior, and the
vision debug window on each supported OS.

## Commit and Pull Request Guidelines

Use focused Conventional Commits, for example `feat: add vision stream reset` or
`test: cover rust timestamp validation`. Pull requests should describe user-visible
and architectural changes, list automated commands run and operating systems
tested, link relevant issues, and include a short recording for visual, animation,
window, or camera-debug changes. Never commit secrets or signing material.
