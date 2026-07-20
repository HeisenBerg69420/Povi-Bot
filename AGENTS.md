# Repository Guidelines

## Project Structure & Module Organization

This repository contains a cross-platform transparent desktop-pet overlay built with Tauri 2 and TypeScript. Keep browser-facing code in `src/` and native window integration in `src-tauri/`.

- `src/main.ts`: application startup and event wiring
- `src/animation.ts`: hard-coded animation states and frame timing
- `src/movement.ts`: movement, dragging, and screen-boundary behavior
- `src/styles.css`: transparent-window and sprite presentation rules
- `public/assets/`: transparent PNG/WebP sprite sheets and icons
- `src-tauri/src/`: Rust commands and platform-specific behavior
- `tests/`: unit and integration tests mirroring the source layout

Avoid placing generated installers, compiled assets, or Tauri `target/` output under version control.

## Build, Test, and Development Commands

Use the scripts defined in `package.json` once the project scaffold is present:

- `npm install`: install JavaScript and Tauri dependencies.
- `npm run tauri dev`: launch the transparent overlay with live reload.
- `npm run build`: compile and type-check the frontend.
- `npm run tauri build`: create the native macOS or Windows package.
- `npm test`: run the automated test suite.
- `npm run lint`: check TypeScript and styling conventions.

Run native builds on the target operating system; test both macOS and Windows before release.

## Coding Style & Naming Conventions

Use TypeScript with two-space indentation, semicolons, and explicit types at module boundaries. Name functions and variables in `camelCase`, classes and types in `PascalCase`, and files in lowercase (`animation.ts`) or kebab-case when multiple words are necessary. Keep animation definitions declarative and centralized rather than scattering frame numbers through UI code. Format Rust with `cargo fmt` and lint it with `cargo clippy`.

## Testing Guidelines

Use Vitest for deterministic animation-state, timing, and boundary tests. Name files `*.test.ts`, for example `tests/movement.test.ts`. Mock time and random values so tests remain repeatable. Before submitting changes, manually verify transparency, dragging, always-on-top behavior, multiple displays, and reduced-motion handling on each supported platform.

## Commit & Pull Request Guidelines

No Git history exists yet, so use Conventional Commits such as `feat: add idle animation` or `fix: clamp pet to work area`. Keep commits focused. Pull requests must describe the behavior change, list tested operating systems, link relevant issues, and include a short screen recording for visual or animation changes. Never commit signing credentials, platform certificates, or local configuration files.
