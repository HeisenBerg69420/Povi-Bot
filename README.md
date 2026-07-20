# Desktop Pet Overlay

Cross-platform scaffold for a transparent desktop pet based on Tauri 2,
TypeScript and Rust.

## Current status

The pet renders frames from `public/assets/spritesheet.webp`, cycles through
declarative animation states, walks horizontally, supports native window
dragging and closes on right-click.

A local webcam vision foundation is also present:

- `src/vision.ts` captures webcam frames on demand, limits the frame rate and
  forwards only one frame at a time.
- `src-tauri/src/vision.rs` implements a two-stage, stateful interface for
  foreground object detection followed by temporal action recognition.
- The runnable backend is intentionally labelled `prototype`: it detects image
  changes and classifies centroid motion, but does not pretend to be a trained
  semantic model.
- `src-tauri/src/movinet.rs` remains as the earlier MoViNet-compatible motion
  prototype for compatibility.

See [docs/vision-architecture.md](docs/vision-architecture.md) for the Rust vs.
Python decision, API contracts, ONNX migration path, privacy requirements and
model checklist.

## Vision commands

- `vision_status`: active detector/action implementations and stream state
- `vision_process_frame`: validate and process one RGB frame through both stages
- `vision_reset`: clear detector background and temporal action state
- `movinet_status`, `movinet_classify_frame`, `movinet_reset`: legacy prototype

The regular pet window never starts the webcam automatically. Product UI must
call `WebcamVisionStream.start(...)` from an explicit user action and show that
the camera is active.

During development, `ENABLE_VISION_DEBUG_VIEW` in `src/config.ts` controls a
second window next to the pet. When enabled, it shows the local camera stream,
foreground bounding boxes, action label, confidence and backend status. The
window also provides camera start/stop controls. Set the variable to `false` to
keep the window hidden and prevent camera startup.

## Development

Install Node.js, npm and the Rust toolchain, then run:

```sh
npm install
npm test
npm run tauri dev
```

Use `npm run build` for the frontend bundle and `npm run tauri build` for the
native package. Native packages must be built and checked on both target
operating systems.

## Controls

- Left-drag the pet to reposition it.
- Right-click the pet to close the application.
- Enable the operating system's reduced-motion setting to disable movement and
  animation.

The sprite sheet is 1536 × 2288 pixels and uses an 8 × 11 grid. Each frame is
192 × 208 pixels. Update `src/spritesheet.ts` and the mappings in
`src/animation.ts` when replacing the artwork.
