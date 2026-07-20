use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;
use tauri::State;

const RECOMMENDED_WIDTH: u32 = 320;
const RECOMMENDED_HEIGHT: u32 = 240;
const FOREGROUND_LUMA_THRESHOLD: u8 = 28;
const TEMPORAL_WINDOW_FRAMES: usize = 8;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisionFrame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObjectDetection {
    pub class_id: u32,
    pub label: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionPrediction {
    pub class_id: u32,
    pub label: String,
    pub confidence: f32,
    pub warmed_up: bool,
    pub observed_frames: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisionPrediction {
    pub detections: Vec<ObjectDetection>,
    pub action: ActionPrediction,
    pub frame_index: u64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisionStatus {
    pub ready: bool,
    pub runtime: &'static str,
    pub detector: &'static str,
    pub action_recognizer: &'static str,
    pub recommended_width: u32,
    pub recommended_height: u32,
    pub temporal_window_frames: usize,
    pub frames_seen: u64,
    pub note: &'static str,
}

pub trait LowLevelDetector: Send {
    fn detect(&mut self, frame: &VisionFrame) -> Result<Vec<ObjectDetection>, String>;
    fn reset(&mut self);
}

pub trait TemporalActionRecognizer: Send {
    fn classify(&mut self, frame: &VisionFrame, detections: &[ObjectDetection])
        -> ActionPrediction;
    fn reset(&mut self);
}

#[derive(Default)]
struct PrototypeForegroundDetector {
    previous_luma: Option<Vec<u8>>,
}

impl LowLevelDetector for PrototypeForegroundDetector {
    fn detect(&mut self, frame: &VisionFrame) -> Result<Vec<ObjectDetection>, String> {
        let previous_luma = self.previous_luma.replace(rgb_to_luma(&frame.rgb));
        let Some(previous_luma) = previous_luma else {
            return Ok(Vec::new());
        };
        let current_luma = self
            .previous_luma
            .as_ref()
            .expect("the current luma frame was inserted above");

        if previous_luma.len() != current_luma.len() {
            return Err("frame dimensions changed; reset the vision stream first".to_string());
        }

        let mut minimum_x = frame.width;
        let mut minimum_y = frame.height;
        let mut maximum_x = 0;
        let mut maximum_y = 0;
        let mut changed_pixels = 0_u32;

        for (index, (previous, current)) in previous_luma.iter().zip(current_luma).enumerate() {
            if previous.abs_diff(*current) < FOREGROUND_LUMA_THRESHOLD {
                continue;
            }

            let index = index as u32;
            let x = index % frame.width;
            let y = index / frame.width;
            minimum_x = minimum_x.min(x);
            minimum_y = minimum_y.min(y);
            maximum_x = maximum_x.max(x);
            maximum_y = maximum_y.max(y);
            changed_pixels += 1;
        }

        let pixel_count = frame.width * frame.height;
        let minimum_changed_pixels = (pixel_count / 500).max(4);
        if changed_pixels < minimum_changed_pixels {
            return Ok(Vec::new());
        }

        let changed_ratio = changed_pixels as f32 / pixel_count as f32;
        Ok(vec![ObjectDetection {
            class_id: 0,
            label: "foreground-region".to_string(),
            confidence: (changed_ratio * 8.0).clamp(0.1, 1.0),
            bounding_box: BoundingBox {
                x: minimum_x,
                y: minimum_y,
                width: maximum_x - minimum_x + 1,
                height: maximum_y - minimum_y + 1,
            },
        }])
    }

    fn reset(&mut self) {
        self.previous_luma = None;
    }
}

#[derive(Default)]
struct PrototypeActionRecognizer {
    centroids: VecDeque<Option<(f32, f32)>>,
}

impl TemporalActionRecognizer for PrototypeActionRecognizer {
    fn classify(
        &mut self,
        frame: &VisionFrame,
        detections: &[ObjectDetection],
    ) -> ActionPrediction {
        let centroid = detections.first().map(|detection| {
            let bounds = &detection.bounding_box;
            (
                (bounds.x as f32 + bounds.width as f32 / 2.0) / frame.width as f32,
                (bounds.y as f32 + bounds.height as f32 / 2.0) / frame.height as f32,
            )
        });

        self.centroids.push_back(centroid);
        if self.centroids.len() > TEMPORAL_WINDOW_FRAMES {
            self.centroids.pop_front();
        }

        let observed_frames = self.centroids.len();
        let warmed_up = observed_frames >= TEMPORAL_WINDOW_FRAMES;
        let visible_count = self.centroids.iter().flatten().count();
        let first_visible = self.centroids.iter().find_map(|centroid| *centroid);
        let last_visible = self.centroids.iter().rev().find_map(|centroid| *centroid);

        let (class_id, label, confidence) = match (first_visible, last_visible) {
            (Some(first), Some(last)) if visible_count >= 2 => {
                let delta_x = last.0 - first.0;
                let delta_y = last.1 - first.1;
                let magnitude = delta_x.hypot(delta_y);

                if magnitude < 0.04 {
                    (1, "stationary", (1.0 - magnitude / 0.04).clamp(0.5, 1.0))
                } else if delta_x.abs() >= delta_y.abs() && delta_x < 0.0 {
                    (2, "moving-left", (magnitude * 3.0).clamp(0.5, 1.0))
                } else if delta_x.abs() >= delta_y.abs() {
                    (3, "moving-right", (magnitude * 3.0).clamp(0.5, 1.0))
                } else if delta_y < 0.0 {
                    (4, "moving-up", (magnitude * 3.0).clamp(0.5, 1.0))
                } else {
                    (5, "moving-down", (magnitude * 3.0).clamp(0.5, 1.0))
                }
            }
            _ => (0, "no-foreground-object", 1.0),
        };

        ActionPrediction {
            class_id,
            label: label.to_string(),
            confidence,
            warmed_up,
            observed_frames,
        }
    }

    fn reset(&mut self) {
        self.centroids.clear();
    }
}

pub struct VisionBackend {
    detector: Box<dyn LowLevelDetector>,
    action_recognizer: Box<dyn TemporalActionRecognizer>,
    frame_dimensions: Option<(u32, u32)>,
    last_timestamp_ms: Option<u64>,
    frames_seen: u64,
}

impl Default for VisionBackend {
    fn default() -> Self {
        Self {
            detector: Box::<PrototypeForegroundDetector>::default(),
            action_recognizer: Box::<PrototypeActionRecognizer>::default(),
            frame_dimensions: None,
            last_timestamp_ms: None,
            frames_seen: 0,
        }
    }
}

impl VisionBackend {
    fn status(&self) -> VisionStatus {
        VisionStatus {
            ready: true,
            runtime: "prototype",
            detector: "frame-difference-foreground-prototype",
            action_recognizer: "centroid-motion-prototype",
            recommended_width: RECOMMENDED_WIDTH,
            recommended_height: RECOMMENDED_HEIGHT,
            temporal_window_frames: TEMPORAL_WINDOW_FRAMES,
            frames_seen: self.frames_seen,
            note: "The complete local pipeline is active, but no ML weights are loaded. Replace both prototype implementations with model runtimes before using semantic labels.",
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.action_recognizer.reset();
        self.frame_dimensions = None;
        self.last_timestamp_ms = None;
        self.frames_seen = 0;
    }

    fn process(&mut self, frame: VisionFrame) -> Result<VisionPrediction, String> {
        validate_frame(&frame)?;

        if let Some(dimensions) = self.frame_dimensions {
            if dimensions != (frame.width, frame.height) {
                return Err(format!(
                    "frame dimensions changed from {}x{} to {}x{}; reset the vision stream first",
                    dimensions.0, dimensions.1, frame.width, frame.height
                ));
            }
        }

        if let Some(previous_timestamp) = self.last_timestamp_ms {
            if frame.timestamp_ms < previous_timestamp {
                return Err("frame timestamps must be monotonically increasing".to_string());
            }
        }

        let detections = self.detector.detect(&frame)?;
        let action = self.action_recognizer.classify(&frame, &detections);
        let prediction = VisionPrediction {
            detections,
            action,
            frame_index: self.frames_seen,
            timestamp_ms: frame.timestamp_ms,
        };

        self.frame_dimensions = Some((frame.width, frame.height));
        self.last_timestamp_ms = Some(frame.timestamp_ms);
        self.frames_seen += 1;
        Ok(prediction)
    }
}

fn validate_frame(frame: &VisionFrame) -> Result<(), String> {
    if frame.width == 0 || frame.height == 0 {
        return Err("frame dimensions must be greater than zero".to_string());
    }

    let expected_len = frame
        .width
        .checked_mul(frame.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "frame dimensions are too large".to_string())?
        as usize;

    if frame.rgb.len() != expected_len {
        return Err(format!(
            "invalid RGB frame: expected {expected_len} bytes, received {}",
            frame.rgb.len()
        ));
    }

    Ok(())
}

fn rgb_to_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|pixel| {
            let red = u32::from(pixel[0]);
            let green = u32::from(pixel[1]);
            let blue = u32::from(pixel[2]);
            ((77 * red + 150 * green + 29 * blue) >> 8) as u8
        })
        .collect()
}

#[tauri::command]
pub fn vision_status(backend: State<'_, Mutex<VisionBackend>>) -> Result<VisionStatus, String> {
    let backend = backend
        .lock()
        .map_err(|_| "vision backend lock is poisoned".to_string())?;
    Ok(backend.status())
}

#[tauri::command]
pub fn vision_reset(backend: State<'_, Mutex<VisionBackend>>) -> Result<(), String> {
    let mut backend = backend
        .lock()
        .map_err(|_| "vision backend lock is poisoned".to_string())?;
    backend.reset();
    Ok(())
}

#[tauri::command]
pub fn vision_process_frame(
    backend: State<'_, Mutex<VisionBackend>>,
    frame: VisionFrame,
) -> Result<VisionPrediction, String> {
    let mut backend = backend
        .lock()
        .map_err(|_| "vision backend lock is poisoned".to_string())?;
    backend.process(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, rgb: Vec<u8>, timestamp_ms: u64) -> VisionFrame {
        VisionFrame {
            width,
            height,
            rgb,
            timestamp_ms,
        }
    }

    #[test]
    fn rejects_invalid_rgb_buffers() {
        let mut backend = VisionBackend::default();
        let result = backend.process(frame(2, 2, vec![0; 11], 0));

        assert!(result.is_err());
    }

    #[test]
    fn detects_a_changed_foreground_region() {
        let mut backend = VisionBackend::default();
        backend.process(frame(4, 4, vec![0; 48], 0)).unwrap();

        let mut changed = vec![0; 48];
        for pixel_index in [5_usize, 6, 9, 10] {
            changed[pixel_index * 3..pixel_index * 3 + 3].fill(255);
        }
        let prediction = backend.process(frame(4, 4, changed, 100)).unwrap();

        assert_eq!(prediction.detections.len(), 1);
        assert_eq!(prediction.detections[0].label, "foreground-region");
        assert_eq!(prediction.detections[0].bounding_box.x, 1);
        assert_eq!(prediction.detections[0].bounding_box.y, 1);
        assert_eq!(prediction.detections[0].bounding_box.width, 2);
        assert_eq!(prediction.detections[0].bounding_box.height, 2);
    }

    #[test]
    fn reset_clears_both_pipeline_stages() {
        let mut backend = VisionBackend::default();
        backend.process(frame(2, 2, vec![0; 12], 0)).unwrap();
        backend.reset();

        assert_eq!(backend.status().frames_seen, 0);
        assert!(backend.frame_dimensions.is_none());
        assert!(backend.last_timestamp_ms.is_none());
    }

    #[test]
    fn rejects_dimension_changes_without_a_reset() {
        let mut backend = VisionBackend::default();
        backend.process(frame(2, 2, vec![0; 12], 0)).unwrap();

        let result = backend.process(frame(3, 2, vec![0; 18], 100));

        assert!(result.is_err());
    }
}
