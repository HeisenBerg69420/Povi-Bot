use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;
use tauri::State;

const RECOMMENDED_WIDTH: u32 = 320;
const RECOMMENDED_HEIGHT: u32 = 240;
const FOREGROUND_LUMA_THRESHOLD: u8 = 28;
const TEMPORAL_WINDOW_FRAMES: usize = 8;
const MODEL_WIDTH: usize = 256;
const MODEL_HEIGHT: usize = 256;
const PREDICTION_WINDOW_FRAMES: usize = 5;
const MODEL_INPUT_NAME: &str = "images";
const MODEL_OUTPUT_NAME: &str = "logits";

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
    fn classify(
        &mut self,
        frame: &VisionFrame,
        detections: &[ObjectDetection],
    ) -> Result<ActionPrediction, String>;
    fn reset(&mut self);
}

#[derive(Default)]
struct PrototypeForegroundDetector {
    previous_luma: Option<Vec<u8>>,
    last_detections: Vec<ObjectDetection>,
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
            return Ok(self.last_detections.clone());
        }

        let changed_ratio = changed_pixels as f32 / pixel_count as f32;
        let detections = vec![ObjectDetection {
            class_id: 0,
            label: "foreground-region".to_string(),
            confidence: (changed_ratio * 8.0).clamp(0.1, 1.0),
            bounding_box: BoundingBox {
                x: minimum_x,
                y: minimum_y,
                width: maximum_x - minimum_x + 1,
                height: maximum_y - minimum_y + 1,
            },
        }];
        self.last_detections.clone_from(&detections);
        Ok(detections)
    }

    fn reset(&mut self) {
        self.previous_luma = None;
        self.last_detections.clear();
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
    ) -> Result<ActionPrediction, String> {
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

        Ok(ActionPrediction {
            class_id,
            label: label.to_string(),
            confidence,
            warmed_up,
            observed_frames,
        })
    }

    fn reset(&mut self) {
        self.centroids.clear();
    }
}

struct OnnxActionRecognizer {
    session: Session,
    labels: Vec<String>,
    probability_history: VecDeque<Vec<f32>>,
}

impl OnnxActionRecognizer {
    fn load(model_path: &Path, labels_path: &Path) -> Result<Self, String> {
        let labels = read_labels(labels_path)?;
        if labels.is_empty() {
            return Err("the action label file must contain at least one label".to_string());
        }

        let session = Session::builder()
            .map_err(|error| format!("failed to create ONNX session builder: {error}"))?
            .commit_from_file(model_path)
            .map_err(|error| {
                format!(
                    "failed to load ONNX action model '{}': {error}",
                    model_path.display()
                )
            })?;

        Ok(Self {
            session,
            labels,
            probability_history: VecDeque::with_capacity(PREDICTION_WINDOW_FRAMES),
        })
    }

    fn infer_probabilities(&mut self, frame: &VisionFrame) -> Result<Vec<f32>, String> {
        let input_values = preprocess_rgb_frame(frame);
        let input = Tensor::from_array((
            [1_usize, 3, MODEL_HEIGHT, MODEL_WIDTH],
            input_values.into_boxed_slice(),
        ))
        .map_err(|error| format!("failed to create ONNX input tensor: {error}"))?;

        let outputs = self
            .session
            .run(ort::inputs![MODEL_INPUT_NAME => input])
            .map_err(|error| format!("ONNX action inference failed: {error}"))?;
        let logits = outputs[MODEL_OUTPUT_NAME]
            .try_extract_array::<f32>()
            .map_err(|error| format!("failed to read ONNX logits: {error}"))?;
        let logits: Vec<f32> = logits.iter().copied().collect();

        if logits.len() != self.labels.len() {
            return Err(format!(
                "ONNX model returned {} logits, but {} labels are configured",
                logits.len(),
                self.labels.len()
            ));
        }

        softmax(&logits)
    }
}

impl TemporalActionRecognizer for OnnxActionRecognizer {
    fn classify(
        &mut self,
        frame: &VisionFrame,
        _detections: &[ObjectDetection],
    ) -> Result<ActionPrediction, String> {
        let probabilities = self.infer_probabilities(frame)?;
        self.probability_history.push_back(probabilities);
        if self.probability_history.len() > PREDICTION_WINDOW_FRAMES {
            self.probability_history.pop_front();
        }

        let averaged_probabilities =
            average_probabilities(&self.probability_history, self.labels.len())?;
        let (class_id, confidence) = maximum_probability(&averaged_probabilities)?;
        let observed_frames = self.probability_history.len();

        Ok(ActionPrediction {
            class_id: class_id as u32,
            label: self.labels[class_id].clone(),
            confidence,
            warmed_up: observed_frames >= PREDICTION_WINDOW_FRAMES,
            observed_frames,
        })
    }

    fn reset(&mut self) {
        self.probability_history.clear();
    }
}

pub struct VisionBackend {
    detector: Box<dyn LowLevelDetector>,
    action_recognizer: Box<dyn TemporalActionRecognizer>,
    frame_dimensions: Option<(u32, u32)>,
    last_timestamp_ms: Option<u64>,
    frames_seen: u64,
    runtime: &'static str,
    action_recognizer_name: &'static str,
    note: &'static str,
}

impl Default for VisionBackend {
    fn default() -> Self {
        Self {
            detector: Box::<PrototypeForegroundDetector>::default(),
            action_recognizer: Box::<PrototypeActionRecognizer>::default(),
            frame_dimensions: None,
            last_timestamp_ms: None,
            frames_seen: 0,
            runtime: "prototype",
            action_recognizer_name: "centroid-motion-prototype",
            note: "The local pipeline is active, but no ML weights are loaded.",
        }
    }
}

impl VisionBackend {
    pub fn with_onnx_action_model(model_path: &Path, labels_path: &Path) -> Result<Self, String> {
        Ok(Self {
            detector: Box::<PrototypeForegroundDetector>::default(),
            action_recognizer: Box::new(OnnxActionRecognizer::load(model_path, labels_path)?),
            frame_dimensions: None,
            last_timestamp_ms: None,
            frames_seen: 0,
            runtime: "onnx-runtime",
            action_recognizer_name: "PoviNet-image-classifier",
            note: "PoviNet classifies each complete camera frame; predictions are averaged across five frames.",
        })
    }

    fn status(&self) -> VisionStatus {
        VisionStatus {
            ready: true,
            runtime: self.runtime,
            detector: "frame-difference-foreground-prototype",
            action_recognizer: self.action_recognizer_name,
            recommended_width: RECOMMENDED_WIDTH,
            recommended_height: RECOMMENDED_HEIGHT,
            temporal_window_frames: if self.runtime == "onnx-runtime" {
                PREDICTION_WINDOW_FRAMES
            } else {
                TEMPORAL_WINDOW_FRAMES
            },
            frames_seen: self.frames_seen,
            note: self.note,
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
        let action = self.action_recognizer.classify(&frame, &detections)?;
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

fn read_labels(path: &Path) -> Result<Vec<String>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read action labels '{}': {error}", path.display()))?;
    Ok(contents
        .lines()
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(str::to_string)
        .collect())
}

fn preprocess_rgb_frame(frame: &VisionFrame) -> Vec<f32> {
    let mut input = vec![0.0_f32; 3 * MODEL_WIDTH * MODEL_HEIGHT];
    let source_width = frame.width as usize;
    let source_height = frame.height as usize;

    for target_y in 0..MODEL_HEIGHT {
        let source_y = ((target_y as f32 + 0.5) * source_height as f32 / MODEL_HEIGHT as f32 - 0.5)
            .clamp(0.0, (source_height - 1) as f32);
        let top = source_y.floor() as usize;
        let bottom = (top + 1).min(source_height - 1);
        let vertical_weight = source_y - top as f32;

        for target_x in 0..MODEL_WIDTH {
            let source_x = ((target_x as f32 + 0.5) * source_width as f32 / MODEL_WIDTH as f32
                - 0.5)
                .clamp(0.0, (source_width - 1) as f32);
            let left = source_x.floor() as usize;
            let right = (left + 1).min(source_width - 1);
            let horizontal_weight = source_x - left as f32;
            let target_index = target_y * MODEL_WIDTH + target_x;

            for channel in 0..3 {
                let top_left = frame.rgb[(top * source_width + left) * 3 + channel] as f32;
                let top_right = frame.rgb[(top * source_width + right) * 3 + channel] as f32;
                let bottom_left = frame.rgb[(bottom * source_width + left) * 3 + channel] as f32;
                let bottom_right = frame.rgb[(bottom * source_width + right) * 3 + channel] as f32;
                let top_value = top_left + (top_right - top_left) * horizontal_weight;
                let bottom_value = bottom_left + (bottom_right - bottom_left) * horizontal_weight;
                let pixel = top_value + (bottom_value - top_value) * vertical_weight;

                input[channel * MODEL_WIDTH * MODEL_HEIGHT + target_index] = pixel / 127.5 - 1.0;
            }
        }
    }

    input
}

fn softmax(logits: &[f32]) -> Result<Vec<f32>, String> {
    if logits.is_empty() {
        return Err("ONNX model returned no logits".to_string());
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("ONNX model returned a non-finite logit".to_string());
    }

    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let sum: f32 = exponentials.iter().sum();
    Ok(exponentials.into_iter().map(|value| value / sum).collect())
}

fn average_probabilities(
    history: &VecDeque<Vec<f32>>,
    class_count: usize,
) -> Result<Vec<f32>, String> {
    if history.is_empty() {
        return Err("cannot average an empty prediction history".to_string());
    }

    let mut averages = vec![0.0_f32; class_count];
    for probabilities in history {
        if probabilities.len() != class_count {
            return Err("prediction history contains incompatible class counts".to_string());
        }
        for (average, probability) in averages.iter_mut().zip(probabilities) {
            *average += probability;
        }
    }
    for average in &mut averages {
        *average /= history.len() as f32;
    }
    Ok(averages)
}

fn maximum_probability(probabilities: &[f32]) -> Result<(usize, f32), String> {
    probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .ok_or_else(|| "model returned no class probabilities".to_string())
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
    fn keeps_the_last_foreground_detection_when_too_few_pixels_change() {
        let mut backend = VisionBackend::default();
        backend.process(frame(4, 4, vec![0; 48], 0)).unwrap();

        let mut changed = vec![0; 48];
        for pixel_index in [5_usize, 6, 9, 10] {
            changed[pixel_index * 3..pixel_index * 3 + 3].fill(255);
        }
        let detected = backend.process(frame(4, 4, changed.clone(), 100)).unwrap();
        let unchanged = backend.process(frame(4, 4, changed, 200)).unwrap();

        assert_eq!(unchanged.detections.len(), 1);
        assert_eq!(
            unchanged.detections[0].bounding_box.x,
            detected.detections[0].bounding_box.x
        );
        assert_eq!(
            unchanged.detections[0].bounding_box.y,
            detected.detections[0].bounding_box.y
        );
        assert_eq!(
            unchanged.detections[0].bounding_box.width,
            detected.detections[0].bounding_box.width
        );
        assert_eq!(
            unchanged.detections[0].bounding_box.height,
            detected.detections[0].bounding_box.height
        );
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

    #[test]
    fn rejects_zero_sized_frames_without_advancing_state() {
        let mut backend = VisionBackend::default();

        let result = backend.process(frame(0, 2, Vec::new(), 0));

        assert_eq!(
            result.unwrap_err(),
            "frame dimensions must be greater than zero"
        );
        assert_eq!(backend.status().frames_seen, 0);
    }

    #[test]
    fn rejects_timestamps_that_move_backwards() {
        let mut backend = VisionBackend::default();
        backend.process(frame(2, 2, vec![0; 12], 100)).unwrap();

        let result = backend.process(frame(2, 2, vec![0; 12], 99));

        assert_eq!(
            result.unwrap_err(),
            "frame timestamps must be monotonically increasing"
        );
        assert_eq!(backend.status().frames_seen, 1);
        assert_eq!(backend.last_timestamp_ms, Some(100));
    }

    #[test]
    fn action_recognizer_tracks_rightward_motion_and_warmup() {
        let mut recognizer = PrototypeActionRecognizer::default();
        let test_frame = frame(100, 100, vec![0; 30_000], 0);

        let prediction = (0..TEMPORAL_WINDOW_FRAMES)
            .map(|index| ObjectDetection {
                class_id: 0,
                label: "foreground-region".to_string(),
                confidence: 1.0,
                bounding_box: BoundingBox {
                    x: (index * 10) as u32,
                    y: 40,
                    width: 10,
                    height: 10,
                },
            })
            .map(|detection| recognizer.classify(&test_frame, &[detection]).unwrap())
            .last()
            .unwrap();

        assert_eq!(prediction.label, "moving-right");
        assert!(prediction.warmed_up);
        assert_eq!(prediction.observed_frames, TEMPORAL_WINDOW_FRAMES);
    }

    #[test]
    fn reset_allows_new_dimensions_and_timestamps() {
        let mut backend = VisionBackend::default();
        backend.process(frame(2, 2, vec![0; 12], 100)).unwrap();
        backend.reset();

        let prediction = backend
            .process(frame(3, 1, vec![0; 9], 10))
            .expect("a reset starts an independent stream");

        assert_eq!(prediction.frame_index, 0);
        assert_eq!(prediction.timestamp_ms, 10);
    }

    #[test]
    fn preprocessing_converts_rgb_to_normalized_nchw() {
        let input = preprocess_rgb_frame(&frame(1, 1, vec![0, 128, 255], 0));
        let plane_size = MODEL_WIDTH * MODEL_HEIGHT;

        assert_eq!(input.len(), plane_size * 3);
        assert_eq!(input[0], -1.0);
        assert!((input[plane_size] - (128.0 / 127.5 - 1.0)).abs() < f32::EPSILON);
        assert_eq!(input[plane_size * 2], 1.0);
    }

    #[test]
    fn softmax_is_stable_and_sums_to_one() {
        let probabilities = softmax(&[1_000.0, 1_001.0, 1_002.0]).unwrap();
        let sum: f32 = probabilities.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(maximum_probability(&probabilities).unwrap().0, 2);
    }

    #[test]
    fn averaging_smooths_probabilities_across_frames() {
        let history = VecDeque::from([vec![0.8, 0.2], vec![0.6, 0.4], vec![0.1, 0.9]]);
        let averaged = average_probabilities(&history, 2).unwrap();

        assert!((averaged[0] - 0.5).abs() < 1e-6);
        assert!((averaged[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires the local, untracked PoviNet ONNX model"]
    fn loads_local_onnx_model_and_classifies_a_frame() {
        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
        let mut recognizer = OnnxActionRecognizer::load(
            &models_dir.join("PoviNet.onnx"),
            &models_dir.join("PoviNet.labels.txt"),
        )
        .unwrap();

        let prediction = recognizer
            .classify(&frame(320, 240, vec![0; 320 * 240 * 3], 0), &[])
            .unwrap();

        assert!(recognizer.labels.contains(&prediction.label));
        assert!(prediction.confidence.is_finite());
        assert!((0.0..=1.0).contains(&prediction.confidence));
    }
}
