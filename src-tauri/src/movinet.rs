use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

const EXPECTED_WIDTH: u32 = 172;
const EXPECTED_HEIGHT: u32 = 172;
const SMOOTHING: f32 = 0.7;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MovinetFrame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub timestamp_ms: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MovinetPrediction {
    pub label: String,
    pub confidence: f32,
    pub frame_index: u64,
    pub warmed_up: bool,
    pub timestamp_ms: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MovinetStatus {
    pub ready: bool,
    pub runtime: &'static str,
    pub model: &'static str,
    pub expected_width: u32,
    pub expected_height: u32,
    pub frames_seen: u64,
    pub note: &'static str,
}

#[derive(Debug, Default)]
pub struct MovinetBackend {
    previous_luma: Option<Vec<u8>>,
    frame_dimensions: Option<(u32, u32)>,
    smoothed_motion: f32,
    frames_seen: u64,
    last_timestamp_ms: Option<u64>,
}

impl MovinetBackend {
    fn status(&self) -> MovinetStatus {
        MovinetStatus {
            ready: true,
            runtime: "demo",
            model: "movinet-compatible-motion-prototype",
            expected_width: EXPECTED_WIDTH,
            expected_height: EXPECTED_HEIGHT,
            frames_seen: self.frames_seen,
            note: "Streaming API is active; replace the demo runtime with MoViNet A0 weights for Kinetics predictions.",
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn classify(&mut self, frame: MovinetFrame) -> Result<MovinetPrediction, String> {
        validate_frame(&frame)?;

        if let Some((width, height)) = self.frame_dimensions {
            if frame.width != width || frame.height != height {
                return Err(format!(
                    "frame dimensions changed from {width}x{height} to {}x{}; reset the stream first",
                    frame.width, frame.height
                ));
            }
        }

        if let Some(previous_timestamp) = self.last_timestamp_ms {
            if frame.timestamp_ms < previous_timestamp {
                return Err("frame timestamps must be monotonically increasing".to_string());
            }
        }

        let luma = rgb_to_luma(&frame.rgb);
        let motion = self
            .previous_luma
            .as_ref()
            .map(|previous| mean_absolute_difference(previous, &luma))
            .unwrap_or(0.0);

        self.smoothed_motion = if self.frames_seen == 0 {
            motion
        } else {
            SMOOTHING * self.smoothed_motion + (1.0 - SMOOTHING) * motion
        };
        self.previous_luma = Some(luma);
        self.frame_dimensions = Some((frame.width, frame.height));
        self.frames_seen += 1;
        self.last_timestamp_ms = Some(frame.timestamp_ms);

        let (label, confidence) = motion_label(self.smoothed_motion);
        Ok(MovinetPrediction {
            label: label.to_string(),
            confidence,
            frame_index: self.frames_seen - 1,
            warmed_up: self.frames_seen >= 3,
            timestamp_ms: frame.timestamp_ms,
        })
    }
}

fn validate_frame(frame: &MovinetFrame) -> Result<(), String> {
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

fn mean_absolute_difference(previous: &[u8], current: &[u8]) -> f32 {
    let total: u64 = previous
        .iter()
        .zip(current)
        .map(|(left, right)| u64::from(left.abs_diff(*right)))
        .sum();
    total as f32 / previous.len() as f32 / 255.0
}

fn motion_label(motion: f32) -> (&'static str, f32) {
    if motion < 0.025 {
        ("no significant motion", 1.0 - (motion / 0.025).min(1.0))
    } else if motion < 0.12 {
        (
            "gentle movement",
            ((motion - 0.025) / 0.095).clamp(0.35, 1.0),
        )
    } else {
        ("vigorous movement", (motion / 0.25).clamp(0.5, 1.0))
    }
}

#[tauri::command]
pub fn movinet_status(backend: State<'_, Mutex<MovinetBackend>>) -> Result<MovinetStatus, String> {
    let backend = backend
        .lock()
        .map_err(|_| "MoViNet backend lock is poisoned".to_string())?;
    Ok(backend.status())
}

#[tauri::command]
pub fn movinet_reset(backend: State<'_, Mutex<MovinetBackend>>) -> Result<(), String> {
    let mut backend = backend
        .lock()
        .map_err(|_| "MoViNet backend lock is poisoned".to_string())?;
    backend.reset();
    Ok(())
}

#[tauri::command]
pub fn movinet_classify_frame(
    backend: State<'_, Mutex<MovinetBackend>>,
    frame: MovinetFrame,
) -> Result<MovinetPrediction, String> {
    let mut backend = backend
        .lock()
        .map_err(|_| "MoViNet backend lock is poisoned".to_string())?;
    backend.classify(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(value: u8, timestamp_ms: u64) -> MovinetFrame {
        MovinetFrame {
            width: 2,
            height: 2,
            rgb: vec![value; 12],
            timestamp_ms,
        }
    }

    #[test]
    fn rejects_an_invalid_rgb_buffer() {
        let mut backend = MovinetBackend::default();
        let frame = MovinetFrame {
            width: 2,
            height: 2,
            rgb: vec![0; 11],
            timestamp_ms: 0,
        };

        assert!(backend.classify(frame).is_err());
    }

    #[test]
    fn keeps_streaming_state_between_frames() {
        let mut backend = MovinetBackend::default();
        let first = backend.classify(solid_frame(0, 0)).unwrap();
        let second = backend.classify(solid_frame(255, 100)).unwrap();

        assert_eq!(first.frame_index, 0);
        assert_eq!(second.frame_index, 1);
        assert_eq!(second.label, "vigorous movement");
    }

    #[test]
    fn reset_clears_the_streaming_state() {
        let mut backend = MovinetBackend::default();
        backend.classify(solid_frame(0, 0)).unwrap();
        backend.reset();

        assert_eq!(backend.status().frames_seen, 0);
        assert!(backend.previous_luma.is_none());
        assert!(backend.frame_dimensions.is_none());
    }

    #[test]
    fn rejects_zero_sized_frames() {
        let mut backend = MovinetBackend::default();
        let result = backend.classify(MovinetFrame {
            width: 0,
            height: 2,
            rgb: Vec::new(),
            timestamp_ms: 0,
        });

        assert_eq!(
            result.unwrap_err(),
            "frame dimensions must be greater than zero"
        );
        assert_eq!(backend.status().frames_seen, 0);
    }

    #[test]
    fn rejects_dimension_changes_without_mutating_stream_state() {
        let mut backend = MovinetBackend::default();
        backend.classify(solid_frame(0, 100)).unwrap();

        let result = backend.classify(MovinetFrame {
            width: 3,
            height: 2,
            rgb: vec![0; 18],
            timestamp_ms: 200,
        });

        assert!(result.unwrap_err().contains("reset the stream first"));
        assert_eq!(backend.status().frames_seen, 1);
        assert_eq!(backend.last_timestamp_ms, Some(100));
    }

    #[test]
    fn rejects_timestamps_that_move_backwards() {
        let mut backend = MovinetBackend::default();
        backend.classify(solid_frame(0, 100)).unwrap();

        let result = backend.classify(solid_frame(0, 99));

        assert_eq!(
            result.unwrap_err(),
            "frame timestamps must be monotonically increasing"
        );
        assert_eq!(backend.status().frames_seen, 1);
    }

    #[test]
    fn reports_warmup_after_three_frames() {
        let mut backend = MovinetBackend::default();

        assert!(!backend.classify(solid_frame(0, 0)).unwrap().warmed_up);
        assert!(!backend.classify(solid_frame(0, 1)).unwrap().warmed_up);
        assert!(backend.classify(solid_frame(0, 2)).unwrap().warmed_up);
    }
}
