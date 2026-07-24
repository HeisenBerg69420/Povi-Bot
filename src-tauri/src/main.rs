#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod movinet;
mod vision;

use movinet::{movinet_classify_frame, movinet_reset, movinet_status, MovinetBackend};
use std::sync::Mutex;
use tauri::Manager;
use vision::{vision_process_frame, vision_reset, vision_status, VisionBackend};

fn main() {
    tauri::Builder::default()
        .manage(Mutex::new(MovinetBackend::default()))
        .setup(|app| {
            let resource_dir = app
                .path()
                .resource_dir()
                .map_err(|error| format!("failed to resolve resource directory: {error}"))?;
            let model_path = resource_dir.join("models").join("PoviNet.onnx");
            let labels_path = resource_dir.join("models").join("PoviNet.labels.txt");
            let backend = VisionBackend::with_onnx_action_model(&model_path, &labels_path)?;
            app.manage(Mutex::new(backend));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            movinet_status,
            movinet_reset,
            movinet_classify_frame,
            vision_status,
            vision_reset,
            vision_process_frame
        ])
        .run(tauri::generate_context!())
        .expect("failed to run desktop pet overlay");
}
