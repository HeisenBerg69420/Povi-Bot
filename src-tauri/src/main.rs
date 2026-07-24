#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod movinet;
mod vision;

use movinet::{movinet_classify_frame, movinet_reset, movinet_status, MovinetBackend};
use std::sync::Mutex;
use vision::{vision_process_frame, vision_reset, vision_status, VisionBackend};

fn main() {
    tauri::Builder::default()
        .manage(Mutex::new(MovinetBackend::default()))
        .manage(Mutex::new(VisionBackend::default()))
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
