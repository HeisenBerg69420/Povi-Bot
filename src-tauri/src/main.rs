#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod movinet;
mod vision;

use movinet::{movinet_classify_frame, movinet_reset, movinet_status, MovinetBackend};
use std::sync::Mutex;
use vision::{vision_process_frame, vision_reset, vision_status, VisionBackend};

#[tauri::command]
fn get_mouse_position() -> Result<(i32, i32), String> {
    match mouse_position::mouse_position::Mouse::get_mouse_position() {
        mouse_position::mouse_position::Mouse::Position { x, y } => Ok((x, y)),
        mouse_position::mouse_position::Mouse::Error => {
            Err("Mausposition konnte nicht gelesen werden".into())
        }
    }
}

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
            vision_process_frame,
            get_mouse_position
        ])
        .run(tauri::generate_context!())
        .expect("failed to run desktop pet overlay");
}
