//! # Psycho-Frontend
//!
//! Leptos-based WebGPU frontend for Psycho-DensePose system.
//!
//! ## Features
//!
//! - **3D Skeletal Visualization**: Real-time skeletal pose rendering with WebGPU
//! - **Trajectory Heatmaps**: GPU-accelerated movement path visualization
//! - **Sales Dashboard**: Live intelligence display with OCEAN scores
//! - **Admin Panel**: System health monitoring and privacy controls
//! - **WebTransport Client**: Bi-directional streaming with low latency

pub mod app;
pub mod components;
pub mod gpu;
pub mod network;
pub mod pages;
pub mod shaders;
pub mod state;
pub mod utils;

pub use app::App;

use wasm_bindgen::prelude::*;

/// Initialize the application
#[wasm_bindgen(start)]
pub fn main() {
    // Set panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize tracing
    tracing_wasm::set_as_global_default();

    tracing::info!("Psycho-DensePose Frontend initialized");
}
