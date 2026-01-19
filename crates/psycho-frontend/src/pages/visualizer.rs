//! 3D visualization page with WebGPU rendering.

use leptos::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use crate::gpu::{GpuContext, SkeletalRenderer, TrajectoryRenderer};

#[component]
pub fn Visualizer() -> impl IntoView {
    let canvas_ref = create_node_ref::<html::Canvas>();

    // Initialize WebGPU on mount
    create_effect(move |_| {
        if let Some(canvas_elem) = canvas_ref.get() {
            let canvas: HtmlCanvasElement = canvas_elem.unchecked_into();

            spawn_local(async move {
                match initialize_webgpu(canvas).await {
                    Ok(_) => tracing::info!("WebGPU initialized successfully"),
                    Err(e) => tracing::error!("WebGPU initialization failed: {:?}", e),
                }
            });
        }
    });

    view! {
        <div class="visualizer">
            <div class="visualizer-header">
                <h1>"3D Skeletal Tracking"</h1>
                <div class="controls">
                    <button class="control-btn">"Reset Camera"</button>
                    <button class="control-btn">"Toggle Heatmap"</button>
                    <button class="control-btn">"Pause"</button>
                </div>
            </div>

            <div class="visualizer-container">
                <canvas
                    node_ref=canvas_ref
                    class="webgpu-canvas"
                    width="1920"
                    height="1080"
                >
                    "Your browser does not support WebGPU"
                </canvas>

                <div class="overlay-info">
                    <div class="info-panel">
                        <h3>"Active Subjects"</h3>
                        <p>"3 tracked"</p>
                    </div>
                    <div class="info-panel">
                        <h3>"Frame Rate"</h3>
                        <p>"60 FPS"</p>
                    </div>
                    <div class="info-panel">
                        <h3>"Latency"</h3>
                        <p>"12ms"</p>
                    </div>
                </div>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="color-box skeleton"></div>
                    <span>"Skeletal Pose"</span>
                </div>
                <div class="legend-item">
                    <div class="color-box trajectory"></div>
                    <span>"Movement Path"</span>
                </div>
                <div class="legend-item">
                    <div class="color-box heatmap"></div>
                    <span>"Activity Heatmap"</span>
                </div>
            </div>
        </div>
    }
}

async fn initialize_webgpu(canvas: HtmlCanvasElement) -> Result<(), String> {
    use crate::gpu::GpuError;

    // Initialize GPU context
    let ctx = GpuContext::new(canvas)
        .await
        .map_err(|e| format!("Failed to initialize WebGPU: {:?}", e))?;

    // Create renderers
    let skeletal_renderer = SkeletalRenderer::new(&ctx)
        .map_err(|e| format!("Failed to create skeletal renderer: {:?}", e))?;

    let trajectory_renderer = TrajectoryRenderer::new(&ctx, 1920, 1080)
        .map_err(|e| format!("Failed to create trajectory renderer: {:?}", e))?;

    tracing::info!("WebGPU renderers created successfully");

    // TODO: Start render loop with requestAnimationFrame
    // TODO: Connect to WebTransport stream for live pose data

    Ok(())
}
