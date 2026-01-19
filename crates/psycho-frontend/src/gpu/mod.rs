//! WebGPU rendering system.

pub mod pipeline;
pub mod renderer;
pub mod skeletal;
pub mod trajectory;

pub use pipeline::*;
pub use renderer::*;
pub use skeletal::*;
pub use trajectory::*;

use thiserror::Error;
use wgpu::*;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("WebGPU not supported")]
    NotSupported,

    #[error("Failed to request adapter")]
    AdapterRequest,

    #[error("Failed to request device: {0}")]
    DeviceRequest(String),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),
}

pub type GpuResult<T> = Result<T, GpuError>;

/// WebGPU context
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface<'static>,
    pub surface_config: SurfaceConfiguration,
}

impl GpuContext {
    /// Initialize WebGPU context
    pub async fn new(canvas: web_sys::HtmlCanvasElement) -> GpuResult<Self> {
        // Create instance
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        // Create surface from canvas
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|_| GpuError::NotSupported)?;

        // Request adapter
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::AdapterRequest)?;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Psycho-DensePose Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_webgl2_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        // Configure surface
        let width = canvas.width();
        let height = canvas.height();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
        })
    }

    /// Resize surface
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Get current frame
    pub fn get_current_frame(&self) -> Result<SurfaceTexture, SurfaceError> {
        self.surface.get_current_texture()
    }
}
