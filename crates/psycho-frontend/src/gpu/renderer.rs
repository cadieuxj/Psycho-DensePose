//! Main rendering coordinator.

use super::{GpuContext, SkeletalRenderer, TrajectoryRenderer};

pub struct Renderer {
    pub skeletal: SkeletalRenderer,
    pub trajectory: TrajectoryRenderer,
}

impl Renderer {
    pub fn new(ctx: &GpuContext) -> Result<Self, super::GpuError> {
        Ok(Self {
            skeletal: SkeletalRenderer::new(ctx)?,
            trajectory: TrajectoryRenderer::new(ctx, ctx.surface_config.width, ctx.surface_config.height)?,
        })
    }
}
