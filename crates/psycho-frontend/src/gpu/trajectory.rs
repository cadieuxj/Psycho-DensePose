//! Trajectory heatmap rendering with compute shaders.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use psycho_core::Trajectory;
use wgpu::*;

use super::{GpuContext, GpuResult};

/// Trajectory point for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuTrajectoryPoint {
    position: [f32; 2],
    timestamp: f32,
    weight: f32,
}

/// Heatmap compute parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct HeatmapParams {
    width: u32,
    height: u32,
    sigma: f32,
    max_intensity: f32,
}

/// Trajectory heatmap renderer using compute shaders
pub struct TrajectoryRenderer {
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    point_buffer: Buffer,
    heatmap_texture: Texture,
    params_buffer: Buffer,
    compute_bind_group: BindGroup,
    render_bind_group: BindGroup,
    num_points: u32,
}

impl TrajectoryRenderer {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> GpuResult<Self> {
        // Create compute shader for heatmap generation
        let compute_shader = ctx
            .device
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("Heatmap Compute Shader"),
                source: ShaderSource::Wgsl(include_str!("../../shaders/heatmap.wgsl").into()),
            });

        // Create render shader for displaying heatmap
        let render_shader = ctx
            .device
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("Heatmap Render Shader"),
                source: ShaderSource::Wgsl(include_str!("../../shaders/heatmap_display.wgsl").into()),
            });

        // Create heatmap texture (R32Float for accumulation)
        let heatmap_texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("Heatmap Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create buffers
        let point_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("Trajectory Point Buffer"),
            size: 10000 * std::mem::size_of::<GpuTrajectoryPoint>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("Heatmap Params Buffer"),
            size: std::mem::size_of::<HeatmapParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compute bind group layout
        let compute_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Compute Bind Group Layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::WriteOnly,
                                format: TextureFormat::R32Float,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let heatmap_view = heatmap_texture.create_view(&TextureViewDescriptor::default());

        let compute_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: point_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&heatmap_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compute pipeline
        let compute_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&compute_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Heatmap Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "main",
            });

        // Render bind group layout
        let render_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Render Bind Group Layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: false },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

        let sampler = ctx.device.create_sampler(&SamplerDescriptor {
            label: Some("Heatmap Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let render_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&heatmap_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Render pipeline
        let render_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&render_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Heatmap Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: VertexState {
                    module: &render_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: &render_shader,
                    entry_point: "fs_main",
                    targets: &[Some(ColorTargetState {
                        format: ctx.surface_config.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview: None,
            });

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            point_buffer,
            heatmap_texture,
            params_buffer,
            compute_bind_group,
            render_bind_group,
            num_points: 0,
        })
    }

    /// Update trajectory data
    pub fn update(&mut self, ctx: &GpuContext, trajectory: &Trajectory) {
        let mut gpu_points = Vec::new();

        for point in &trajectory.points {
            gpu_points.push(GpuTrajectoryPoint {
                position: [point.position.x as f32, point.position.y as f32],
                timestamp: point.timestamp.as_secs_f64() as f32,
                weight: 1.0,
            });
        }

        if !gpu_points.is_empty() {
            ctx.queue
                .write_buffer(&self.point_buffer, 0, bytemuck::cast_slice(&gpu_points));
            self.num_points = gpu_points.len() as u32;
        }

        // Update parameters
        let params = HeatmapParams {
            width: self.heatmap_texture.width(),
            height: self.heatmap_texture.height(),
            sigma: 0.5,
            max_intensity: 1.0,
        };
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Compute and render heatmap
    pub fn render(&self, encoder: &mut CommandEncoder, view: &TextureView) {
        if self.num_points == 0 {
            return;
        }

        // Compute pass - generate heatmap
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Heatmap Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            let workgroup_size = 16;
            let width = self.heatmap_texture.width();
            let height = self.heatmap_texture.height();
            let dispatch_x = (width + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (height + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Render pass - display heatmap
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Heatmap Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
    }
}
