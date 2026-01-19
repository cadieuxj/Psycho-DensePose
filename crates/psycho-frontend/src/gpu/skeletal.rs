//! Skeletal pose rendering with WebGPU.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use psycho_core::{Keypoint, SkeletalPose};
use wgpu::*;

use super::{GpuContext, GpuResult};

/// Vertex for skeletal rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkeletalVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl SkeletalVertex {
    const ATTRIBS: [VertexAttribute; 2] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x4];

    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Skeletal bone connections (COCO keypoint pairs)
const SKELETON_BONES: &[(Keypoint, Keypoint)] = &[
    // Torso
    (Keypoint::Neck, Keypoint::RightShoulder),
    (Keypoint::Neck, Keypoint::LeftShoulder),
    (Keypoint::RightShoulder, Keypoint::RightElbow),
    (Keypoint::RightElbow, Keypoint::RightWrist),
    (Keypoint::LeftShoulder, Keypoint::LeftElbow),
    (Keypoint::LeftElbow, Keypoint::LeftWrist),
    // Legs
    (Keypoint::RightHip, Keypoint::RightKnee),
    (Keypoint::RightKnee, Keypoint::RightAnkle),
    (Keypoint::LeftHip, Keypoint::LeftKnee),
    (Keypoint::LeftKnee, Keypoint::LeftAnkle),
    // Head
    (Keypoint::Nose, Keypoint::Neck),
    (Keypoint::Nose, Keypoint::RightEye),
    (Keypoint::Nose, Keypoint::LeftEye),
    (Keypoint::RightEye, Keypoint::RightEar),
    (Keypoint::LeftEye, Keypoint::LeftEar),
];

/// Skeletal renderer
pub struct SkeletalRenderer {
    pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    uniform_buffer: Buffer,
    bind_group: BindGroup,
    num_indices: u32,
}

impl SkeletalRenderer {
    pub fn new(ctx: &GpuContext) -> GpuResult<Self> {
        // Create shader module
        let shader = ctx
            .device
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("Skeletal Shader"),
                source: ShaderSource::Wgsl(include_str!("../../shaders/skeletal.wgsl").into()),
            });

        // Create uniform buffer for camera matrix
        let uniform_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<Mat4>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Skeletal Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Skeletal Bind Group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create pipeline layout
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Skeletal Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create render pipeline
        let pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Skeletal Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[SkeletalVertex::desc()],
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(ColorTargetState {
                        format: ctx.surface_config.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        // Create placeholder buffers
        let vertex_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("Skeletal Vertex Buffer"),
            size: 1024 * std::mem::size_of::<SkeletalVertex>() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("Skeletal Index Buffer"),
            size: 1024 * std::mem::size_of::<u16>() as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            bind_group,
            num_indices: 0,
        })
    }

    /// Update skeletal data from pose
    pub fn update(&mut self, ctx: &GpuContext, pose: &SkeletalPose, camera_matrix: Mat4) {
        // Build vertices and indices for bones
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for (start_kp, end_kp) in SKELETON_BONES {
            if let (Some(start), Some(end)) = (
                pose.keypoints[*start_kp as usize].as_ref(),
                pose.keypoints[*end_kp as usize].as_ref(),
            ) {
                let start_idx = vertices.len() as u16;

                // Add start vertex
                vertices.push(SkeletalVertex {
                    position: [
                        start.position.x as f32,
                        start.position.y as f32,
                        start.position.z as f32,
                    ],
                    color: [0.0, 1.0, 0.5, start.confidence as f32],
                });

                // Add end vertex
                vertices.push(SkeletalVertex {
                    position: [
                        end.position.x as f32,
                        end.position.y as f32,
                        end.position.z as f32,
                    ],
                    color: [0.0, 1.0, 0.5, end.confidence as f32],
                });

                // Add line indices
                indices.push(start_idx);
                indices.push(start_idx + 1);
            }
        }

        // Upload vertex data
        if !vertices.is_empty() {
            ctx.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            ctx.queue
                .write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
            self.num_indices = indices.len() as u32;
        }

        // Upload camera matrix
        ctx.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[camera_matrix]),
        );
    }

    /// Render skeletal pose
    pub fn render(&self, encoder: &mut CommandEncoder, view: &TextureView) {
        if self.num_indices == 0 {
            return;
        }

        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Skeletal Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}
