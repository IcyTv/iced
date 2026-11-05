//! GPU-resident immutable frames. No widget references or CPU pixel readback.
use crate::{
    Renderer,
    core::{Color, Size},
    graphics::{
        Viewport,
        compositor::{Frame, FramePresentation, SurfaceError},
    },
};
use wgpu::util::DeviceExt;

struct Snapshot {
    texture: wgpu::Texture,
    size: Size<f32>,
}

impl Renderer {
    pub(crate) fn capture_frame(&mut self, viewport: &Viewport, background: Color) -> Frame {
        let size = viewport.physical_size();
        let texture = self.engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced immutable frame"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.engine.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let _ = self.present(Some(background), self.engine.format, &view, viewport);
        Frame::new(Snapshot {
            texture,
            size: viewport.logical_size(),
        })
    }
}

pub(crate) struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    transform: [f32; 16],
    sizes: [f32; 4],
    clip: [f32; 4],
    opacity: [f32; 4],
}

impl Pipeline {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iced immutable frame"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iced immutable frame"),
            source: wgpu::ShaderSource::Wgsl(include_str!("snapshot.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iced immutable frame"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iced immutable frame"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("iced immutable frame"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
        }
    }

    pub(crate) fn present(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: Option<&Frame>,
        presentation: FramePresentation,
        surface: &wgpu::Surface<'static>,
        viewport: &Viewport,
        on_pre_present: impl FnOnce(),
    ) -> Result<(), SurfaceError> {
        let snapshot = frame
            .map(|frame| frame.downcast_ref::<Snapshot>().ok_or(SurfaceError::Other))
            .transpose()?;
        let surface_frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(_) | wgpu::CurrentSurfaceTexture::Outdated => {
                return Err(SurfaceError::Outdated);
            }
            wgpu::CurrentSurfaceTexture::Timeout => return Err(SurfaceError::Timeout),
            wgpu::CurrentSurfaceTexture::Occluded => return Err(SurfaceError::Occluded),
            wgpu::CurrentSurfaceTexture::Lost => return Err(SurfaceError::Lost),
            wgpu::CurrentSurfaceTexture::Validation => return Err(SurfaceError::Other),
        };
        let view = surface_frame.texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        let bindings = snapshot.map(|snapshot| {
            let logical = viewport.logical_size();
            let clip = presentation.clip;
            let uniforms = Uniforms {
                transform: presentation.transformation.into(),
                sizes: [
                    snapshot.size.width,
                    snapshot.size.height,
                    logical.width,
                    logical.height,
                ],
                clip: [clip.x, clip.y, clip.x + clip.width, clip.y + clip.height],
                opacity: [presentation.opacity.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
            };
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iced frame pose"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let texture_view = snapshot.texture.create_view(&Default::default());
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced immutable frame"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iced immutable frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(bindings) = &bindings {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bindings, &[]);
                pass.draw(0..6, 0..1);
            }
        }
        let _ = queue.submit([encoder.finish()]);
        on_pre_present();
        surface_frame.present();
        Ok(())
    }
}
