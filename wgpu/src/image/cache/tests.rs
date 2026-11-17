use super::{Cache, Pending, load_image};
use crate::core::image::{Allocation, Error, Handle};
use crate::core::{Rectangle, Size, Transformation};
use crate::graphics::Shell;
use crate::graphics::shell::Notifier;
use crate::image::{Pipeline, State, raster};

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(10);

fn pending(handle: &Handle) -> Pending {
    Pending {
        handle: handle.clone(),
        callbacks: Vec::new(),
        layout: false,
        redraw: false,
    }
}

#[test]
fn pending_cold_image_skips_decode_and_records_demand() {
    let handle = Handle::from_bytes(vec![0]);
    let mut cache = raster::Cache::default();
    let mut pending = HashMap::from([(handle.id(), pending(&handle))]);

    assert!(load_image(&mut cache, &mut pending, &handle, true).is_none());
    assert!(load_image(&mut cache, &mut pending, &handle, false).is_none());

    assert!(!cache.contains(&handle));
    assert!(pending[&handle.id()].layout);
    assert!(pending[&handle.id()].redraw);
}

#[test]
fn pending_host_image_only_exposes_dimensions() {
    let handle = red_image(2, 3);
    let mut cache = raster::Cache::default();
    cache.insert(&handle, raster::Memory::load(&handle));
    let mut pending = HashMap::from([(handle.id(), pending(&handle))]);

    let dimensions = load_image(&mut cache, &mut pending, &handle, true)
        .unwrap()
        .dimensions();

    assert_eq!(dimensions, Size::new(2, 3));
    assert!(load_image(&mut cache, &mut pending, &handle, false).is_none());
    assert!(matches!(
        cache.get_mut(&handle),
        Some(raster::Memory::Host(_))
    ));
}

#[test]
fn ordinary_image_decode_stays_synchronous_without_pending_work() {
    let handle = red_image(2, 3);
    let mut cache = raster::Cache::default();
    let mut pending = HashMap::new();

    let memory = load_image(&mut cache, &mut pending, &handle, true).unwrap();

    assert_eq!(memory.dimensions(), Size::new(2, 3));
    assert!(matches!(memory, raster::Memory::Host(_)));
    assert!(pending.is_empty());
}

#[derive(Clone)]
struct Notifications {
    ticks: mpsc::Sender<()>,
    layouts: Arc<AtomicUsize>,
    redraws: Arc<AtomicUsize>,
}

impl Notifier for Notifications {
    fn tick(&self) {
        let _ = self.ticks.send(());
    }

    fn invalidate_layout(&self) {
        let _ = self.layouts.fetch_add(1, Ordering::SeqCst);
    }

    fn request_redraw(&self) {
        let _ = self.redraws.fetch_add(1, Ordering::SeqCst);
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: wgpu::Backend,
}

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();

    GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            ..Default::default()
        }))
        .expect(
            "image cache GPU tests require a surface-free adapter; adapter absence is not a pass",
        );
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("image cache tests"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .expect("image cache GPU tests require a device");

        Gpu {
            device,
            queue,
            backend: adapter.get_info().backend,
        }
    })
}

fn block_on<T>(future: impl Future<Output = T>) -> T {
    struct Thread(thread::Thread);

    impl Wake for Thread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(Thread(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

fn setup() -> (Cache, Pipeline, Notifications, mpsc::Receiver<()>) {
    let gpu = gpu();
    let (ticks, receiver) = mpsc::channel();
    let notifications = Notifications {
        ticks,
        layouts: Arc::default(),
        redraws: Arc::default(),
    };
    let pipeline = Pipeline::new(&gpu.device, format(), gpu.backend);
    let cache = pipeline.create_cache(&gpu.device, &gpu.queue, &Shell::new(notifications.clone()));

    (cache, pipeline, notifications, receiver)
}

fn format() -> wgpu::TextureFormat {
    if crate::graphics::color::GAMMA_CORRECTION {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

fn red_image(width: u32, height: u32) -> Handle {
    Handle::from_rgba(
        width,
        height,
        [255, 0, 0, 255].repeat((width * height) as usize),
    )
}

fn request(cache: &mut Cache, handle: &Handle) -> mpsc::Receiver<Result<Allocation, Error>> {
    let (sender, receiver) = mpsc::channel();
    let gpu = gpu();
    cache.allocate_image(&gpu.device, &gpu.queue, handle, move |result| {
        sender.send(result).unwrap();
    });

    receiver
}

fn assert_same_allocation(left: &Allocation, right: &Allocation) {
    assert!(std::sync::Weak::ptr_eq(
        &left.downgrade(),
        &right.downgrade()
    ));
}

#[test]
fn cold_dispatch_is_gated_deduplicated_draw_ready_and_retained() {
    let gpu = gpu();
    let (mut cache, pipeline, notifications, ticks) = setup();
    let (entered, resume) = cache.worker.gate();
    entered.recv_timeout(TIMEOUT).unwrap();
    let handle = red_image(2, 2);

    let first = request(&mut cache, &handle);
    let second = request(&mut cache, &handle);
    assert!(matches!(first.try_recv(), Err(mpsc::TryRecvError::Empty)));
    assert!(matches!(second.try_recv(), Err(mpsc::TryRecvError::Empty)));
    assert_eq!(cache.raster.pending.len(), 1);
    assert_eq!(cache.raster.pending[&handle.id()].callbacks.len(), 2);
    assert_eq!(cache.measure_image(&handle), None);

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut belt = wgpu::util::StagingBelt::new(gpu.device.clone(), 1024);
    assert!(
        cache
            .upload_raster(&gpu.device, &mut encoder, &mut belt, &handle)
            .is_none()
    );
    assert!(!cache.raster.cache.contains(&handle));

    resume.send(()).unwrap();
    ticks.recv_timeout(TIMEOUT).unwrap();
    cache.receive();

    let first = first.recv_timeout(TIMEOUT).unwrap().unwrap();
    let second = second.recv_timeout(TIMEOUT).unwrap().unwrap();
    assert_same_allocation(&first, &second);
    assert_eq!(first.size(), Size::new(2, 2));
    assert!(cache.raster.pending.is_empty());
    assert_eq!(notifications.layouts.load(Ordering::SeqCst), 1);
    assert_eq!(notifications.redraws.load(Ordering::SeqCst), 1);
    assert_same_allocation(
        &first,
        &request(&mut cache, &handle).try_recv().unwrap().unwrap(),
    );

    // Force repeated trim passes with unrelated insertions and no image cache hits.
    for _ in 0..3 {
        let other = Handle::from_bytes(vec![0]);
        cache
            .raster
            .cache
            .insert(&other, raster::Memory::Error(Error::Empty));
        cache.trim();
    }

    assert!(
        matches!(cache.raster.cache.get_mut(&handle), Some(raster::Memory::Device {
        allocation: Some(reference), ..
    }) if reference.strong_count() >= 2)
    );
    assert_red_draw(&mut cache, &pipeline, &handle);
    assert!(matches!(ticks.try_recv(), Err(mpsc::TryRecvError::Empty)));
}

#[test]
fn sync_load_joins_pending_ordinary_upload_and_allocation_callbacks() {
    let gpu = gpu();
    let (mut cache, _pipeline, _notifications, ticks) = setup();
    let (entered, resume) = cache.worker.gate();
    entered.recv_timeout(TIMEOUT).unwrap();
    let handle = red_image(1024, 512);
    assert_eq!(cache.measure_image(&handle), Some(Size::new(1024, 512)));

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut belt = wgpu::util::StagingBelt::new(gpu.device.clone(), 1024);
    assert!(
        cache
            .upload_raster(&gpu.device, &mut encoder, &mut belt, &handle)
            .is_none()
    );
    let callback = request(&mut cache, &handle);
    assert_eq!(cache.measure_image(&handle), Some(Size::new(1024, 512)));
    assert!(
        cache
            .upload_raster(&gpu.device, &mut encoder, &mut belt, &handle)
            .is_none()
    );
    assert_eq!(cache.raster.pending.len(), 1);

    resume.send(()).unwrap();
    let joined = cache.load_image(&gpu.device, &gpu.queue, &handle).unwrap();
    let callback = callback.recv_timeout(TIMEOUT).unwrap().unwrap();

    assert_same_allocation(&joined, &callback);
    assert_eq!(joined.size(), Size::new(1024, 512));
    assert!(cache.raster.pending.is_empty());
    ticks.recv_timeout(TIMEOUT).unwrap();
    cache.worker.quit();
    assert!(matches!(
        cache.worker.try_recv(),
        Err(mpsc::TryRecvError::Disconnected)
    ));
}

#[test]
fn terminal_errors_wake_and_resolve_callbacks_exactly_once() {
    let (mut cache, _pipeline, notifications, ticks) = setup();
    let (entered, resume) = cache.worker.gate();
    entered.recv_timeout(TIMEOUT).unwrap();
    let invalid = Handle::from_bytes(vec![0]);
    let empty = Handle::from_rgba(0, 0, Vec::new());

    let first = request(&mut cache, &invalid);
    let duplicate = request(&mut cache, &invalid);
    let empty_result = request(&mut cache, &empty);
    assert_eq!(cache.measure_image(&invalid), None);
    resume.send(()).unwrap();

    for _ in 0..2 {
        ticks.recv_timeout(TIMEOUT).unwrap();
    }
    cache.receive();
    cache.receive();

    assert!(first.recv_timeout(TIMEOUT).unwrap().is_err());
    assert!(duplicate.recv_timeout(TIMEOUT).unwrap().is_err());
    assert!(matches!(
        empty_result.recv_timeout(TIMEOUT).unwrap(),
        Err(Error::Empty)
    ));
    assert!(matches!(
        first.try_recv(),
        Err(mpsc::TryRecvError::Disconnected)
    ));
    assert!(matches!(
        duplicate.try_recv(),
        Err(mpsc::TryRecvError::Disconnected)
    ));
    assert!(cache.raster.pending.is_empty());
    assert_eq!(notifications.layouts.load(Ordering::SeqCst), 1);
    assert!(request(&mut cache, &invalid).try_recv().unwrap().is_err());
    assert!(matches!(ticks.try_recv(), Err(mpsc::TryRecvError::Empty)));
}

#[test]
fn disconnected_worker_resolves_pending_join_and_new_requests() {
    let gpu = gpu();
    let (mut cache, _pipeline, _notifications, _ticks) = setup();
    let (entered, resume) = cache.worker.gate();
    entered.recv_timeout(TIMEOUT).unwrap();
    let handle = red_image(2, 2);
    let result = request(&mut cache, &handle);

    // Signal quit before releasing the gate, so queued loads cannot run.
    cache.worker.quit_gated(resume);
    let joined = cache.load_image(&gpu.device, &gpu.queue, &handle);

    assert!(matches!(joined, Err(Error::Invalid(_))));
    assert!(matches!(
        result.recv_timeout(TIMEOUT).unwrap(),
        Err(Error::Invalid(_))
    ));
    assert!(cache.raster.pending.is_empty());
    assert!(matches!(
        request(&mut cache, &red_image(1, 1)).try_recv().unwrap(),
        Err(Error::Invalid(_))
    ));
    cache.receive();
}

#[test]
fn job_and_result_channels_do_not_block_past_old_capacity() {
    let gpu = gpu();
    let (mut cache, _pipeline, _notifications, ticks) = setup();
    let (entered, resume) = cache.worker.gate();
    entered.recv_timeout(TIMEOUT).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    const COUNT: usize = 1_100;

    for _ in 0..COUNT {
        let calls = calls.clone();
        cache.allocate_image(
            &gpu.device,
            &gpu.queue,
            &Handle::from_bytes(vec![0]),
            move |result| {
                assert!(result.is_err());
                let _ = calls.fetch_add(1, Ordering::SeqCst);
            },
        );
    }

    assert_eq!(cache.raster.pending.len(), COUNT);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    resume.send(()).unwrap();

    // No receives until every worker result has been published.
    for _ in 0..COUNT {
        ticks.recv_timeout(TIMEOUT).unwrap();
    }
    cache.receive();

    assert!(cache.raster.pending.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), COUNT);
}

fn assert_red_draw(cache: &mut Cache, pipeline: &Pipeline, handle: &Handle) {
    let gpu = gpu();
    let extent = wgpu::Extent3d {
        width: 2,
        height: 2,
        depth_or_array_layers: 1,
    };
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("image cache draw target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("image cache readback"),
        size: 512,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut state = State::new();
    let mut belt = wgpu::util::StagingBelt::new(gpu.device.clone(), 1024);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let bounds = Rectangle::with_size(Size::new(2.0, 2.0));

    state.prepare(
        pipeline,
        &gpu.device,
        &mut belt,
        &mut encoder,
        cache,
        &vec![crate::image::Image::Raster {
            image: crate::core::image::Image::new(handle.clone()),
            bounds,
            clip_bounds: bounds,
        }],
        Transformation::orthographic(2, 2),
        1.0,
    );

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        state.render(
            pipeline,
            0,
            Rectangle::with_size(Size::new(2, 2)),
            &mut pass,
        );
    }

    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        extent,
    );
    belt.finish();
    let submission = gpu.queue.submit([encoder.finish()]);
    belt.recall();
    let (sender, receiver) = mpsc::channel();
    output
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    let _ = gpu
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(TIMEOUT),
        })
        .unwrap();
    receiver.recv_timeout(TIMEOUT).unwrap().unwrap();

    let pixels = output.slice(..).get_mapped_range();
    for row in [0, 256] {
        assert_eq!(&pixels[row..row + 8], &[255, 0, 0, 255, 255, 0, 0, 255]);
    }
}
