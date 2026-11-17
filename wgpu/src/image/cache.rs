use crate::core::{self, Size};
use crate::graphics::Shell;
use crate::image::atlas::{self, Atlas};

#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
use worker::Worker;

#[cfg(feature = "image")]
use std::collections::HashMap;

use std::sync::Arc;

#[cfg(all(test, feature = "image", not(target_arch = "wasm32")))]
mod tests;

pub struct Cache {
    atlas: Atlas,
    #[cfg(feature = "image")]
    raster: Raster,
    #[cfg(feature = "svg")]
    vector: crate::image::vector::Cache,
    #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
    worker: Worker,
}

impl Cache {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        backend: wgpu::Backend,
        layout: wgpu::BindGroupLayout,
        _shell: &Shell,
    ) -> Self {
        #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
        let worker = Worker::new(device, _queue, backend, layout.clone(), _shell);

        log::debug!("iced_wgpu image cache: creating cache backend={backend:?}");

        Self {
            atlas: Atlas::new(device, backend, layout),
            #[cfg(feature = "image")]
            raster: Raster {
                cache: crate::image::raster::Cache::default(),
                pending: HashMap::new(),
                belt: wgpu::util::StagingBelt::new(device.clone(), 2 * 1024 * 1024),
            },
            #[cfg(feature = "svg")]
            vector: crate::image::vector::Cache::default(),
            #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
            worker,
        }
    }

    #[cfg(feature = "image")]
    pub fn allocate_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: &core::image::Handle,
        callback: impl FnOnce(Result<core::image::Allocation, core::image::Error>) + Send + 'static,
    ) {
        use crate::image::raster::Memory;

        if let Some(Memory::Device {
            allocation, entry, ..
        }) = self.raster.cache.get_mut(handle)
        {
            if let Some(allocation) = allocation
                .as_ref()
                .and_then(core::image::Allocation::upgrade)
            {
                callback(Ok(allocation));
                return;
            }

            #[allow(unsafe_code)]
            let new = unsafe { core::image::allocate(handle, entry.size()) };
            *allocation = Some(new.downgrade());
            callback(Ok(new));

            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (device, queue);

            if let Some(pending) = self.raster.pending.get_mut(&handle.id()) {
                pending.callbacks.push(Box::new(callback));
                return;
            }

            if let Some(Memory::Error(error)) = self.raster.cache.get_mut(handle) {
                callback(Err(error.clone()));
                return;
            }

            let _ = self.raster.pending.insert(
                handle.id(),
                Pending {
                    handle: handle.clone(),
                    callbacks: vec![Box::new(callback)],
                    layout: false,
                    redraw: false,
                },
            );

            let result = match self.raster.cache.get_mut(handle) {
                Some(Memory::Host(image)) => self.worker.upload(handle, image.clone()),
                _ => self.worker.load(handle),
            };

            if let Err(error) = result {
                self.apply(worker::Work::Error {
                    handle: handle.clone(),
                    error,
                });
            }
        }

        #[cfg(target_arch = "wasm32")]
        callback(self.load_image(device, queue, handle));
    }

    #[cfg(feature = "image")]
    pub fn load_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: &core::image::Handle,
    ) -> Result<core::image::Allocation, core::image::Error> {
        use crate::image::raster::Memory;

        #[cfg(not(target_arch = "wasm32"))]
        while self.raster.pending.contains_key(&handle.id()) {
            match self.worker.recv() {
                Ok(work) => self.apply(work),
                Err(_) => self.disconnected(),
            }
        }

        if !self.raster.cache.contains(handle) {
            self.raster.cache.insert(handle, Memory::load(handle));
        }

        match self.raster.cache.get_mut(handle).unwrap() {
            Memory::Host(image) => {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("raster image upload"),
                });

                let entry = self.atlas.upload(
                    device,
                    &mut encoder,
                    &mut self.raster.belt,
                    image.width(),
                    image.height(),
                    image,
                );

                self.raster.belt.finish();
                let submission = queue.submit([encoder.finish()]);
                self.raster.belt.recall();

                let Some(entry) = entry else {
                    return Err(core::image::Error::OutOfMemory);
                };

                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                });

                #[allow(unsafe_code)]
                let allocation = unsafe {
                    core::image::allocate(handle, Size::new(image.width(), image.height()))
                };

                self.raster.cache.insert(
                    handle,
                    Memory::Device {
                        entry,
                        bind_group: None,
                        allocation: Some(allocation.downgrade()),
                    },
                );

                Ok(allocation)
            }
            Memory::Device {
                entry, allocation, ..
            } => {
                if let Some(allocation) = allocation
                    .as_ref()
                    .and_then(core::image::Allocation::upgrade)
                {
                    return Ok(allocation);
                }

                #[allow(unsafe_code)]
                let new = unsafe { core::image::allocate(handle, entry.size()) };

                *allocation = Some(new.downgrade());

                Ok(new)
            }
            Memory::Error(error) => Err(error.clone()),
        }
    }

    #[cfg(feature = "image")]
    pub fn measure_image(&mut self, handle: &core::image::Handle) -> Option<Size<u32>> {
        self.receive();

        let image = load_image(
            &mut self.raster.cache,
            &mut self.raster.pending,
            handle,
            true,
        )?;

        Some(image.dimensions())
    }

    #[cfg(feature = "svg")]
    pub fn measure_svg(&mut self, handle: &core::svg::Handle) -> Size<u32> {
        // TODO: Concurrency
        self.vector.load(handle).viewport_dimensions()
    }

    #[cfg(feature = "image")]
    pub fn upload_raster(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        handle: &core::image::Handle,
    ) -> Option<(&atlas::Entry, &Arc<wgpu::BindGroup>)> {
        use crate::image::raster::Memory;

        self.receive();

        let image = load_image(
            &mut self.raster.cache,
            &mut self.raster.pending,
            handle,
            false,
        )?
        .host();

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(image) = &image {
            const MAX_SYNC_SIZE: usize = 2 * 1024 * 1024;

            if image.len() >= MAX_SYNC_SIZE {
                let _ = self.raster.pending.insert(
                    handle.id(),
                    Pending {
                        handle: handle.clone(),
                        callbacks: Vec::new(),
                        layout: false,
                        redraw: true,
                    },
                );

                if let Err(error) = self.worker.upload(handle, image.clone()) {
                    self.apply(worker::Work::Error {
                        handle: handle.clone(),
                        error,
                    });
                }

                return None;
            }
        }

        let memory = self.raster.cache.get_mut(handle)?;

        if let Some(image) = image {
            let entry =
                self.atlas
                    .upload(device, encoder, belt, image.width(), image.height(), &image)?;

            *memory = Memory::Device {
                entry,
                bind_group: None,
                allocation: None,
            };
        }

        if let Memory::Device {
            entry, bind_group, ..
        } = memory
        {
            Some((
                entry,
                bind_group.as_ref().unwrap_or(self.atlas.bind_group()),
            ))
        } else {
            None
        }
    }

    #[cfg(feature = "svg")]
    pub fn upload_vector(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        handle: &core::svg::Handle,
        color: Option<core::Color>,
        size: Size<u32>,
    ) -> Option<(&atlas::Entry, &Arc<wgpu::BindGroup>)> {
        // TODO: Concurrency
        self.vector
            .upload(device, encoder, belt, handle, color, size, &mut self.atlas)
            .map(|entry| (entry, self.atlas.bind_group()))
    }

    pub fn trim(&mut self) {
        #[cfg(feature = "image")]
        {
            self.receive();
            self.raster.cache.trim(&mut self.atlas, |_bind_group| {
                #[cfg(not(target_arch = "wasm32"))]
                self.worker.drop(_bind_group);
            });
        }

        #[cfg(feature = "svg")]
        self.vector.trim(&mut self.atlas); // TODO: Concurrency
    }

    pub fn clear_old_textures(&mut self) {
        self.atlas.clear_old_textures();
    }

    #[cfg(feature = "image")]
    pub fn receive(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        loop {
            match self.worker.try_recv() {
                Ok(work) => self.apply(work),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.disconnected();
                    break;
                }
            }
        }
    }

    #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
    fn apply(&mut self, work: worker::Work) {
        use crate::image::raster::Memory;

        let handle = match &work {
            worker::Work::Upload { handle, .. } | worker::Work::Error { handle, .. } => handle,
        };
        let Some(pending) = self.raster.pending.remove(&handle.id()) else {
            return;
        };

        let result = match work {
            worker::Work::Upload {
                handle,
                entry,
                bind_group,
            } => {
                let allocation = if pending.callbacks.is_empty() {
                    None
                } else {
                    // The worker only publishes uploads after their GPU submission completes.
                    #[allow(unsafe_code)]
                    Some(unsafe { core::image::allocate(&handle, entry.size()) })
                };

                self.raster.cache.insert(
                    &handle,
                    Memory::Device {
                        entry,
                        bind_group: Some(bind_group),
                        allocation: allocation.as_ref().map(core::image::Allocation::downgrade),
                    },
                );

                Ok(allocation)
            }
            worker::Work::Error { handle, error } => {
                self.raster
                    .cache
                    .insert(&handle, Memory::Error(error.clone()));

                Err(error)
            }
        };

        for callback in pending.callbacks {
            callback(match &result {
                Ok(allocation) => Ok(allocation.as_ref().unwrap().clone()),
                Err(error) => Err(error.clone()),
            });
        }

        self.worker.notify(pending.layout, pending.redraw);
    }

    #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
    fn disconnected(&mut self) {
        let handles: Vec<_> = self
            .raster
            .pending
            .values()
            .map(|pending| pending.handle.clone())
            .collect();

        for handle in handles {
            self.apply(worker::Work::Error {
                handle,
                error: worker::disconnected(),
            });
        }
    }
}

#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
impl Drop for Cache {
    fn drop(&mut self) {
        self.worker.quit();
    }
}

#[cfg(feature = "image")]
struct Raster {
    cache: crate::image::raster::Cache,
    pending: HashMap<core::image::Id, Pending>,
    belt: wgpu::util::StagingBelt,
}

#[cfg(feature = "image")]
type Callback = Box<dyn FnOnce(Result<core::image::Allocation, core::image::Error>) + Send>;

#[cfg(feature = "image")]
struct Pending {
    handle: core::image::Handle,
    callbacks: Vec<Callback>,
    layout: bool,
    redraw: bool,
}

#[cfg(feature = "image")]
fn load_image<'a>(
    cache: &'a mut crate::image::raster::Cache,
    pending: &mut HashMap<core::image::Id, Pending>,
    handle: &core::image::Handle,
    measure: bool,
) -> Option<&'a mut crate::image::raster::Memory> {
    use crate::image::raster::Memory;

    if let Some(pending) = pending.get_mut(&handle.id()) {
        if measure {
            pending.layout = true;

            // A previously decoded host image can still answer layout queries.
            return cache.get_mut(handle);
        }

        pending.redraw = true;
        return None;
    }

    if !cache.contains(handle) {
        cache.insert(handle, Memory::load(handle));
    }

    cache.get_mut(handle)
}

#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
mod worker {
    use crate::core::Bytes;
    use crate::core::image;
    use crate::graphics::Shell;
    use crate::image::atlas::{self, Atlas};
    use crate::image::raster;

    use std::sync::Arc;
    use std::sync::mpsc;
    use std::thread;

    pub struct Worker {
        jobs: mpsc::Sender<Job>,
        quit: mpsc::Sender<()>,
        work: mpsc::Receiver<Work>,
        shell: Shell,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl Worker {
        pub fn new(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            backend: wgpu::Backend,
            texture_layout: wgpu::BindGroupLayout,
            shell: &Shell,
        ) -> Self {
            let (jobs_sender, jobs_receiver) = mpsc::channel();
            let (quit_sender, quit_receiver) = mpsc::channel();
            let (work_sender, work_receiver) = mpsc::channel();

            let instance = Instance {
                device: device.clone(),
                queue: queue.clone(),
                backend,
                texture_layout,
                shell: shell.clone(),
                belt: wgpu::util::StagingBelt::new(device.clone(), 4 * 1024 * 1024),
                jobs: jobs_receiver,
                output: work_sender,
                quit: quit_receiver,
            };

            let shell_on_exit = shell.clone();
            let handle = thread::spawn(move || {
                let _wake_on_exit = WakeOnExit(shell_on_exit);
                instance.run();
            });

            Self {
                jobs: jobs_sender,
                quit: quit_sender,
                work: work_receiver,
                shell: shell.clone(),
                handle: Some(handle),
            }
        }

        pub fn load(&self, handle: &image::Handle) -> Result<(), image::Error> {
            self.jobs
                .send(Job::Load(handle.clone()))
                .map_err(|_| disconnected())
        }

        pub fn upload(
            &self,
            handle: &image::Handle,
            image: raster::Image,
        ) -> Result<(), image::Error> {
            self.jobs
                .send(Job::Upload {
                    handle: handle.clone(),
                    width: image.width(),
                    height: image.height(),
                    rgba: image.into_raw(),
                })
                .map_err(|_| disconnected())
        }

        pub fn drop(&self, bind_group: Arc<wgpu::BindGroup>) {
            let _ = self.jobs.send(Job::Drop(bind_group));
        }

        pub fn try_recv(&self) -> Result<Work, mpsc::TryRecvError> {
            self.work.try_recv()
        }

        pub fn recv(&self) -> Result<Work, mpsc::RecvError> {
            self.work.recv()
        }

        #[cfg(test)]
        pub fn gate(&self) -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
            let (entered, ready) = mpsc::channel();
            let (resume, wait) = mpsc::channel();
            self.jobs
                .send(Job::Gate {
                    entered,
                    resume: wait,
                })
                .unwrap();

            (ready, resume)
        }

        pub fn notify(&self, layout: bool, redraw: bool) {
            if layout {
                self.shell.invalidate_layout();
            }

            if redraw {
                self.shell.request_redraw();
            }
        }

        #[cfg(test)]
        pub fn quit_gated(&mut self, resume: mpsc::Sender<()>) {
            let _ = self.quit.send(());
            drop(resume);
            self.quit();
        }

        pub fn quit(&mut self) {
            let _ = self.quit.send(());
            let _ = self.jobs.send(Job::Quit);
            let _ = self.handle.take().map(thread::JoinHandle::join);
        }
    }

    struct WakeOnExit(Shell);

    impl Drop for WakeOnExit {
        fn drop(&mut self) {
            // The result sender has already dropped, including when the worker unwinds.
            self.0.tick();
        }
    }

    pub fn disconnected() -> image::Error {
        image::Error::Invalid(Arc::new(std::io::Error::other("image worker disconnected")))
    }

    pub struct Instance {
        device: wgpu::Device,
        queue: wgpu::Queue,
        backend: wgpu::Backend,
        texture_layout: wgpu::BindGroupLayout,
        shell: Shell,
        belt: wgpu::util::StagingBelt,
        jobs: mpsc::Receiver<Job>,
        output: mpsc::Sender<Work>,
        quit: mpsc::Receiver<()>,
    }

    #[derive(Debug)]
    enum Job {
        Load(image::Handle),
        Upload {
            handle: image::Handle,
            rgba: Bytes,
            width: u32,
            height: u32,
        },
        Drop(Arc<wgpu::BindGroup>),
        #[cfg(test)]
        Gate {
            entered: mpsc::Sender<()>,
            resume: mpsc::Receiver<()>,
        },
        Quit,
    }

    pub enum Work {
        Upload {
            handle: image::Handle,
            entry: atlas::Entry,
            bind_group: Arc<wgpu::BindGroup>,
        },
        Error {
            handle: image::Handle,
            error: image::Error,
        },
    }

    impl Instance {
        fn run(mut self) {
            loop {
                if self.quit.try_recv().is_ok() {
                    return;
                }

                let Ok(job) = self.jobs.recv() else {
                    return;
                };

                match job {
                    Job::Load(handle) => match crate::graphics::image::load(&handle) {
                        Ok(image) => {
                            self.upload(handle, image.width(), image.height(), image.into_raw())
                        }
                        Err(error) => self.publish(Work::Error { handle, error }),
                    },
                    Job::Upload {
                        handle,
                        rgba,
                        width,
                        height,
                    } => {
                        self.upload(handle, width, height, rgba);
                    }
                    Job::Drop(bind_group) => {
                        drop(bind_group);
                    }
                    #[cfg(test)]
                    Job::Gate { entered, resume } => {
                        let _ = entered.send(());
                        let _ = resume.recv();
                    }
                    Job::Quit => return,
                }
            }
        }

        fn publish(&self, work: Work) {
            if self.output.send(work).is_ok() {
                self.shell.tick();
            }
        }

        fn upload(&mut self, handle: image::Handle, width: u32, height: u32, rgba: Bytes) {
            if width == 0 || height == 0 {
                self.publish(Work::Error {
                    handle,
                    error: image::Error::Empty,
                });
                return;
            }

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("raster image upload"),
                });

            let mut atlas = Atlas::with_size(
                &self.device,
                self.backend,
                self.texture_layout.clone(),
                width.max(height),
            );

            let Some(entry) = atlas.upload(
                &self.device,
                &mut encoder,
                &mut self.belt,
                width,
                height,
                &rgba,
            ) else {
                // A fragmented upload may already have staged pixels before allocation fails.
                self.belt.finish();
                let _ = self.queue.submit([encoder.finish()]);
                self.belt.recall();

                self.publish(Work::Error {
                    handle,
                    error: image::Error::OutOfMemory,
                });
                return;
            };

            self.belt.finish();
            let submission = self.queue.submit([encoder.finish()]);
            self.belt.recall();

            let work = match self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            }) {
                Ok(_) => Work::Upload {
                    handle,
                    entry,
                    bind_group: atlas.bind_group().clone(),
                },
                Err(error) => Work::Error {
                    handle,
                    error: image::Error::Invalid(Arc::new(error)),
                },
            };

            self.publish(work);
        }
    }
}
