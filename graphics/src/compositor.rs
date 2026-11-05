//! A compositor is responsible for initializing a renderer and managing window
//! surfaces.
use crate::core::Color;
use crate::core::backend;
use crate::core::font;
use crate::core::renderer;
use crate::futures::{MaybeSend, MaybeSync};
use crate::{Shell, Viewport};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;

use std::borrow::Cow;
use std::fmt::Debug;
use std::{any::Any, sync::Arc};

/// Immutable rendered content, owned independently of its renderer and widgets.
#[derive(Clone)]
pub struct Frame(Arc<dyn Any + Send + Sync>);

impl Frame {
    /// Wraps a backend-owned frame. The payload must retain all render resources.
    pub fn new(payload: impl Any + Send + Sync) -> Self {
        Self(Arc::new(payload))
    }
    /// Obtains the payload belonging to a specific compositor implementation.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}

#[cfg(test)]
mod frame_tests {
    use super::Frame;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Resources(Arc<AtomicUsize>);
    impl Drop for Resources {
        fn drop(&mut self) {
            let _ = self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn retained_frames_own_resources_after_the_source_releases_them() {
        let drops = Arc::new(AtomicUsize::new(0));
        let source = Frame::new(Resources(drops.clone()));
        let exit = source.clone();
        drop(source);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(exit.downcast_ref::<Resources>().is_some());
        drop(exit);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

/// Presentation of an immutable frame. Geometry is in application-logical units.
#[derive(Debug, Clone, Copy)]
pub struct FramePresentation {
    /// Maps source content into the target's application-logical coordinates.
    pub transformation: crate::core::Transformation,
    /// Multiplies premultiplied RGB and alpha, clamped to zero through one.
    pub opacity: f32,
    /// Clips transformed content in target application-logical coordinates.
    pub clip: crate::core::Rectangle,
}

/// A graphics compositor that can draw to windows.
pub trait Compositor: Sized {
    /// The iced renderer of the backend.
    type Renderer;

    /// The surface of the backend.
    type Surface;

    /// Whether this compositor can retain and replay immutable rendered frames.
    fn supports_frames(&self) -> bool {
        false
    }

    /// Captures the current primitives without borrowing the widget tree.
    /// GPU implementations must not read pixels back to the CPU.
    fn capture_frame(
        &mut self,
        _renderer: &mut Self::Renderer,
        _viewport: &Viewport,
        _background_color: Color,
    ) -> Option<Frame> {
        None
    }

    /// Presents retained content, or clears to transparent when `frame` is None.
    /// Implementations call `on_pre_present` only after successful acquisition.
    fn present_frame(
        &mut self,
        _frame: Option<&Frame>,
        _presentation: FramePresentation,
        _surface: &mut Self::Surface,
        _viewport: &Viewport,
        _on_pre_present: impl FnOnce(),
    ) -> Result<(), SurfaceError> {
        Err(SurfaceError::Other)
    }

    /// Creates a new [`Compositor`] with the given [`backend::Settings`].
    fn new(
        settings: backend::Settings,
        display: impl Display + Clone,
        compatible_window: impl Window + Clone,
        shell: Shell,
    ) -> impl Future<Output = Result<Self, backend::Error>>;

    /// Creates a [`Self::Renderer`] for the [`Compositor`].
    fn create_renderer(&self, settings: renderer::Settings) -> Self::Renderer;

    /// Crates a new [`Surface`] for the given window.
    ///
    /// [`Surface`]: Self::Surface
    fn create_surface(
        &mut self,
        window: impl Window + Clone,
        width: u32,
        height: u32,
    ) -> Self::Surface;

    /// Configures a new [`Surface`] with the given dimensions.
    ///
    /// [`Surface`]: Self::Surface
    fn configure_surface(&mut self, surface: &mut Self::Surface, width: u32, height: u32);

    /// Returns [`Information`] used by this [`Compositor`].
    fn information(&self) -> Information;

    /// Loads a font from its bytes.
    fn load_font(&mut self, font: Cow<'static, [u8]>) -> Result<(), font::Error> {
        crate::text::font_system()
            .write()
            .expect("Write to font system")
            .load_font(font);

        // TODO: Error handling
        Ok(())
    }

    /// Lists all the available font families.
    fn list_fonts(&mut self) -> Result<Vec<font::Family>, font::Error> {
        use std::collections::BTreeSet;

        let font_system = crate::text::font_system()
            .read()
            .expect("Read from font system");

        let families = BTreeSet::from_iter(font_system.families());

        Ok(families.into_iter().map(font::Family::name).collect())
    }

    /// Presents the [`Renderer`] primitives to the next frame of the given [`Surface`].
    ///
    /// [`Renderer`]: Self::Renderer
    /// [`Surface`]: Self::Surface
    fn present(
        &mut self,
        renderer: &mut Self::Renderer,
        surface: &mut Self::Surface,
        viewport: &Viewport,
        background_color: Color,
        on_pre_present: impl FnOnce(),
    ) -> Result<(), SurfaceError>;

    /// Screenshots the current [`Renderer`] primitives to an offscreen texture, and returns the bytes of
    /// the texture ordered as `RGBA` in the `sRGB` color space.
    ///
    /// [`Renderer`]: Self::Renderer
    fn screenshot(
        &mut self,
        renderer: &mut Self::Renderer,
        viewport: &Viewport,
        background_color: Color,
    ) -> Vec<u8>;
}

/// A window that can be used in a [`Compositor`].
///
/// This is just a convenient super trait of the `raw-window-handle`
/// traits.
pub trait Window: HasWindowHandle + Debug + MaybeSend + MaybeSync + 'static {}

impl<T> Window for T where T: HasWindowHandle + Debug + MaybeSend + MaybeSync + 'static {}

/// An owned display handle that can be used in a [`Compositor`].
///
/// This is just a convenient super trait of the `raw-window-handle`
/// trait.
pub trait Display: HasDisplayHandle + Debug + Send + Sync + 'static {}

impl<T> Display for T where T: HasDisplayHandle + Debug + Send + Sync + 'static {}

/// Defines the default compositor of a renderer.
pub trait Default {
    /// The compositor of the renderer.
    type Compositor: Compositor<Renderer = Self>;
}

/// Result of an unsuccessful call to [`Compositor::present`].
#[derive(Clone, PartialEq, Eq, Debug, Error)]
pub enum SurfaceError {
    /// A timeout was encountered while trying to acquire the next frame.
    #[error("A timeout was encountered while trying to acquire the next frame")]
    Timeout,
    /// The underlying surface has changed, and therefore the surface must be updated.
    #[error("The underlying surface has changed, and therefore the surface must be updated.")]
    Outdated,
    /// The swap chain has been lost and needs to be recreated.
    #[error("The surface has been lost and needs to be recreated")]
    Lost,
    /// There is no more memory left to allocate a new frame.
    #[error("There is no more memory left to allocate a new frame")]
    OutOfMemory,
    /// The surface is occluded and must not be drawn to.
    #[error("The surface is occluded and must not be drawn to")]
    Occluded,
    /// Acquiring a texture failed with a generic error.
    #[error("Acquiring a texture failed with a generic error")]
    Other,
}

/// Contains information about the graphics (e.g. graphics adapter, graphics backend).
#[derive(Debug)]
pub struct Information {
    /// Contains the graphics adapter.
    pub adapter: String,
    /// Contains the graphics backend.
    pub backend: String,
}

#[cfg(debug_assertions)]
impl Compositor for () {
    type Renderer = ();
    type Surface = ();

    async fn new(
        _settings: backend::Settings,
        _display: impl Display,
        _compatible_window: impl Window + Clone,
        _shell: Shell,
    ) -> Result<Self, backend::Error> {
        Ok(())
    }

    fn create_renderer(&self, _settings: renderer::Settings) -> Self::Renderer {}

    fn create_surface(
        &mut self,
        _window: impl Window + Clone,
        _width: u32,
        _height: u32,
    ) -> Self::Surface {
    }

    fn configure_surface(&mut self, _surface: &mut Self::Surface, _width: u32, _height: u32) {}

    fn load_font(&mut self, _font: Cow<'static, [u8]>) -> Result<(), font::Error> {
        Ok(())
    }

    fn list_fonts(&mut self) -> Result<Vec<font::Family>, font::Error> {
        Ok(Vec::new())
    }

    fn information(&self) -> Information {
        Information {
            adapter: String::from("Null Renderer"),
            backend: String::from("Null"),
        }
    }

    fn present(
        &mut self,
        _renderer: &mut Self::Renderer,
        _surface: &mut Self::Surface,
        _viewport: &Viewport,
        _background_color: Color,
        _on_pre_present: impl FnOnce(),
    ) -> Result<(), SurfaceError> {
        Ok(())
    }

    fn screenshot(
        &mut self,
        _renderer: &mut Self::Renderer,
        _viewport: &Viewport,
        _background_color: Color,
    ) -> Vec<u8> {
        vec![]
    }
}

#[cfg(debug_assertions)]
impl Default for () {
    type Compositor = ();
}
