//! Native Metal presentation for preview surfaces.
//!
//! A browser multipart-PNG stream or terminal image protocol necessarily
//! consumes CPU-visible pixels. This module is the distinct native path: it
//! keeps an AppKit window and `CAMetalLayer` inside frankentorch's sanctioned
//! unsafe boundary, reads a [`SharedBuffer`] directly on the GPU, and presents
//! the drawable without copying frame pixels into host-owned memory.
//!
//! The presenter is deliberately main-thread-only. It is neither `Send` nor
//! `Sync`, and every lifecycle method verifies the calling thread before
//! touching AppKit. A missing drawable is an occlusion outcome, not a fatal
//! Metal failure. Resize and close are idempotent.

pub use crate::CommandBufferState;
use crate::compute::{Gateway, SharedBuffer};
use std::fmt;

/// Native preview-window configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationConfig {
    /// Initial drawable width in pixels.
    pub width: u32,
    /// Initial drawable height in pixels.
    pub height: u32,
    /// Window title.
    pub title: String,
}

impl PresentationConfig {
    /// Construct a configuration for a native preview window.
    pub fn new(width: u32, height: u32, title: impl Into<String>) -> Self {
        Self {
            width,
            height,
            title: title.into(),
        }
    }

    /// Reject a zero-sized drawable before touching platform APIs.
    pub fn validate(&self) -> Result<(), PresentationError> {
        validate_dimensions(self.width, self.height)
    }
}

impl Default for PresentationConfig {
    fn default() -> Self {
        Self::new(1280, 720, "FrankenTorch Metal Preview")
    }
}

/// Observable state of the native presentation surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationState {
    /// The window is visible and eligible to acquire a drawable.
    Visible,
    /// The window is minimized, covered, or temporarily has no drawable.
    Occluded,
    /// The user closed the window, or [`NativePresenter::close`] ran.
    Closed,
}

/// Outcome of one presentation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentOutcome {
    /// The command buffer completed and the drawable was presented.
    Presented,
    /// No drawable was acquired because the surface is currently occluded.
    Occluded,
}

/// Occupancy facts for the fixed RGBA8-to-drawable presentation kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationPipelineInfo {
    /// Chosen two-dimensional threadgroup shape.
    pub threads_per_threadgroup: [usize; 2],
    /// Maximum total threads supported by the pipeline.
    pub max_threads_per_threadgroup: usize,
    /// Pipeline SIMD width reported by Metal.
    pub thread_execution_width: usize,
}

/// Failure to create, update, or present a native preview surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentationError {
    /// Native Metal presentation is unavailable on this target or machine.
    Unavailable,
    /// AppKit lifecycle work was attempted away from the process main thread.
    WrongThread,
    /// A zero-sized drawable was requested.
    InvalidDimensions { width: u32, height: u32 },
    /// The source row stride is smaller than one packed RGBA8 row.
    InvalidStride { minimum: usize, actual: usize },
    /// Dimension or byte-layout arithmetic overflowed.
    SizeOverflow,
    /// The shared buffer cannot cover the requested source layout.
    BufferTooSmall { required: usize, actual: usize },
    /// AppKit could not create or retain a required native object.
    Window(&'static str),
    /// The fixed presentation shader or pipeline could not be built.
    Pipeline(String),
    /// Presentation was requested after the window closed.
    Closed,
    /// The command buffer did not finish successfully.
    CommandBuffer(CommandBufferState),
}

impl fmt::Display for PresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "native Metal presentation unavailable"),
            Self::WrongThread => {
                write!(f, "native Metal presentation must run on the main thread")
            }
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid presentation dimensions {width}x{height}")
            }
            Self::InvalidStride { minimum, actual } => {
                write!(
                    f,
                    "RGBA8 source stride {actual} is smaller than the required {minimum}"
                )
            }
            Self::SizeOverflow => write!(f, "presentation surface byte size overflow"),
            Self::BufferTooSmall { required, actual } => write!(
                f,
                "RGBA8 source needs {required} bytes but the shared buffer has {actual}"
            ),
            Self::Window(message) => write!(f, "native preview window error: {message}"),
            Self::Pipeline(message) => {
                write!(f, "native presentation pipeline error: {message}")
            }
            Self::Closed => write!(f, "native preview window is closed"),
            Self::CommandBuffer(state) => {
                write!(f, "Metal presentation command buffer ended in {state:?}")
            }
        }
    }
}

impl std::error::Error for PresentationError {}

fn validate_dimensions(width: u32, height: u32) -> Result<(), PresentationError> {
    if width == 0 || height == 0 {
        return Err(PresentationError::InvalidDimensions { width, height });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn required_source_bytes(
    buffer: &SharedBuffer,
    width: u32,
    height: u32,
    stride: usize,
) -> Result<usize, PresentationError> {
    required_bytes_for_layout(buffer.len_bytes(), width, height, stride)
}

#[cfg(any(test, target_os = "macos"))]
fn required_bytes_for_layout(
    available: usize,
    width: u32,
    height: u32,
    stride: usize,
) -> Result<usize, PresentationError> {
    validate_dimensions(width, height)?;
    let row_bytes = usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or(PresentationError::SizeOverflow)?;
    if stride < row_bytes {
        return Err(PresentationError::InvalidStride {
            minimum: row_bytes,
            actual: stride,
        });
    }
    let height = usize::try_from(height).map_err(|_| PresentationError::SizeOverflow)?;
    let required = height
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(stride))
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .ok_or(PresentationError::SizeOverflow)?;
    u32::try_from(stride).map_err(|_| PresentationError::SizeOverflow)?;
    u32::try_from(required).map_err(|_| PresentationError::SizeOverflow)?;
    if required > available {
        return Err(PresentationError::BufferTooSmall {
            required,
            actual: available,
        });
    }
    Ok(required)
}

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)]
mod imp {
    use super::{
        CommandBufferState, Gateway, PresentOutcome, PresentationConfig, PresentationError,
        PresentationPipelineInfo, PresentationState, SharedBuffer, required_source_bytes,
        validate_dimensions,
    };
    use core_graphics_types::geometry::CGSize;
    use foreign_types::ForeignType;
    use metal::{
        CompileOptions, ComputePipelineDescriptor, ComputePipelineState, MTLPixelFormat, MTLSize,
        MetalLayer,
    };
    use objc::{
        Encode, Encoding, class, msg_send,
        rc::autoreleasepool,
        runtime::{BOOL, NO, Object, YES},
        sel, sel_impl,
    };
    use std::{ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};

    const NS_WINDOW_STYLE_TITLED: u64 = 1 << 0;
    const NS_WINDOW_STYLE_CLOSABLE: u64 = 1 << 1;
    const NS_WINDOW_STYLE_MINIATURIZABLE: u64 = 1 << 2;
    const NS_BACKING_STORE_BUFFERED: u64 = 2;
    const NS_APPLICATION_ACTIVATION_POLICY_ACCESSORY: i64 = 1;
    const NS_WINDOW_OCCLUSION_STATE_VISIBLE: u64 = 1 << 1;
    const UTF8_ENCODING: usize = 4;

    const PRESENT_RGBA8_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct PresentParams {
    uint width;
    uint height;
    uint stride;
};

kernel void present_rgba8(
    device const uchar* source [[buffer(0)]],
    constant PresentParams& params [[buffer(1)]],
    texture2d<half, access::write> target [[texture(0)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= params.width || position.y >= params.height) {
        return;
    }
    uint offset = position.y * params.stride + position.x * 4;
    half4 rgba = half4(
        half(source[offset]),
        half(source[offset + 1]),
        half(source[offset + 2]),
        half(source[offset + 3])
    ) / half(255.0);
    target.write(rgba, position);
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NativePoint {
        x: f64,
        y: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NativeSize {
        width: f64,
        height: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NativeRect {
        origin: NativePoint,
        size: NativeSize,
    }

    unsafe impl Encode for NativePoint {
        fn encode() -> Encoding {
            unsafe { Encoding::from_str("{CGPoint=dd}") }
        }
    }

    unsafe impl Encode for NativeSize {
        fn encode() -> Encoding {
            unsafe { Encoding::from_str("{CGSize=dd}") }
        }
    }

    unsafe impl Encode for NativeRect {
        fn encode() -> Encoding {
            unsafe { Encoding::from_str("{CGRect={CGPoint=dd}{CGSize=dd}}") }
        }
    }

    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "QuartzCore", kind = "framework")]
    unsafe extern "C" {}

    /// A main-thread-only AppKit window and `CAMetalLayer`.
    ///
    /// The `Rc` marker makes the type neither `Send` nor `Sync`, so safe Rust
    /// cannot move its AppKit lifetimes to another thread after construction.
    pub struct NativePresenter {
        app: NonNull<Object>,
        window: Option<NonNull<Object>>,
        layer: MetalLayer,
        gateway: Gateway,
        pipeline: ComputePipelineState,
        pipeline_info: PresentationPipelineInfo,
        width: u32,
        height: u32,
        closed: bool,
        _main_thread_only: PhantomData<Rc<()>>,
    }

    impl NativePresenter {
        /// Open and show a native preview window.
        pub fn open(
            gateway: &Gateway,
            config: PresentationConfig,
        ) -> Result<Self, PresentationError> {
            require_main_thread()?;
            config.validate()?;
            let (pipeline, pipeline_info) = build_pipeline(gateway)?;
            autoreleasepool(|| unsafe {
                let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
                let app =
                    NonNull::new(app).ok_or(PresentationError::Window("NSApplication was null"))?;
                let _: BOOL = msg_send![
                    app.as_ptr(),
                    setActivationPolicy:NS_APPLICATION_ACTIVATION_POLICY_ACCESSORY
                ];
                let _: () = msg_send![app.as_ptr(), finishLaunching];

                let rect = NativeRect {
                    origin: NativePoint { x: 0.0, y: 0.0 },
                    size: NativeSize {
                        width: f64::from(config.width),
                        height: f64::from(config.height),
                    },
                };
                let style = NS_WINDOW_STYLE_TITLED
                    | NS_WINDOW_STYLE_CLOSABLE
                    | NS_WINDOW_STYLE_MINIATURIZABLE;
                let allocated: *mut Object = msg_send![class!(NSWindow), alloc];
                if allocated.is_null() {
                    return Err(PresentationError::Window("NSWindow allocation failed"));
                }
                let window: *mut Object = msg_send![
                    allocated,
                    initWithContentRect:rect
                    styleMask:style
                    backing:NS_BACKING_STORE_BUFFERED
                    defer:NO
                ];
                let Some(window) = NonNull::new(window) else {
                    return Err(PresentationError::Window("NSWindow initialization failed"));
                };
                let _: () = msg_send![window.as_ptr(), setReleasedWhenClosed:NO];
                let title = nsstring(&config.title);
                let _: () = msg_send![window.as_ptr(), setTitle:title];

                let view: *mut Object = msg_send![window.as_ptr(), contentView];
                let Some(view) = NonNull::new(view) else {
                    let _: () = msg_send![window.as_ptr(), close];
                    let _: () = msg_send![window.as_ptr(), release];
                    return Err(PresentationError::Window("NSWindow contentView was null"));
                };

                let layer = MetalLayer::new();
                layer.set_device(gateway.metal_device());
                layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                layer.set_framebuffer_only(false);
                layer.set_presents_with_transaction(false);
                layer.set_display_sync_enabled(true);
                layer.set_maximum_drawable_count(3);
                layer.set_contents_scale(1.0);
                layer.set_opaque(true);
                layer.set_drawable_size(CGSize::new(
                    f64::from(config.width),
                    f64::from(config.height),
                ));

                let _: () = msg_send![view.as_ptr(), setWantsLayer:YES];
                let layer_object = layer.as_ptr().cast::<Object>();
                let _: () = msg_send![view.as_ptr(), setLayer:layer_object];
                let nil: *mut Object = std::ptr::null_mut();
                let _: () = msg_send![window.as_ptr(), makeKeyAndOrderFront:nil];
                let _: () = msg_send![app.as_ptr(), activateIgnoringOtherApps:YES];

                Ok(Self {
                    app,
                    window: Some(window),
                    layer,
                    gateway: *gateway,
                    pipeline,
                    pipeline_info,
                    width: config.width,
                    height: config.height,
                    closed: false,
                    _main_thread_only: PhantomData,
                })
            })
        }

        /// Fixed pipeline occupancy facts used by this presenter.
        pub fn pipeline_info(&self) -> PresentationPipelineInfo {
            self.pipeline_info
        }

        /// Whether the presenter has completed teardown.
        pub fn is_closed(&self) -> bool {
            self.closed
        }

        /// Drain pending AppKit events and report the resulting window state.
        pub fn poll_events(&mut self) -> Result<PresentationState, PresentationError> {
            require_main_thread()?;
            autoreleasepool(|| unsafe {
                self.pump_events();
                Ok(self.presentation_state())
            })
        }

        /// Resize the window content and drawable. Repeating the same size is a
        /// no-op.
        pub fn resize(&mut self, width: u32, height: u32) -> Result<(), PresentationError> {
            require_main_thread()?;
            validate_dimensions(width, height)?;
            if self.closed {
                return Err(PresentationError::Closed);
            }
            if self.width == width && self.height == height {
                return Ok(());
            }
            autoreleasepool(|| unsafe {
                let window = self.window.ok_or(PresentationError::Closed)?;
                let size = NativeSize {
                    width: f64::from(width),
                    height: f64::from(height),
                };
                let _: () = msg_send![window.as_ptr(), setContentSize:size];
                self.layer
                    .set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
                self.width = width;
                self.height = height;
                Ok(())
            })
        }

        /// Present packed RGBA8 pixels directly from a Metal shared buffer.
        ///
        /// `stride` is measured in bytes. Padding after each row is permitted.
        /// The source remains GPU-visible throughout; the only per-frame CPU
        /// values are the three integer layout parameters.
        pub fn present_rgba8(
            &mut self,
            source: &SharedBuffer,
            width: u32,
            height: u32,
            stride: usize,
        ) -> Result<PresentOutcome, PresentationError> {
            require_main_thread()?;
            required_source_bytes(source, width, height, stride)?;
            if self.closed {
                return Err(PresentationError::Closed);
            }
            self.resize(width, height)?;

            autoreleasepool(|| unsafe {
                self.pump_events();
                match self.presentation_state() {
                    PresentationState::Closed => return Err(PresentationError::Closed),
                    PresentationState::Occluded => return Ok(PresentOutcome::Occluded),
                    PresentationState::Visible => {}
                }
                let Some(drawable) = self.layer.next_drawable() else {
                    return Ok(PresentOutcome::Occluded);
                };

                let params = [
                    width,
                    height,
                    u32::try_from(stride).map_err(|_| PresentationError::SizeOverflow)?,
                ];
                let command = self.gateway.metal_queue().new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.pipeline);
                encoder.set_buffer(0, Some(source.metal_buffer()), 0);
                encoder.set_bytes(
                    1,
                    std::mem::size_of_val(&params) as u64,
                    params.as_ptr().cast::<c_void>(),
                );
                encoder.set_texture(0, Some(drawable.texture()));
                encoder.dispatch_threads(
                    MTLSize::new(u64::from(width), u64::from(height), 1),
                    MTLSize::new(
                        self.pipeline_info.threads_per_threadgroup[0] as u64,
                        self.pipeline_info.threads_per_threadgroup[1] as u64,
                        1,
                    ),
                );
                encoder.end_encoding();
                command.present_drawable(drawable);
                command.commit();
                command.wait_until_completed();

                let state = CommandBufferState::from(command.status());
                if state != CommandBufferState::Completed {
                    return Err(PresentationError::CommandBuffer(state));
                }
                Ok(PresentOutcome::Presented)
            })
        }

        /// Close and detach the window. Repeating the call is a no-op.
        pub fn close(&mut self) -> Result<(), PresentationError> {
            require_main_thread()?;
            autoreleasepool(|| unsafe {
                self.close_inner();
            });
            Ok(())
        }

        unsafe fn pump_events(&mut self) {
            if self.closed {
                return;
            }
            let distant_past: *mut Object = msg_send![class!(NSDate), distantPast];
            let mode = unsafe { nsstring("kCFRunLoopDefaultMode") };
            loop {
                let event: *mut Object = msg_send![
                    self.app.as_ptr(),
                    nextEventMatchingMask:u64::MAX
                    untilDate:distant_past
                    inMode:mode
                    dequeue:YES
                ];
                if event.is_null() {
                    break;
                }
                let _: () = msg_send![self.app.as_ptr(), sendEvent:event];
            }
            let _: () = msg_send![self.app.as_ptr(), updateWindows];
        }

        unsafe fn presentation_state(&mut self) -> PresentationState {
            if self.closed {
                return PresentationState::Closed;
            }
            let Some(window) = self.window else {
                self.closed = true;
                return PresentationState::Closed;
            };
            let visible: BOOL = msg_send![window.as_ptr(), isVisible];
            if visible == NO {
                self.closed = true;
                return PresentationState::Closed;
            }
            let miniaturized: BOOL = msg_send![window.as_ptr(), isMiniaturized];
            if miniaturized == YES {
                return PresentationState::Occluded;
            }
            let occlusion: u64 = msg_send![window.as_ptr(), occlusionState];
            if occlusion & NS_WINDOW_OCCLUSION_STATE_VISIBLE == 0 {
                PresentationState::Occluded
            } else {
                PresentationState::Visible
            }
        }

        unsafe fn close_inner(&mut self) {
            if self.closed && self.window.is_none() {
                return;
            }
            if let Some(window) = self.window.take() {
                let view: *mut Object = msg_send![window.as_ptr(), contentView];
                if let Some(view) = NonNull::new(view) {
                    let nil: *mut Object = std::ptr::null_mut();
                    let _: () = msg_send![view.as_ptr(), setLayer:nil];
                }
                let _: () = msg_send![window.as_ptr(), close];
                let _: () = msg_send![window.as_ptr(), release];
            }
            self.closed = true;
        }
    }

    impl Drop for NativePresenter {
        fn drop(&mut self) {
            debug_assert!(
                is_main_thread(),
                "NativePresenter cannot leave its main-thread lifetime"
            );
            autoreleasepool(|| unsafe {
                self.close_inner();
            });
        }
    }

    fn build_pipeline(
        gateway: &Gateway,
    ) -> Result<(ComputePipelineState, PresentationPipelineInfo), PresentationError> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = gateway
            .metal_device()
            .new_library_with_source(PRESENT_RGBA8_SOURCE, &options)
            .map_err(PresentationError::Pipeline)?;
        let function = library
            .get_function("present_rgba8", None)
            .map_err(PresentationError::Pipeline)?;
        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        let function = descriptor
            .compute_function()
            .ok_or_else(|| PresentationError::Pipeline("compute function was lost".into()))?;
        let pipeline = gateway
            .metal_device()
            .new_compute_pipeline_state_with_function(function)
            .map_err(PresentationError::Pipeline)?;
        let width = pipeline.thread_execution_width() as usize;
        let maximum = pipeline.max_total_threads_per_threadgroup() as usize;
        if width == 0 || maximum < width {
            return Err(PresentationError::Pipeline(format!(
                "invalid occupancy: execution width {width}, maximum {maximum}"
            )));
        }
        let height = (maximum / width).clamp(1, 16);
        let info = PresentationPipelineInfo {
            threads_per_threadgroup: [width, height],
            max_threads_per_threadgroup: maximum,
            thread_execution_width: width,
        };
        Ok((pipeline, info))
    }

    fn require_main_thread() -> Result<(), PresentationError> {
        if is_main_thread() {
            Ok(())
        } else {
            Err(PresentationError::WrongThread)
        }
    }

    fn is_main_thread() -> bool {
        unsafe {
            let main: BOOL = msg_send![class!(NSThread), isMainThread];
            main == YES
        }
    }

    unsafe fn nsstring(value: &str) -> *mut Object {
        let allocated: *mut Object = msg_send![class!(NSString), alloc];
        let string: *mut Object = msg_send![
            allocated,
            initWithBytes:value.as_ptr().cast::<c_void>()
            length:value.len()
            encoding:UTF8_ENCODING
        ];
        let string: *mut Object = msg_send![string, autorelease];
        string
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::{
        Gateway, PresentOutcome, PresentationConfig, PresentationError, PresentationPipelineInfo,
        PresentationState, SharedBuffer,
    };

    /// Unconstructible native-presenter stub for non-macOS targets.
    pub struct NativePresenter {
        _never: (),
    }

    impl NativePresenter {
        /// Native Metal presentation is unavailable off macOS.
        pub fn open(
            _gateway: &Gateway,
            _config: PresentationConfig,
        ) -> Result<Self, PresentationError> {
            Err(PresentationError::Unavailable)
        }

        /// Unreachable off macOS.
        pub fn pipeline_info(&self) -> PresentationPipelineInfo {
            PresentationPipelineInfo {
                threads_per_threadgroup: [0, 0],
                max_threads_per_threadgroup: 0,
                thread_execution_width: 0,
            }
        }

        /// Unreachable off macOS.
        pub fn is_closed(&self) -> bool {
            true
        }

        /// Unreachable off macOS.
        pub fn poll_events(&mut self) -> Result<PresentationState, PresentationError> {
            Err(PresentationError::Unavailable)
        }

        /// Unreachable off macOS.
        pub fn resize(&mut self, _width: u32, _height: u32) -> Result<(), PresentationError> {
            Err(PresentationError::Unavailable)
        }

        /// Unreachable off macOS.
        pub fn present_rgba8(
            &mut self,
            _source: &SharedBuffer,
            _width: u32,
            _height: u32,
            _stride: usize,
        ) -> Result<PresentOutcome, PresentationError> {
            Err(PresentationError::Unavailable)
        }

        /// Unreachable off macOS.
        pub fn close(&mut self) -> Result<(), PresentationError> {
            Err(PresentationError::Unavailable)
        }
    }
}

pub use imp::NativePresenter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_dimensions_are_rejected() {
        assert_eq!(
            PresentationConfig::new(0, 720, "preview").validate(),
            Err(PresentationError::InvalidDimensions {
                width: 0,
                height: 720,
            })
        );
        assert_eq!(
            PresentationConfig::new(1280, 0, "preview").validate(),
            Err(PresentationError::InvalidDimensions {
                width: 1280,
                height: 0,
            })
        );
    }

    #[test]
    fn padded_source_layout_is_bounds_checked_through_the_last_pixel() {
        assert_eq!(required_bytes_for_layout(28, 2, 3, 10), Ok(28));
        assert_eq!(
            required_bytes_for_layout(27, 2, 3, 10),
            Err(PresentationError::BufferTooSmall {
                required: 28,
                actual: 27,
            })
        );
        assert_eq!(
            required_bytes_for_layout(64, 3, 2, 11),
            Err(PresentationError::InvalidStride {
                minimum: 12,
                actual: 11,
            })
        );
    }

    #[test]
    fn source_layout_overflow_is_typed() {
        assert_eq!(
            required_bytes_for_layout(usize::MAX, 1, 2, usize::MAX),
            Err(PresentationError::SizeOverflow)
        );
        assert_eq!(
            required_bytes_for_layout(usize::MAX, 1, 2, u32::MAX as usize),
            Err(PresentationError::SizeOverflow)
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn native_presenter_is_an_explicit_stub_off_macos() {
        assert!(matches!(Gateway::open(), Err(crate::Error::Unavailable)));
    }
}
