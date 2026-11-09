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
//!
//! # Input
//!
//! Every event pump also translates the window's AppKit events into typed
//! [`InputEvent`]s: pointer motion, buttons and scroll, keys and modifiers,
//! focus changes, pointer-capture loss and surface resizes. They wait in a
//! bounded queue until [`NativePresenter::poll_input`] drains them in
//! delivery order. Positions are [`LogicalPoint`]s; [`SurfaceMetrics`] relates
//! them to drawable pixels. With the `synthetic-input` feature,
//! `NativePresenter::post_synthetic` posts real AppKit events into the
//! application queue, so a dependent's test exercises the same dispatch and
//! translation path as a user.

pub use crate::CommandBufferState;
use crate::compute::{Gateway, SharedBuffer};
use std::collections::VecDeque;
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

/// Input events held between polls. Continuous events coalesce first, so a
/// host that polls once per frame never reaches this bound.
pub const INPUT_QUEUE_CAPACITY: usize = 1024;

/// Modifier keys held when an input event occurred.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Modifiers {
    /// Either Shift key.
    pub shift: bool,
    /// Either Control key.
    pub control: bool,
    /// Either Option (Alt) key.
    pub option: bool,
    /// Either Command key.
    pub command: bool,
    /// Caps Lock is engaged.
    pub caps_lock: bool,
}

const APPKIT_CAPS_LOCK: u64 = 1 << 16;
const APPKIT_SHIFT: u64 = 1 << 17;
const APPKIT_CONTROL: u64 = 1 << 18;
const APPKIT_OPTION: u64 = 1 << 19;
const APPKIT_COMMAND: u64 = 1 << 20;

impl Modifiers {
    /// No modifier held.
    pub const NONE: Self = Self {
        shift: false,
        control: false,
        option: false,
        command: false,
        caps_lock: false,
    };

    /// Decode the device-independent bits of AppKit's
    /// `NSEventModifierFlags`. Device-dependent bits are ignored.
    #[must_use]
    pub const fn from_appkit_flags(flags: u64) -> Self {
        Self {
            shift: flags & APPKIT_SHIFT != 0,
            control: flags & APPKIT_CONTROL != 0,
            option: flags & APPKIT_OPTION != 0,
            command: flags & APPKIT_COMMAND != 0,
            caps_lock: flags & APPKIT_CAPS_LOCK != 0,
        }
    }

    /// The `NSEventModifierFlags` bits these modifiers set.
    #[must_use]
    pub const fn appkit_flags(self) -> u64 {
        let mut flags = 0;
        if self.shift {
            flags |= APPKIT_SHIFT;
        }
        if self.control {
            flags |= APPKIT_CONTROL;
        }
        if self.option {
            flags |= APPKIT_OPTION;
        }
        if self.command {
            flags |= APPKIT_COMMAND;
        }
        if self.caps_lock {
            flags |= APPKIT_CAPS_LOCK;
        }
        flags
    }
}

/// A pointer button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerButton {
    /// The primary (usually left) button.
    Primary,
    /// The secondary (usually right) button.
    Secondary,
    /// The middle button or wheel press.
    Middle,
    /// Another button, by AppKit button number (3 or more).
    Other(u8),
}

impl PointerButton {
    /// Decode AppKit's `buttonNumber`. Numbers past 31 saturate to
    /// `Other(31)`, the last one a [`ButtonSet`] can hold.
    #[must_use]
    pub fn from_appkit_number(number: i64) -> Self {
        match number {
            0 => Self::Primary,
            1 => Self::Secondary,
            2 => Self::Middle,
            n => Self::Other(u8::try_from(n.clamp(3, 31)).unwrap_or(31)),
        }
    }

    const fn bit(self) -> u32 {
        match self {
            Self::Primary => 1,
            Self::Secondary => 1 << 1,
            Self::Middle => 1 << 2,
            Self::Other(n) => {
                if n >= 3 && n <= 31 {
                    1 << n
                } else {
                    1 << 31
                }
            }
        }
    }
}

/// The pointer buttons held at an event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ButtonSet(u32);

impl ButtonSet {
    /// No button held.
    pub const EMPTY: Self = Self(0);

    /// Whether no button is held.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether `button` is held.
    #[must_use]
    pub const fn contains(self, button: PointerButton) -> bool {
        self.0 & button.bit() != 0
    }

    /// This set with `button` held.
    #[must_use]
    pub const fn with(self, button: PointerButton) -> Self {
        Self(self.0 | button.bit())
    }

    /// This set with `button` released.
    #[must_use]
    pub const fn without(self, button: PointerButton) -> Self {
        Self(self.0 & !button.bit())
    }
}

/// A point in logical window coordinates: AppKit points from the top-left
/// corner of the content area, x rightward and y downward. It is not
/// clamped, so a drag can report points outside the content area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogicalPoint {
    /// Points from the content area's left edge.
    pub x: f64,
    /// Points from the content area's top edge.
    pub y: f64,
}

/// The unit of a scroll delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScrollUnit {
    /// Precise deltas in logical points (trackpads, Magic Mouse).
    Points,
    /// Line deltas from a notched wheel.
    Lines,
}

/// Geometry of the presentation surface.
///
/// A [`LogicalPoint`] maps to the drawable by the ratio of drawable pixels
/// to logical points on each axis. The presenter keeps its layer's
/// `contentsScale` at 1, so that ratio is 1 until the window is resized and
/// the next frame is presented at the new size; `backing_scale` is the
/// screen's own pixel density, for a host that wants to render denser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceMetrics {
    /// Content-area width in logical points.
    pub logical_width: f64,
    /// Content-area height in logical points.
    pub logical_height: f64,
    /// Device pixels per logical point on the window's screen.
    pub backing_scale: f64,
    /// Drawable width in pixels: the width `present_rgba8` writes.
    pub drawable_width: u32,
    /// Drawable height in pixels.
    pub drawable_height: u32,
}

impl SurfaceMetrics {
    /// The continuous drawable-pixel position of a logical point (pixel
    /// `(i, j)` covers `[i, i + 1) × [j, j + 1)`, top-left origin), or
    /// `None` while the content area is empty.
    #[must_use]
    pub fn to_drawable(&self, point: LogicalPoint) -> Option<[f64; 2]> {
        if !(self.logical_width > 0.0 && self.logical_height > 0.0) {
            return None;
        }
        Some([
            point.x * f64::from(self.drawable_width) / self.logical_width,
            point.y * f64::from(self.drawable_height) / self.logical_height,
        ])
    }
}

/// One typed input event from the native window, in delivery order.
#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    /// The pointer moved; `buttons` are held during a drag.
    PointerMoved {
        /// New position.
        position: LogicalPoint,
        /// Buttons held.
        buttons: ButtonSet,
        /// Modifiers held.
        modifiers: Modifiers,
    },
    /// A pointer button was pressed or released.
    Button {
        /// The button.
        button: PointerButton,
        /// Pressed (true) or released (false).
        pressed: bool,
        /// Pointer position.
        position: LogicalPoint,
        /// AppKit's click count (2 for a double click).
        clicks: u32,
        /// Modifiers held.
        modifiers: Modifiers,
    },
    /// A scroll: AppKit's `scrollingDeltaX`/`scrollingDeltaY`, unchanged
    /// (positive y scrolls toward the top of the content, after the user's
    /// natural-scrolling setting).
    Scroll {
        /// Horizontal and vertical delta.
        delta: [f64; 2],
        /// Points or lines.
        unit: ScrollUnit,
        /// Pointer position.
        position: LogicalPoint,
        /// Modifiers held.
        modifiers: Modifiers,
    },
    /// A key went down or up.
    Key {
        /// AppKit's hardware-independent virtual key code (`kVK_*`).
        code: u16,
        /// `charactersIgnoringModifiers`, possibly empty. Function keys use
        /// AppKit's private-use characters (U+F700 and up).
        text: String,
        /// Down (true) or up (false).
        pressed: bool,
        /// An auto-repeat of a held key.
        repeat: bool,
        /// Modifiers held.
        modifiers: Modifiers,
    },
    /// The held modifier keys changed.
    ModifiersChanged {
        /// The modifiers now held.
        modifiers: Modifiers,
    },
    /// The window became the key window and receives keys.
    FocusGained,
    /// The window stopped being the key window.
    FocusLost,
    /// Focus left while buttons were held. They are reported released; no
    /// matching release event follows.
    PointerCaptureLost {
        /// The buttons that were held.
        released: ButtonSet,
    },
    /// The surface geometry changed. The first poll reports the initial
    /// geometry.
    Resized(SurfaceMetrics),
}

impl InputEvent {
    /// Motion, scroll and resize carry continuous state: a later one of the
    /// same kind supersedes or adds to an earlier one, so they may be merged
    /// or, as a last resort, dropped.
    #[must_use]
    pub const fn is_continuous(&self) -> bool {
        matches!(
            self,
            Self::PointerMoved { .. } | Self::Scroll { .. } | Self::Resized(_)
        )
    }
}

/// The presenter's bounded event queue.
///
/// A push first merges into the newest queued event: motion with the same
/// buttons and modifiers keeps the latest position, scroll with the same
/// unit and modifiers adds its delta, and a resize replaces a resize.
/// Delivery order is otherwise preserved. When the queue is full, the
/// oldest continuous event is dropped to make room; with none left, the new
/// event is dropped. Every drop is counted.
// Off macOS the presenter is an unconstructible stub, so only the unit
// tests use the queue there.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[derive(Debug)]
pub(crate) struct InputQueue {
    events: VecDeque<InputEvent>,
    capacity: usize,
    dropped: u64,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl InputQueue {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            dropped: 0,
        }
    }

    pub(crate) fn push(&mut self, event: InputEvent) {
        if let Some(last) = self.events.back_mut() {
            match (last, &event) {
                (
                    InputEvent::PointerMoved {
                        position,
                        buttons,
                        modifiers,
                    },
                    InputEvent::PointerMoved {
                        position: next,
                        buttons: next_buttons,
                        modifiers: next_modifiers,
                    },
                ) if buttons == next_buttons && modifiers == next_modifiers => {
                    *position = *next;
                    return;
                }
                (
                    InputEvent::Scroll {
                        delta,
                        unit,
                        position,
                        modifiers,
                    },
                    InputEvent::Scroll {
                        delta: next_delta,
                        unit: next_unit,
                        position: next,
                        modifiers: next_modifiers,
                    },
                ) if unit == next_unit && modifiers == next_modifiers => {
                    delta[0] += next_delta[0];
                    delta[1] += next_delta[1];
                    *position = *next;
                    return;
                }
                (InputEvent::Resized(metrics), InputEvent::Resized(next)) => {
                    *metrics = *next;
                    return;
                }
                _ => {}
            }
        }
        if self.events.len() >= self.capacity {
            self.dropped = self.dropped.saturating_add(1);
            match self.events.iter().position(InputEvent::is_continuous) {
                Some(index) => {
                    self.events.remove(index);
                }
                None => return,
            }
        }
        self.events.push_back(event);
    }

    pub(crate) fn drain_into(&mut self, out: &mut Vec<InputEvent>) {
        out.extend(self.events.drain(..));
    }

    pub(crate) fn dropped(&self) -> u64 {
        self.dropped
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.events.len()
    }
}

#[cfg(feature = "synthetic-input")]
/// A real AppKit event to post into the application queue (test seam).
///
/// Each one is built with `NSEvent`'s or Core Graphics' constructors for the
/// presenter's own window and posted with `-[NSApplication postEvent:]`, so
/// it reaches the presenter through the same `nextEventMatchingMask` pump,
/// `sendEvent:` dispatch and translation as user input. Focus changes
/// activate or deactivate the application itself.
#[derive(Debug, Clone, PartialEq)]
pub enum SyntheticInput {
    /// Move the pointer, dragging if `dragging` names a held button
    /// (primary or secondary).
    PointerMoved {
        /// Position.
        position: LogicalPoint,
        /// The button being dragged, if any.
        dragging: Option<PointerButton>,
        /// Modifiers.
        modifiers: Modifiers,
    },
    /// Press or release the primary or secondary button.
    Button {
        /// Primary or secondary.
        button: PointerButton,
        /// Press (true) or release (false).
        pressed: bool,
        /// Position.
        position: LogicalPoint,
        /// Click count.
        clicks: u32,
        /// Modifiers.
        modifiers: Modifiers,
    },
    /// A pixel-unit scroll-wheel event at `position`.
    Scroll {
        /// Whole-point deltas, x then y.
        delta: [i32; 2],
        /// Position.
        position: LogicalPoint,
    },
    /// A key down or up.
    Key {
        /// Virtual key code.
        code: u16,
        /// Characters ignoring modifiers.
        text: String,
        /// Down or up.
        pressed: bool,
        /// Auto-repeat.
        repeat: bool,
        /// Modifiers.
        modifiers: Modifiers,
    },
    /// A modifier-flags change.
    ModifiersChanged {
        /// Modifiers now held.
        modifiers: Modifiers,
    },
    /// Make the presenter's window key again (true), or make a second
    /// window of this application key (false).
    Focus(bool),
    /// Resize the window's content area to this many logical points, as a
    /// user's resize ends: through `-[NSWindow setContentSize:]`, leaving the
    /// drawable at its size until the next frame is presented.
    Resize {
        /// Content width in points, at least 1.
        width: f64,
        /// Content height in points, at least 1.
        height: f64,
    },
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
    #[cfg(feature = "synthetic-input")]
    use super::SyntheticInput;
    use super::{
        ButtonSet, CommandBufferState, Gateway, INPUT_QUEUE_CAPACITY, InputEvent, InputQueue,
        LogicalPoint, Modifiers, PointerButton, PresentOutcome, PresentationConfig,
        PresentationError, PresentationPipelineInfo, PresentationState, ScrollUnit, SharedBuffer,
        SurfaceMetrics, required_source_bytes, validate_dimensions,
    };
    use core_graphics_types::geometry::CGSize;
    use foreign_types::ForeignType;
    use metal::{
        CompileOptions, ComputePipelineDescriptor, ComputePipelineState, MTLPixelFormat, MTLSize,
        MetalLayer,
    };
    use objc::{
        Encode, Encoding, Message, class, msg_send,
        rc::autoreleasepool,
        runtime::{BOOL, NO, Object, Sel, YES},
        sel, sel_impl,
    };
    use std::{ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};

    const NS_WINDOW_STYLE_TITLED: u64 = 1 << 0;
    const NS_WINDOW_STYLE_CLOSABLE: u64 = 1 << 1;
    const NS_WINDOW_STYLE_MINIATURIZABLE: u64 = 1 << 2;
    const NS_WINDOW_STYLE_RESIZABLE: u64 = 1 << 3;
    const NS_EVENT_LEFT_MOUSE_DOWN: u64 = 1;
    const NS_EVENT_LEFT_MOUSE_UP: u64 = 2;
    const NS_EVENT_RIGHT_MOUSE_DOWN: u64 = 3;
    const NS_EVENT_RIGHT_MOUSE_UP: u64 = 4;
    const NS_EVENT_MOUSE_MOVED: u64 = 5;
    const NS_EVENT_LEFT_MOUSE_DRAGGED: u64 = 6;
    const NS_EVENT_RIGHT_MOUSE_DRAGGED: u64 = 7;
    const NS_EVENT_KEY_DOWN: u64 = 10;
    const NS_EVENT_KEY_UP: u64 = 11;
    const NS_EVENT_FLAGS_CHANGED: u64 = 12;
    const NS_EVENT_SCROLL_WHEEL: u64 = 22;
    const NS_EVENT_OTHER_MOUSE_DOWN: u64 = 25;
    const NS_EVENT_OTHER_MOUSE_UP: u64 = 26;
    const NS_EVENT_OTHER_MOUSE_DRAGGED: u64 = 27;
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

    #[cfg(feature = "synthetic-input")]
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventCreateScrollWheelEvent2(
            source: *const c_void,
            units: u32,
            wheel_count: u32,
            wheel1: i32,
            wheel2: i32,
            wheel3: i32,
        ) -> *mut c_void;
        fn CGEventSetLocation(event: *mut c_void, location: NativePoint);
        fn CGEventSetIntegerValueField(event: *mut c_void, field: u32, value: i64);
    }

    #[cfg(feature = "synthetic-input")]
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFRelease(value: *const c_void);
    }

    #[cfg(feature = "synthetic-input")]
    const CG_SCROLL_EVENT_UNIT_PIXEL: u32 = 0;
    #[cfg(feature = "synthetic-input")]
    const CG_MOUSE_EVENT_WINDOW_UNDER_POINTER: u32 = 91;
    #[cfg(feature = "synthetic-input")]
    const CG_MOUSE_EVENT_WINDOW_UNDER_POINTER_THAT_CAN_HANDLE: u32 = 92;

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
        input: InputQueue,
        buttons: ButtonSet,
        focused: bool,
        metrics: Option<SurfaceMetrics>,
        /// A second window that takes key status for `Focus(false)`.
        #[cfg(feature = "synthetic-input")]
        focus_decoy: Option<NonNull<Object>>,
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
                let _: () = msg_send![window.as_ptr(), setAcceptsMouseMovedEvents:YES];
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
                    input: InputQueue::new(INPUT_QUEUE_CAPACITY),
                    buttons: ButtonSet::EMPTY,
                    focused: false,
                    metrics: None,
                    #[cfg(feature = "synthetic-input")]
                    focus_decoy: None,
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
        ///
        /// Typed input from these events stays queued, within
        /// [`INPUT_QUEUE_CAPACITY`], for the next [`Self::poll_input`].
        pub fn poll_events(&mut self) -> Result<PresentationState, PresentationError> {
            require_main_thread()?;
            autoreleasepool(|| unsafe {
                self.pump_events();
                Ok(self.presentation_state())
            })
        }

        /// Drain pending AppKit events, append every typed input event
        /// queued since the last drain to `events` in delivery order, and
        /// report the resulting window state.
        pub fn poll_input(
            &mut self,
            events: &mut Vec<InputEvent>,
        ) -> Result<PresentationState, PresentationError> {
            require_main_thread()?;
            let state = autoreleasepool(|| unsafe {
                self.pump_events();
                self.presentation_state()
            });
            self.input.drain_into(events);
            Ok(state)
        }

        /// Input events dropped because the queue was full between polls.
        pub fn input_dropped(&self) -> u64 {
            self.input.dropped()
        }

        /// Current surface geometry.
        pub fn surface_metrics(&self) -> Result<SurfaceMetrics, PresentationError> {
            require_main_thread()?;
            if self.closed {
                return Err(PresentationError::Closed);
            }
            autoreleasepool(|| unsafe { self.read_metrics() }).ok_or(PresentationError::Closed)
        }

        /// Let the user resize the window (or stop them). A user resize is
        /// reported as [`InputEvent::Resized`]; the drawable follows when the
        /// next frame is presented at the new size.
        pub fn set_resizable(&mut self, resizable: bool) -> Result<(), PresentationError> {
            require_main_thread()?;
            if self.closed {
                return Err(PresentationError::Closed);
            }
            autoreleasepool(|| unsafe {
                let window = self.window.ok_or(PresentationError::Closed)?;
                let mask: u64 = msg_send![window.as_ptr(), styleMask];
                let mask = if resizable {
                    mask | NS_WINDOW_STYLE_RESIZABLE
                } else {
                    mask & !NS_WINDOW_STYLE_RESIZABLE
                };
                let _: () = msg_send![window.as_ptr(), setStyleMask:mask];
                Ok(())
            })
        }

        /// Post one real AppKit event for this window into the application
        /// queue (test seam; see [`SyntheticInput`]). It is translated by the
        /// next poll, exactly like user input.
        #[cfg(feature = "synthetic-input")]
        pub fn post_synthetic(&mut self, input: &SyntheticInput) -> Result<(), PresentationError> {
            require_main_thread()?;
            if self.closed {
                return Err(PresentationError::Closed);
            }
            autoreleasepool(|| unsafe { self.post_synthetic_inner(input) })
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
                if unsafe { self.translate(event) } {
                    let _: () = msg_send![self.app.as_ptr(), sendEvent:event];
                }
            }
            let _: () = msg_send![self.app.as_ptr(), updateWindows];
            unsafe { self.observe_window() };
        }

        /// Queue the typed input carried by one dequeued AppKit event, and
        /// report whether AppKit should still dispatch it. Key events go to
        /// AppKit only as Command key equivalents; every other key would end
        /// at `noResponderFor:` and beep.
        ///
        /// Each accessor is sent only to the event types AppKit defines it
        /// for, because the others raise an Objective-C exception.
        unsafe fn translate(&mut self, event: *mut Object) -> bool {
            let Some(window) = self.window else {
                return true;
            };
            // `type` is a Rust keyword, so `sel!` cannot name it.
            let kind: u64 = match unsafe { (*event).send_message(Sel::register("type"), ()) } {
                Ok(kind) => kind,
                Err(_) => return true,
            };
            let target: *mut Object = msg_send![event, window];
            if target != window.as_ptr() {
                // A scroll with no window was posted from Core Graphics in
                // this process; it belongs to the window under its screen
                // location, as AppKit routes scrolls.
                if !(target.is_null() && kind == NS_EVENT_SCROLL_WHEEL) {
                    return true;
                }
                let screen: NativePoint = msg_send![event, locationInWindow];
                let location: NativePoint =
                    msg_send![window.as_ptr(), convertPointFromScreen:screen];
                let position = unsafe { self.logical_from_window(window, location) };
                let Some(metrics) = (unsafe { self.read_metrics() }) else {
                    return true;
                };
                let inside = position.x >= 0.0
                    && position.y >= 0.0
                    && position.x < metrics.logical_width
                    && position.y < metrics.logical_height;
                if inside {
                    unsafe { self.translate_scroll(event, position, flags_of(event)) };
                }
                return true;
            }
            let modifiers = unsafe { flags_of(event) };
            match kind {
                NS_EVENT_MOUSE_MOVED
                | NS_EVENT_LEFT_MOUSE_DRAGGED
                | NS_EVENT_RIGHT_MOUSE_DRAGGED
                | NS_EVENT_OTHER_MOUSE_DRAGGED => {
                    let location: NativePoint = msg_send![event, locationInWindow];
                    let position = unsafe { self.logical_from_window(window, location) };
                    self.input.push(InputEvent::PointerMoved {
                        position,
                        buttons: self.buttons,
                        modifiers,
                    });
                    true
                }
                NS_EVENT_LEFT_MOUSE_DOWN
                | NS_EVENT_LEFT_MOUSE_UP
                | NS_EVENT_RIGHT_MOUSE_DOWN
                | NS_EVENT_RIGHT_MOUSE_UP
                | NS_EVENT_OTHER_MOUSE_DOWN
                | NS_EVENT_OTHER_MOUSE_UP => {
                    let clicks: i64 = msg_send![event, clickCount];
                    let location: NativePoint = msg_send![event, locationInWindow];
                    let position = unsafe { self.logical_from_window(window, location) };
                    // The event type names the primary and secondary
                    // buttons; `buttonNumber` only distinguishes the others.
                    let button = match kind {
                        NS_EVENT_LEFT_MOUSE_DOWN | NS_EVENT_LEFT_MOUSE_UP => PointerButton::Primary,
                        NS_EVENT_RIGHT_MOUSE_DOWN | NS_EVENT_RIGHT_MOUSE_UP => {
                            PointerButton::Secondary
                        }
                        _ => {
                            let number: i64 = msg_send![event, buttonNumber];
                            PointerButton::from_appkit_number(number)
                        }
                    };
                    let pressed = matches!(
                        kind,
                        NS_EVENT_LEFT_MOUSE_DOWN
                            | NS_EVENT_RIGHT_MOUSE_DOWN
                            | NS_EVENT_OTHER_MOUSE_DOWN
                    );
                    self.buttons = if pressed {
                        self.buttons.with(button)
                    } else {
                        self.buttons.without(button)
                    };
                    self.input.push(InputEvent::Button {
                        button,
                        pressed,
                        position,
                        clicks: u32::try_from(clicks.max(0)).unwrap_or(u32::MAX),
                        modifiers,
                    });
                    true
                }
                NS_EVENT_SCROLL_WHEEL => {
                    let location: NativePoint = msg_send![event, locationInWindow];
                    let position = unsafe { self.logical_from_window(window, location) };
                    unsafe { self.translate_scroll(event, position, modifiers) };
                    true
                }
                NS_EVENT_KEY_DOWN | NS_EVENT_KEY_UP => {
                    let code: u16 = msg_send![event, keyCode];
                    let repeat: BOOL = msg_send![event, isARepeat];
                    let characters: *mut Object = msg_send![event, charactersIgnoringModifiers];
                    let text = unsafe { string_from_nsstring(characters) };
                    self.input.push(InputEvent::Key {
                        code,
                        text,
                        pressed: kind == NS_EVENT_KEY_DOWN,
                        repeat: repeat == YES,
                        modifiers,
                    });
                    modifiers.command
                }
                NS_EVENT_FLAGS_CHANGED => {
                    self.input.push(InputEvent::ModifiersChanged { modifiers });
                    true
                }
                _ => true,
            }
        }

        unsafe fn translate_scroll(
            &mut self,
            event: *mut Object,
            position: LogicalPoint,
            modifiers: Modifiers,
        ) {
            let dx: f64 = msg_send![event, scrollingDeltaX];
            let dy: f64 = msg_send![event, scrollingDeltaY];
            let precise: BOOL = msg_send![event, hasPreciseScrollingDeltas];
            self.input.push(InputEvent::Scroll {
                delta: [dx, dy],
                unit: if precise == YES {
                    ScrollUnit::Points
                } else {
                    ScrollUnit::Lines
                },
                position,
                modifiers,
            });
        }

        /// Queue focus, capture-loss and geometry changes seen since the
        /// last pump. The first pump reports the initial focus and geometry.
        unsafe fn observe_window(&mut self) {
            if self.closed {
                return;
            }
            let Some(window) = self.window else {
                return;
            };
            let key: BOOL = msg_send![window.as_ptr(), isKeyWindow];
            let focused = key == YES;
            if focused != self.focused {
                if !focused && !self.buttons.is_empty() {
                    self.input.push(InputEvent::PointerCaptureLost {
                        released: self.buttons,
                    });
                    self.buttons = ButtonSet::EMPTY;
                }
                self.input.push(if focused {
                    InputEvent::FocusGained
                } else {
                    InputEvent::FocusLost
                });
                self.focused = focused;
            }
            if let Some(metrics) = unsafe { self.read_metrics() }
                && self.metrics != Some(metrics)
            {
                self.input.push(InputEvent::Resized(metrics));
                self.metrics = Some(metrics);
            }
        }

        unsafe fn read_metrics(&self) -> Option<SurfaceMetrics> {
            let window = self.window?;
            let view: *mut Object = msg_send![window.as_ptr(), contentView];
            if view.is_null() {
                return None;
            }
            let bounds: NativeRect = msg_send![view, bounds];
            let scale: f64 = msg_send![window.as_ptr(), backingScaleFactor];
            Some(SurfaceMetrics {
                logical_width: bounds.size.width,
                logical_height: bounds.size.height,
                backing_scale: scale,
                drawable_width: self.width,
                drawable_height: self.height,
            })
        }

        /// Window base coordinates to logical content coordinates.
        unsafe fn logical_from_window(
            &self,
            window: NonNull<Object>,
            location: NativePoint,
        ) -> LogicalPoint {
            let view: *mut Object = msg_send![window.as_ptr(), contentView];
            if view.is_null() {
                return LogicalPoint {
                    x: location.x,
                    y: location.y,
                };
            }
            let nil: *mut Object = std::ptr::null_mut();
            let local: NativePoint = msg_send![view, convertPoint:location fromView:nil];
            let bounds: NativeRect = msg_send![view, bounds];
            let flipped: BOOL = msg_send![view, isFlipped];
            let x = local.x - bounds.origin.x;
            let y = if flipped == YES {
                local.y - bounds.origin.y
            } else {
                bounds.origin.y + bounds.size.height - local.y
            };
            LogicalPoint { x, y }
        }

        /// Logical content coordinates to window base coordinates.
        #[cfg(feature = "synthetic-input")]
        unsafe fn window_from_logical(
            &self,
            window: NonNull<Object>,
            point: LogicalPoint,
        ) -> NativePoint {
            let view: *mut Object = msg_send![window.as_ptr(), contentView];
            if view.is_null() {
                return NativePoint {
                    x: point.x,
                    y: point.y,
                };
            }
            let bounds: NativeRect = msg_send![view, bounds];
            let flipped: BOOL = msg_send![view, isFlipped];
            let local = NativePoint {
                x: bounds.origin.x + point.x,
                y: if flipped == YES {
                    bounds.origin.y + point.y
                } else {
                    bounds.origin.y + bounds.size.height - point.y
                },
            };
            let nil: *mut Object = std::ptr::null_mut();
            msg_send![view, convertPoint:local toView:nil]
        }

        #[cfg(feature = "synthetic-input")]
        unsafe fn post_synthetic_inner(
            &mut self,
            input: &SyntheticInput,
        ) -> Result<(), PresentationError> {
            let window = self.window.ok_or(PresentationError::Closed)?;
            let number: i64 = msg_send![window.as_ptr(), windowNumber];
            let info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
            let timestamp: f64 = msg_send![info, systemUptime];
            let nil: *mut Object = std::ptr::null_mut();
            let event: *mut Object = match input {
                SyntheticInput::PointerMoved {
                    position,
                    dragging,
                    modifiers,
                } => {
                    let (kind, pressure) = match dragging {
                        None => (NS_EVENT_MOUSE_MOVED, 0.0_f32),
                        Some(PointerButton::Primary) => (NS_EVENT_LEFT_MOUSE_DRAGGED, 1.0),
                        Some(PointerButton::Secondary) => (NS_EVENT_RIGHT_MOUSE_DRAGGED, 1.0),
                        Some(_) => {
                            return Err(PresentationError::Window(
                                "synthetic drags use the primary or secondary button",
                            ));
                        }
                    };
                    let location = unsafe { self.window_from_logical(window, *position) };
                    let flags = modifiers.appkit_flags();
                    msg_send![
                        class!(NSEvent),
                        mouseEventWithType:kind
                        location:location
                        modifierFlags:flags
                        timestamp:timestamp
                        windowNumber:number
                        context:nil
                        eventNumber:0_i64
                        clickCount:0_i64
                        pressure:pressure
                    ]
                }
                SyntheticInput::Button {
                    button,
                    pressed,
                    position,
                    clicks,
                    modifiers,
                } => {
                    let kind = match (button, pressed) {
                        (PointerButton::Primary, true) => NS_EVENT_LEFT_MOUSE_DOWN,
                        (PointerButton::Primary, false) => NS_EVENT_LEFT_MOUSE_UP,
                        (PointerButton::Secondary, true) => NS_EVENT_RIGHT_MOUSE_DOWN,
                        (PointerButton::Secondary, false) => NS_EVENT_RIGHT_MOUSE_UP,
                        _ => {
                            return Err(PresentationError::Window(
                                "synthetic buttons are primary or secondary",
                            ));
                        }
                    };
                    let location = unsafe { self.window_from_logical(window, *position) };
                    let flags = modifiers.appkit_flags();
                    let clicks = i64::from(*clicks);
                    let pressure = if *pressed { 1.0_f32 } else { 0.0_f32 };
                    msg_send![
                        class!(NSEvent),
                        mouseEventWithType:kind
                        location:location
                        modifierFlags:flags
                        timestamp:timestamp
                        windowNumber:number
                        context:nil
                        eventNumber:0_i64
                        clickCount:clicks
                        pressure:pressure
                    ]
                }
                SyntheticInput::Key {
                    code,
                    text,
                    pressed,
                    repeat,
                    modifiers,
                } => {
                    let kind = if *pressed {
                        NS_EVENT_KEY_DOWN
                    } else {
                        NS_EVENT_KEY_UP
                    };
                    let characters = unsafe { nsstring(text) };
                    let flags = modifiers.appkit_flags();
                    let repeat: BOOL = if *repeat { YES } else { NO };
                    let origin = NativePoint { x: 0.0, y: 0.0 };
                    let code: u16 = *code;
                    msg_send![
                        class!(NSEvent),
                        keyEventWithType:kind
                        location:origin
                        modifierFlags:flags
                        timestamp:timestamp
                        windowNumber:number
                        context:nil
                        characters:characters
                        charactersIgnoringModifiers:characters
                        isARepeat:repeat
                        keyCode:code
                    ]
                }
                SyntheticInput::ModifiersChanged { modifiers } => {
                    let empty = unsafe { nsstring("") };
                    let flags = modifiers.appkit_flags();
                    let origin = NativePoint { x: 0.0, y: 0.0 };
                    // kVK_Shift; the code does not change the decoded flags.
                    let code: u16 = 56;
                    msg_send![
                        class!(NSEvent),
                        keyEventWithType:NS_EVENT_FLAGS_CHANGED
                        location:origin
                        modifierFlags:flags
                        timestamp:timestamp
                        windowNumber:number
                        context:nil
                        characters:empty
                        charactersIgnoringModifiers:empty
                        isARepeat:NO
                        keyCode:code
                    ]
                }
                SyntheticInput::Scroll { delta, position } => {
                    let location = unsafe { self.window_from_logical(window, *position) };
                    let screen: NativePoint =
                        msg_send![window.as_ptr(), convertPointToScreen:location];
                    let screens: *mut Object = msg_send![class!(NSScreen), screens];
                    let primary: *mut Object = msg_send![screens, firstObject];
                    if primary.is_null() {
                        return Err(PresentationError::Window(
                            "no screen for a synthetic scroll",
                        ));
                    }
                    let frame: NativeRect = msg_send![primary, frame];
                    let global = NativePoint {
                        x: screen.x,
                        y: frame.size.height - screen.y,
                    };
                    let cg = unsafe {
                        CGEventCreateScrollWheelEvent2(
                            std::ptr::null(),
                            CG_SCROLL_EVENT_UNIT_PIXEL,
                            2,
                            delta[1],
                            delta[0],
                            0,
                        )
                    };
                    if cg.is_null() {
                        return Err(PresentationError::Window("CGEvent scroll creation failed"));
                    }
                    unsafe {
                        CGEventSetLocation(cg, global);
                        CGEventSetIntegerValueField(
                            cg,
                            CG_MOUSE_EVENT_WINDOW_UNDER_POINTER,
                            number,
                        );
                        CGEventSetIntegerValueField(
                            cg,
                            CG_MOUSE_EVENT_WINDOW_UNDER_POINTER_THAT_CAN_HANDLE,
                            number,
                        );
                    }
                    let wrapped: *mut Object = msg_send![class!(NSEvent), eventWithCGEvent:cg];
                    unsafe { CFRelease(cg.cast_const()) };
                    wrapped
                }
                SyntheticInput::Focus(true) => {
                    let _: () = msg_send![self.app.as_ptr(), activateIgnoringOtherApps:YES];
                    let _: () = msg_send![window.as_ptr(), makeKeyAndOrderFront:nil];
                    if let Some(decoy) = self.focus_decoy {
                        let _: () = msg_send![decoy.as_ptr(), orderOut:nil];
                    }
                    return Ok(());
                }
                SyntheticInput::Resize { width, height } => {
                    if !(width.is_finite() && height.is_finite() && *width >= 1.0 && *height >= 1.0)
                    {
                        return Err(PresentationError::Window(
                            "a synthetic resize needs a finite size of at least one point",
                        ));
                    }
                    let size = NativeSize {
                        width: *width,
                        height: *height,
                    };
                    let _: () = msg_send![window.as_ptr(), setContentSize:size];
                    return Ok(());
                }
                SyntheticInput::Focus(false) => {
                    let decoy = match self.focus_decoy {
                        Some(decoy) => decoy,
                        None => {
                            let decoy = unsafe { focus_decoy_window()? };
                            self.focus_decoy = Some(decoy);
                            decoy
                        }
                    };
                    let _: () = msg_send![decoy.as_ptr(), makeKeyAndOrderFront:nil];
                    return Ok(());
                }
            };
            if event.is_null() {
                return Err(PresentationError::Window(
                    "synthetic NSEvent construction failed",
                ));
            }
            let _: () = msg_send![self.app.as_ptr(), postEvent:event atStart:NO];
            Ok(())
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
            #[cfg(feature = "synthetic-input")]
            if let Some(decoy) = self.focus_decoy.take() {
                let _: () = msg_send![decoy.as_ptr(), close];
                let _: () = msg_send![decoy.as_ptr(), release];
            }
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

    /// A small titled window that can take key status from the presenter.
    #[cfg(feature = "synthetic-input")]
    unsafe fn focus_decoy_window() -> Result<NonNull<Object>, PresentationError> {
        let rect = NativeRect {
            origin: NativePoint { x: 0.0, y: 0.0 },
            size: NativeSize {
                width: 160.0,
                height: 40.0,
            },
        };
        let allocated: *mut Object = msg_send![class!(NSWindow), alloc];
        if allocated.is_null() {
            return Err(PresentationError::Window("focus window allocation failed"));
        }
        let window: *mut Object = msg_send![
            allocated,
            initWithContentRect:rect
            styleMask:NS_WINDOW_STYLE_TITLED
            backing:NS_BACKING_STORE_BUFFERED
            defer:NO
        ];
        let window = NonNull::new(window).ok_or(PresentationError::Window(
            "focus window initialization failed",
        ))?;
        let _: () = msg_send![window.as_ptr(), setReleasedWhenClosed:NO];
        Ok(window)
    }

    unsafe fn flags_of(event: *mut Object) -> Modifiers {
        let flags: u64 = msg_send![event, modifierFlags];
        Modifiers::from_appkit_flags(flags)
    }

    /// Copy an `NSString` into a Rust string; nil reads as empty.
    unsafe fn string_from_nsstring(value: *mut Object) -> String {
        if value.is_null() {
            return String::new();
        }
        let bytes: *const std::ffi::c_char = msg_send![value, UTF8String];
        if bytes.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(bytes) }
            .to_string_lossy()
            .into_owned()
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
    #[cfg(feature = "synthetic-input")]
    use super::SyntheticInput;
    use super::{
        Gateway, InputEvent, PresentOutcome, PresentationConfig, PresentationError,
        PresentationPipelineInfo, PresentationState, SharedBuffer, SurfaceMetrics,
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
        pub fn poll_input(
            &mut self,
            _events: &mut Vec<InputEvent>,
        ) -> Result<PresentationState, PresentationError> {
            Err(PresentationError::Unavailable)
        }

        /// Unreachable off macOS.
        pub fn input_dropped(&self) -> u64 {
            0
        }

        /// Unreachable off macOS.
        pub fn surface_metrics(&self) -> Result<SurfaceMetrics, PresentationError> {
            Err(PresentationError::Unavailable)
        }

        /// Unreachable off macOS.
        pub fn set_resizable(&mut self, _resizable: bool) -> Result<(), PresentationError> {
            Err(PresentationError::Unavailable)
        }

        /// Unreachable off macOS.
        #[cfg(feature = "synthetic-input")]
        pub fn post_synthetic(&mut self, _input: &SyntheticInput) -> Result<(), PresentationError> {
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

    fn moved(x: f64, buttons: ButtonSet, modifiers: Modifiers) -> InputEvent {
        InputEvent::PointerMoved {
            position: LogicalPoint { x, y: 0.0 },
            buttons,
            modifiers,
        }
    }

    fn key(code: u16, pressed: bool) -> InputEvent {
        InputEvent::Key {
            code,
            text: String::new(),
            pressed,
            repeat: false,
            modifiers: Modifiers::NONE,
        }
    }

    #[test]
    fn appkit_modifier_flags_round_trip_and_ignore_device_bits() {
        let all = Modifiers {
            shift: true,
            control: true,
            option: true,
            command: true,
            caps_lock: true,
        };
        assert_eq!(all.appkit_flags(), 0x1F_0000);
        assert_eq!(Modifiers::from_appkit_flags(all.appkit_flags()), all);
        // Device-dependent left/right bits and numeric-pad/function bits.
        assert_eq!(
            Modifiers::from_appkit_flags(0x0000_FFFF | 1 << 21 | 1 << 23),
            Modifiers::NONE
        );
        let command = Modifiers::from_appkit_flags(1 << 20 | 0x8);
        assert!(command.command && !command.shift && !command.option);
    }

    #[test]
    fn appkit_button_numbers_decode_and_fit_the_set() {
        assert_eq!(PointerButton::from_appkit_number(0), PointerButton::Primary);
        assert_eq!(
            PointerButton::from_appkit_number(1),
            PointerButton::Secondary
        );
        assert_eq!(PointerButton::from_appkit_number(2), PointerButton::Middle);
        assert_eq!(
            PointerButton::from_appkit_number(7),
            PointerButton::Other(7)
        );
        assert_eq!(
            PointerButton::from_appkit_number(99),
            PointerButton::Other(31)
        );
        let set = ButtonSet::EMPTY
            .with(PointerButton::Primary)
            .with(PointerButton::Other(31));
        assert!(set.contains(PointerButton::Primary));
        assert!(!set.contains(PointerButton::Secondary));
        assert!(set.contains(PointerButton::Other(31)));
        let released = set
            .without(PointerButton::Primary)
            .without(PointerButton::Other(31));
        assert!(released.is_empty());
    }

    #[test]
    fn continuous_input_coalesces_without_reordering_discrete_events() {
        let mut queue = InputQueue::new(8);
        let held = ButtonSet::EMPTY.with(PointerButton::Primary);
        queue.push(moved(1.0, ButtonSet::EMPTY, Modifiers::NONE));
        queue.push(moved(2.0, ButtonSet::EMPTY, Modifiers::NONE));
        queue.push(moved(3.0, held, Modifiers::NONE));
        queue.push(key(49, true));
        queue.push(moved(4.0, held, Modifiers::NONE));
        let scroll = |dy: f64| InputEvent::Scroll {
            delta: [0.5, dy],
            unit: ScrollUnit::Points,
            position: LogicalPoint { x: dy, y: 0.0 },
            modifiers: Modifiers::NONE,
        };
        queue.push(scroll(1.0));
        queue.push(scroll(2.5));
        let mut events = Vec::new();
        queue.drain_into(&mut events);
        assert_eq!(
            events,
            vec![
                moved(2.0, ButtonSet::EMPTY, Modifiers::NONE),
                moved(3.0, held, Modifiers::NONE),
                key(49, true),
                moved(4.0, held, Modifiers::NONE),
                InputEvent::Scroll {
                    delta: [1.0, 3.5],
                    unit: ScrollUnit::Points,
                    position: LogicalPoint { x: 2.5, y: 0.0 },
                    modifiers: Modifiers::NONE,
                },
            ]
        );
        assert_eq!(queue.dropped(), 0);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn a_full_queue_drops_the_oldest_continuous_event_then_new_ones() {
        let mut queue = InputQueue::new(3);
        queue.push(moved(1.0, ButtonSet::EMPTY, Modifiers::NONE));
        queue.push(key(0, true));
        queue.push(key(0, false));
        queue.push(key(1, true));
        assert_eq!(queue.dropped(), 1);
        queue.push(key(1, false));
        assert_eq!(queue.dropped(), 2);
        let mut events = Vec::new();
        queue.drain_into(&mut events);
        assert_eq!(events, vec![key(0, true), key(0, false), key(1, true)]);
    }

    #[test]
    fn logical_points_map_to_drawable_pixels_by_the_axis_ratio() {
        let metrics = SurfaceMetrics {
            logical_width: 400.0,
            logical_height: 250.0,
            backing_scale: 2.0,
            drawable_width: 800,
            drawable_height: 250,
        };
        assert_eq!(
            metrics.to_drawable(LogicalPoint { x: 100.0, y: 125.0 }),
            Some([200.0, 125.0])
        );
        let empty = SurfaceMetrics {
            logical_width: 0.0,
            ..metrics
        };
        assert_eq!(empty.to_drawable(LogicalPoint { x: 1.0, y: 1.0 }), None);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn native_presenter_is_an_explicit_stub_off_macos() {
        assert!(matches!(Gateway::open(), Err(crate::Error::Unavailable)));
    }
}
