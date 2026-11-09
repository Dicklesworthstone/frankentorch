//! A **generic Metal compute gateway**: compile MSL, bind unified-memory
//! buffers, dispatch a grid, read results back — with every `unsafe` Metal FFI
//! call contained here, exactly as [`crate::sgemm`] and [`crate::fused`]
//! contain theirs.
//!
//! ## Why this exists
//!
//! The rest of this crate ships *frankentorch's own* kernels (GEMM, layernorm,
//! GELU, softmax, attention) behind fixed entry points. That serves consumers
//! whose GPU work is a tensor op. It does not serve a consumer whose GPU work
//! is **its own kernel** — a rasterizer's stroke signed-distance pass, an
//! image transform, a physics integrator — and such a consumer has, until now,
//! had no route to a GPU through frankentorch at all.
//!
//! That gap matters beyond convenience for any project whose dependency
//! doctrine names frankentorch as its *only* GPU gateway: without a generic
//! path, "only gateway" and "custom kernels" are contradictory, and the
//! pressure is to pull in a second, much larger GPU dependency. This module
//! removes the contradiction. The consumer supplies Metal Shading Language
//! source and flat buffers; frankentorch supplies the device, the compile, the
//! unified-memory allocation, the dispatch, and the safety boundary.
//!
//! ## The model
//!
//! Deliberately small and synchronous:
//!
//! 1. [`Gateway::open`] — the cached system device and command queue.
//! 2. [`Gateway::library`] — compile a MSL source string.
//! 3. [`Library::pipeline`] — a compute pipeline state for one `kernel`
//!    function, with its occupancy limits readable ([`Pipeline::
//!    max_threads_per_threadgroup`], [`Pipeline::thread_execution_width`]) so
//!    callers size threadgroups from pipeline introspection rather than habit.
//! 4. [`Gateway::buffer_f32`] / [`buffer_u32`](Gateway::buffer_u32) /
//!    [`buffer_zeroed`](Gateway::buffer_zeroed) — `StorageModeShared`
//!    (unified-memory) allocations. Host data is **copied in**, never aliased,
//!    so no host borrow outlives the call and the API stays sound without
//!    lifetime gymnastics.
//! 5. [`Gateway::dispatch`] — bind buffers to indices `0..n`, encode one
//!    command buffer, commit, and **wait**. Synchronous completion is what
//!    makes [`SharedBuffer::read_f32`] and friends safe to call immediately
//!    afterwards with no fences in the public API.
//!
//! 6. [`Gateway::batch`](crate::compute::Gateway::batch) — several ordered
//!    kernels in one command buffer, with each buffer registered once as an
//!    [`Input`](crate::compute::Input) (shared borrow, read only) or an
//!    [`Output`](crate::compute::Output) (exclusive borrow, may be written),
//!    each kernel's arguments declared read or written and checked for
//!    aliases, and a [`Receipt`](crate::compute::Receipt) once the command
//!    buffer completed. A consumer whose tick is
//!    a chain of kernels pays one commit and one wait for the chain, and the
//!    borrows keep the host away from every buffer until the GPU is done.
//!
//! Anything richer — asynchronous completion handlers, resource heaps — is
//! deliberately absent until a consumer proves it needs it.
//!
//! Off macOS the whole module is a stub whose [`Gateway::open`] returns
//! [`Error::Unavailable`], so consumers compile everywhere and fall back to
//! their CPU path on non-Apple targets.

#[cfg(any(test, target_os = "macos"))]
use crate::CommandBufferState;
use crate::Error;

/// How the Metal compiler is allowed to rearrange floating-point arithmetic.
///
/// **Metal's own default is `Fast`**, and that is the right default for
/// graphics and for most tensor work: it lets the compiler assume no NaNs or
/// infinities, flush denormals, contract multiply-adds, and use reduced-
/// precision reciprocals and square roots. For a kernel whose output feeds a
/// comparison — a root solver, a nearest-point search, a convergence test —
/// those assumptions are not free, and discovering them by staring at a
/// handful of wrong pixels is an expensive way to learn the default.
///
/// [`Gateway::library`] therefore compiles [`MathMode::Safe`]. Callers who
/// want the speed ask for it by name through [`Gateway::library_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathMode {
    /// IEEE-faithful: no fast-math relaxations. The default here, and the mode
    /// to reach for when the kernel's result is compared, sorted, or minimized
    /// rather than merely displayed.
    Safe,
    /// Metal's own default: fast-math relaxations enabled.
    Fast,
}

/// Dispatch geometry: threadgroup count and threadgroup size, in threads.
///
/// Both are 3-vectors in Metal's `(x, y, z)` order. Sizes must be positive;
/// [`Gateway::dispatch`] rejects a zero in either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// Number of threadgroups dispatched, `(x, y, z)`.
    pub threadgroups: [usize; 3],
    /// Threads per threadgroup, `(x, y, z)`. The product must not exceed
    /// [`Pipeline::max_threads_per_threadgroup`].
    pub threads_per_threadgroup: [usize; 3],
}

impl Grid {
    /// A 1-D grid of `groups` threadgroups of `threads` threads each.
    pub fn linear(groups: usize, threads: usize) -> Grid {
        Grid {
            threadgroups: [groups, 1, 1],
            threads_per_threadgroup: [threads, 1, 1],
        }
    }

    /// A 2-D grid: `(gx, gy)` threadgroups of `(tx, ty)` threads each.
    pub fn grid_2d(gx: usize, gy: usize, tx: usize, ty: usize) -> Grid {
        Grid {
            threadgroups: [gx, gy, 1],
            threads_per_threadgroup: [tx, ty, 1],
        }
    }

    /// Reject a zero extent up front. [`Gateway::dispatch`] calls this, but it
    /// is public so a caller computing a grid from data can check it before
    /// building buffers: a zero-extent dispatch is a silent no-op in Metal,
    /// which reads as "the kernel is wrong" rather than "the grid was empty".
    pub fn validate(&self) -> Result<(), Error> {
        let bad = self
            .threadgroups
            .iter()
            .chain(self.threads_per_threadgroup.iter())
            .any(|&n| n == 0);
        if bad {
            return Err(Error::Kernel(format!(
                "grid has a zero extent: threadgroups {:?}, threads {:?}",
                self.threadgroups, self.threads_per_threadgroup
            )));
        }
        let groups = self.threadgroup_count()?;
        let threads = self.threads_per_threadgroup_count()?;
        groups
            .checked_mul(threads)
            .ok_or(Error::SizeOverflow("dispatch thread count"))?;
        Ok(())
    }

    fn threadgroup_count(&self) -> Result<usize, Error> {
        checked_product(self.threadgroups, "threadgroup count")
    }

    fn threads_per_threadgroup_count(&self) -> Result<usize, Error> {
        checked_product(
            self.threads_per_threadgroup,
            "threads-per-threadgroup count",
        )
    }
}

fn checked_product(values: [usize; 3], what: &'static str) -> Result<usize, Error> {
    values
        .into_iter()
        .try_fold(1usize, |product, value| product.checked_mul(value))
        .ok_or(Error::SizeOverflow(what))
}

#[cfg(any(test, target_os = "macos"))]
fn checked_byte_len(
    elements: usize,
    element_bytes: usize,
    what: &'static str,
) -> Result<usize, Error> {
    elements
        .checked_mul(element_bytes)
        .ok_or(Error::SizeOverflow(what))
}

#[cfg(any(test, target_os = "macos"))]
fn validate_buffer_size(bytes: usize, maximum: usize) -> Result<(), Error> {
    if bytes == 0 {
        return Err(Error::Kernel("zero-length buffer".into()));
    }
    if bytes > maximum {
        return Err(Error::BufferTooLarge {
            requested: bytes,
            maximum,
        });
    }
    Ok(())
}

#[cfg(any(test, target_os = "macos"))]
fn ensure_command_completed(state: CommandBufferState) -> Result<(), Error> {
    if state == CommandBufferState::Completed {
        Ok(())
    } else {
        Err(Error::CommandBuffer(state))
    }
}

/// The most buffer arguments one kernel of a [`Batch`] takes: the 31
/// entries of a Metal compute function's buffer argument table.
pub const MAX_ARGUMENTS: usize = 31;

/// Batch identities, so an argument of one batch is refused by another.
static NEXT_BATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A buffer a [`Batch`] only reads, registered by [`Batch::input`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Input {
    batch: u64,
    slot: usize,
}

/// A buffer a [`Batch`] may write, registered by [`Batch::output`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Output {
    batch: u64,
    slot: usize,
}

/// One buffer argument of a kernel in a [`Batch`]: a registered buffer and
/// whether the kernel writes it.
///
/// The declaration is the caller's statement of what the kernel does. The
/// gateway cannot see inside a kernel, so it checks declarations against each
/// other (no buffer written while it is also bound elsewhere in the same
/// kernel) and against the borrows ([`Input`]s cannot be declared written),
/// not against the shader source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Argument {
    batch: u64,
    slot: usize,
    writes: bool,
}

impl Input {
    /// The buffer as an argument the kernel reads.
    #[must_use]
    pub fn read(self) -> Argument {
        Argument {
            batch: self.batch,
            slot: self.slot,
            writes: false,
        }
    }
}

impl Output {
    /// The buffer as an argument the kernel only reads.
    #[must_use]
    pub fn read(self) -> Argument {
        Argument {
            batch: self.batch,
            slot: self.slot,
            writes: false,
        }
    }

    /// The buffer as an argument the kernel writes (and may also read).
    #[must_use]
    pub fn write(self) -> Argument {
        Argument {
            batch: self.batch,
            slot: self.slot,
            writes: true,
        }
    }
}

impl Argument {
    /// Whether the kernel writes this argument.
    #[must_use]
    pub fn writes(&self) -> bool {
        self.writes
    }
}

/// What a completed [`Batch`] ran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Receipt {
    /// Kernels the command buffer ran, in order.
    pub kernels: usize,
    /// The device's time for the command buffer, in seconds: from when the
    /// GPU started executing it to when it finished (Metal's `GPUStartTime`
    /// and `GPUEndTime`). Excludes encoding, queueing and the host's wait.
    pub device_seconds: f64,
}

/// One kernel of a [`Batch`]: its pipeline, its arguments' slots in
/// argument-index order, and its grid. Only the macOS encoder reads it.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct Step<'a> {
    pipeline: &'a Pipeline,
    slots: Vec<usize>,
    grid: Grid,
}

/// A **command batch**: several kernels encoded, in order, into one command
/// buffer, committed once, and waited for once.
///
/// - **Ordered.** The kernels run in the order they were added, each one
///   after the previous one finished, so a kernel reads what earlier kernels
///   of the batch wrote. They are encoded into one serial compute encoder.
/// - **Declared bindings.** Buffers are registered once:
///   [`input`](Batch::input) borrows a buffer shared, for kernels that only
///   read it; [`output`](Batch::output) borrows it exclusively, for kernels
///   that may write it. Each kernel then names its arguments as
///   [`Input::read`], [`Output::read`] or [`Output::write`].
/// - **Retained resources.** The batch holds every registered buffer and
///   pipeline until [`submit`](Batch::submit) returns, so no host code can
///   read an output, write anything the batch uses, or free it while the
///   GPU runs. The borrow checker enforces it; Metal's command buffer also
///   retains each bound buffer.
/// - **Checked before encoding.** [`dispatch`](Batch::dispatch) refuses a
///   kernel whose grid is empty or too large for its pipeline, an argument
///   of another batch, more than [`MAX_ARGUMENTS`] arguments, or an alias:
///   one buffer bound twice in a kernel when either binding writes it (two
///   reads are compatible). A refused kernel adds nothing to the batch.
/// - **Completion receipts.** [`submit`](Batch::submit) returns a
///   [`Receipt`] only after the command buffer completed; any other terminal
///   state is [`Error::CommandBuffer`]. A failed batch is not re-run: its
///   kernels may have written buffers they read, so the caller decides.
///
/// Dropping a batch without submitting it encodes and runs nothing.
///
/// ```no_run
/// use ft_kernel_metal::compute::{Gateway, Grid};
/// # fn main() -> Result<(), ft_kernel_metal::Error> {
/// let gateway = Gateway::open()?;
/// let library = gateway.library(
///     "kernel void twice(device const float* x [[buffer(0)]],
///                        device float* y [[buffer(1)]],
///                        uint i [[thread_position_in_grid]]) { y[i] = 2.0f * x[i]; }",
/// )?;
/// let twice = library.pipeline("twice")?;
/// let x = gateway.buffer_f32(&[1.0; 64])?;
/// let mut y = gateway.buffer_zeroed(64 * 4)?;
/// let mut z = gateway.buffer_zeroed(64 * 4)?;
///
/// let mut batch = gateway.batch();
/// let (x_in, y_out, z_out) = (batch.input(&x), batch.output(&mut y), batch.output(&mut z));
/// batch.dispatch(&twice, &[x_in.read(), y_out.write()], Grid::linear(1, 64))?;
/// batch.dispatch(&twice, &[y_out.read(), z_out.write()], Grid::linear(1, 64))?;
/// let receipt = batch.submit()?;
/// assert_eq!(receipt.kernels, 2);
///
/// let mut out = [0.0f32; 64];
/// z.read_f32(&mut out)?;
/// assert_eq!(out, [4.0; 64]);
/// # Ok(())
/// # }
/// ```
///
/// A host read of an output while the batch holds it does not compile:
///
/// ```compile_fail,E0502
/// # use ft_kernel_metal::compute::{Gateway, SharedBuffer};
/// # fn race(gateway: Gateway, mut y: SharedBuffer) {
/// let mut batch = gateway.batch();
/// let y_out = batch.output(&mut y);
/// let mut seen = [0.0f32; 4];
/// let _ = y.read_f32(&mut seen); // the batch holds `y` until it is submitted
/// let _ = (y_out, batch.submit());
/// # }
/// ```
///
/// Nor does registering an output through a shared reference, which would
/// leave the host free to read it while a kernel writes it:
///
/// ```compile_fail,E0308
/// # use ft_kernel_metal::compute::{Gateway, SharedBuffer};
/// # fn race(gateway: Gateway, y: SharedBuffer) {
/// let mut batch = gateway.batch();
/// let y_out = batch.output(&y);
/// let _ = (y_out, batch.submit());
/// # }
/// ```
///
/// Nor does a host write of an input:
///
/// ```compile_fail,E0502
/// # use ft_kernel_metal::compute::{Gateway, SharedBuffer};
/// # fn race(gateway: Gateway, mut x: SharedBuffer) {
/// let mut batch = gateway.batch();
/// let x_in = batch.input(&x);
/// let _ = x.write_f32(0, &[1.0]); // the batch reads `x` until it is submitted
/// let _ = (x_in, batch.submit());
/// # }
/// ```
pub struct Batch<'a> {
    gateway: Gateway,
    id: u64,
    buffers: Vec<&'a SharedBuffer>,
    steps: Vec<Step<'a>>,
}

impl Gateway {
    /// An empty [`Batch`] on this gateway's queue.
    #[must_use]
    pub fn batch<'a>(&self) -> Batch<'a> {
        Batch {
            gateway: *self,
            id: NEXT_BATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            buffers: Vec::new(),
            steps: Vec::new(),
        }
    }
}

impl<'a> Batch<'a> {
    /// Register `buffer` for kernels that only read it, holding it shared
    /// until the batch is submitted or dropped.
    pub fn input(&mut self, buffer: &'a SharedBuffer) -> Input {
        self.buffers.push(buffer);
        Input {
            batch: self.id,
            slot: self.buffers.len() - 1,
        }
    }

    /// Register `buffer` for kernels that may write it, holding it
    /// exclusively until the batch is submitted or dropped.
    pub fn output(&mut self, buffer: &'a mut SharedBuffer) -> Output {
        self.buffers.push(buffer);
        Output {
            batch: self.id,
            slot: self.buffers.len() - 1,
        }
    }

    /// Add a kernel: `pipeline` over `grid`, with `arguments` bound to buffer
    /// indices `0..arguments.len()`. It runs after every kernel added before
    /// it.
    ///
    /// # Errors
    /// [`Error::Kernel`] or [`Error::SizeOverflow`] for an empty or
    /// overflowing grid, a threadgroup larger than the pipeline's maximum, an
    /// argument of another batch, more than [`MAX_ARGUMENTS`] arguments, or an
    /// alias. The batch is unchanged.
    pub fn dispatch(
        &mut self,
        pipeline: &'a Pipeline,
        arguments: &[Argument],
        grid: Grid,
    ) -> Result<(), Error> {
        grid.validate()?;
        let max = pipeline.max_threads_per_threadgroup();
        let want = grid.threads_per_threadgroup_count()?;
        if want > max {
            return Err(Error::Kernel(format!(
                "kernel {}: {want} threads/threadgroup exceeds the pipeline maximum {max}",
                self.steps.len()
            )));
        }
        let slots = check_arguments(self.id, self.buffers.len(), arguments)?;
        self.steps.push(Step {
            pipeline,
            slots,
            grid,
        });
        Ok(())
    }

    /// Kernels added so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether no kernel has been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Encode every kernel in order into one command buffer, commit it, and
    /// **block until the GPU finishes**, releasing the batch's buffers.
    ///
    /// # Errors
    /// [`Error::Kernel`] for a batch of no kernels, with nothing submitted;
    /// [`Error::CommandBuffer`] when the command buffer ends in any state but
    /// completed. Its outputs then hold whatever the GPU wrote before it
    /// stopped, possibly nothing.
    pub fn submit(self) -> Result<Receipt, Error> {
        if self.steps.is_empty() {
            return Err(Error::Kernel("a batch of no kernels".into()));
        }
        imp::submit_batch(self.gateway, &self.buffers, &self.steps)
    }
}

/// Check one kernel's arguments against the batch `batch` of `registered`
/// buffers, returning their slots in argument order.
fn check_arguments(
    batch: u64,
    registered: usize,
    arguments: &[Argument],
) -> Result<Vec<usize>, Error> {
    if arguments.len() > MAX_ARGUMENTS {
        return Err(Error::Kernel(format!(
            "{} buffer arguments: a kernel takes at most {MAX_ARGUMENTS}",
            arguments.len()
        )));
    }
    for (at, argument) in arguments.iter().enumerate() {
        if argument.batch != batch || argument.slot >= registered {
            return Err(Error::Kernel(format!(
                "argument {at} names a buffer of another batch"
            )));
        }
        let alias = arguments[..at].iter().position(|earlier| {
            earlier.slot == argument.slot && (earlier.writes || argument.writes)
        });
        if let Some(earlier) = alias {
            return Err(Error::Kernel(format!(
                "arguments {earlier} and {at} bind one buffer and one of them writes it: an alias"
            )));
        }
    }
    Ok(arguments.iter().map(|argument| argument.slot).collect())
}

#[cfg(target_os = "macos")]
mod imp {
    use super::{
        Error, Grid, MathMode, Receipt, Step, checked_byte_len, ensure_command_completed,
        validate_buffer_size,
    };
    use metal::{
        CompileOptions, ComputePipelineDescriptor, ComputePipelineState, Device, MTLDispatchType,
        MTLResourceOptions, MTLSize,
    };
    use objc::{msg_send, sel, sel_impl};
    use std::sync::OnceLock;

    const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

    /// The process-wide device + queue. `MTLDevice` and `MTLCommandQueue` are
    /// documented thread-safe; the raw objc pointers just aren't auto-`Send`.
    struct Ctx {
        device: Device,
        queue: metal::CommandQueue,
    }
    unsafe impl Send for Ctx {}
    unsafe impl Sync for Ctx {}

    static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

    fn ctx() -> Option<&'static Ctx> {
        CTX.get_or_init(|| {
            let device = Device::system_default()?;
            let queue = device.new_command_queue();
            Some(Ctx { device, queue })
        })
        .as_ref()
    }

    /// A handle to this machine's Metal device and command queue.
    ///
    /// Cheap to copy around: the underlying device/queue are created once per
    /// process and shared. Obtain one with [`Gateway::open`].
    #[derive(Clone, Copy)]
    pub struct Gateway {
        ctx: &'static Ctx,
    }

    /// A compiled Metal Shading Language library. Immutable after creation.
    #[derive(Debug)]
    pub struct Library {
        lib: metal::Library,
    }
    unsafe impl Send for Library {}
    unsafe impl Sync for Library {}

    /// A compute pipeline state for one `kernel` function. Immutable after
    /// creation, so it is safe to reuse across dispatches and threads.
    pub struct Pipeline {
        pso: ComputePipelineState,
        name: String,
    }
    unsafe impl Send for Pipeline {}
    unsafe impl Sync for Pipeline {}

    /// A unified-memory (`StorageModeShared`) GPU buffer.
    ///
    /// Host data is copied in at construction or by the `write_*` methods and
    /// copied out by the `read_*` methods; the buffer never aliases host
    /// memory, which is what keeps this API safe without lifetime plumbing.
    /// Every GPU use of a buffer (a [`Gateway::dispatch`], a [`Batch`](super::Batch), a
    /// presentation blit) borrows it and waits for its command buffer to
    /// complete, so a `write_*` through `&mut self` can never race the GPU.
    /// A [`Batch`](super::Batch) borrows the buffers it may write exclusively, so no host
    /// read races its kernels either.
    pub struct SharedBuffer {
        buf: metal::Buffer,
        bytes: usize,
    }
    unsafe impl Send for SharedBuffer {}
    unsafe impl Sync for SharedBuffer {}

    impl Gateway {
        /// Open the system default Metal device, or [`Error::Unavailable`] if
        /// this machine has none (or the target is not macOS).
        pub fn open() -> Result<Gateway, Error> {
            ctx().map(|ctx| Gateway { ctx }).ok_or(Error::Unavailable)
        }

        /// The device's name, e.g. `"Apple M4 Pro"` — useful to journal into a
        /// consumer's provenance record.
        pub fn device_name(&self) -> String {
            self.ctx.device.name().to_string()
        }

        /// `true` iff the device has unified memory (all Apple silicon does),
        /// meaning host and device see one physical pool and the CPU↔GPU
        /// handoff through a [`SharedBuffer`] is a pointer, not a transfer.
        pub fn has_unified_memory(&self) -> bool {
            self.ctx.device.has_unified_memory()
        }

        /// The largest threadgroup memory allocation the device permits, in
        /// bytes. Kernels that stage tile-local scratch in `threadgroup`
        /// address space must fit inside this.
        pub fn max_threadgroup_memory(&self) -> usize {
            self.ctx.device.max_threadgroup_memory_length() as usize
        }

        /// Largest shared buffer this device can allocate, in bytes.
        pub fn max_buffer_length(&self) -> usize {
            usize::try_from(self.ctx.device.max_buffer_length()).unwrap_or(usize::MAX)
        }

        pub(crate) fn metal_device(&self) -> &metal::DeviceRef {
            &self.ctx.device
        }

        pub(crate) fn metal_queue(&self) -> &metal::CommandQueueRef {
            &self.ctx.queue
        }

        /// Compile MSL source into a [`Library`] under [`MathMode::Safe`].
        ///
        /// The compiler's diagnostics come back verbatim inside
        /// [`Error::Kernel`] — a shader syntax error should read like a shader
        /// syntax error, not like "GPU unavailable".
        pub fn library(&self, source: &str) -> Result<Library, Error> {
            self.library_with(source, MathMode::Safe)
        }

        /// Compile MSL source with an explicit [`MathMode`].
        pub fn library_with(&self, source: &str, mode: MathMode) -> Result<Library, Error> {
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(mode == MathMode::Fast);
            self.ctx
                .device
                .new_library_with_source(source, &opts)
                .map(|lib| Library { lib })
                .map_err(Error::Kernel)
        }

        /// Allocate a shared buffer holding a copy of `data`.
        pub fn buffer_f32(&self, data: &[f32]) -> Result<SharedBuffer, Error> {
            checked_byte_len(
                data.len(),
                std::mem::size_of::<f32>(),
                "f32 buffer byte count",
            )?;
            self.buffer_from_bytes(bytes_of_f32(data))
        }

        /// Allocate a shared buffer holding a copy of `data`.
        pub fn buffer_u32(&self, data: &[u32]) -> Result<SharedBuffer, Error> {
            checked_byte_len(
                data.len(),
                std::mem::size_of::<u32>(),
                "u32 buffer byte count",
            )?;
            self.buffer_from_bytes(bytes_of_u32(data))
        }

        /// Allocate a zero-filled shared buffer of `bytes` bytes — the usual
        /// shape for a kernel's output surface.
        pub fn buffer_zeroed(&self, bytes: usize) -> Result<SharedBuffer, Error> {
            validate_buffer_size(bytes, self.max_buffer_length())?;
            let length = u64::try_from(bytes)
                .map_err(|_| Error::SizeOverflow("buffer allocation length"))?;
            let buf = self.ctx.device.new_buffer(length, SHARED);
            // `new_buffer` does not promise zeroed contents; make it explicit so
            // an output surface a kernel only partially writes is deterministic.
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            Ok(SharedBuffer { buf, bytes })
        }

        fn buffer_from_bytes(&self, data: &[u8]) -> Result<SharedBuffer, Error> {
            validate_buffer_size(data.len(), self.max_buffer_length())?;
            let length = u64::try_from(data.len())
                .map_err(|_| Error::SizeOverflow("buffer upload length"))?;
            let buf =
                self.ctx
                    .device
                    .new_buffer_with_data(data.as_ptr() as *const _, length, SHARED);
            Ok(SharedBuffer {
                buf,
                bytes: data.len(),
            })
        }

        /// Bind `buffers` to argument indices `0..buffers.len()`, dispatch
        /// `grid`, and **block until the GPU finishes**.
        ///
        /// Synchronous completion is the point: when this returns, every
        /// `read_*` on an output buffer observes the kernel's writes, with no
        /// fence in the public API for a caller to forget.
        pub fn dispatch(
            &self,
            pipeline: &Pipeline,
            buffers: &[&SharedBuffer],
            grid: Grid,
        ) -> Result<(), Error> {
            grid.validate()?;
            let max = pipeline.max_threads_per_threadgroup();
            let want = grid.threads_per_threadgroup_count()?;
            if want > max {
                return Err(Error::Kernel(format!(
                    "kernel '{}': {want} threads/threadgroup exceeds the pipeline maximum {max}",
                    pipeline.name
                )));
            }
            let threadgroups = msize(grid.threadgroups, "threadgroup extent")?;
            let threads_per_threadgroup = msize(
                grid.threads_per_threadgroup,
                "threads-per-threadgroup extent",
            )?;
            let mut bound = Vec::with_capacity(buffers.len());
            for (i, b) in buffers.iter().enumerate() {
                let index =
                    u64::try_from(i).map_err(|_| Error::SizeOverflow("buffer argument index"))?;
                bound.push((index, b.metal_buffer()));
            }

            // The command buffer and its encoder are autoreleased. Without a
            // pool, a thread that never exits keeps every one it dispatched:
            // about 1.8 KiB each, measured over 200,000 dispatches on an M5
            // Ultra (+368 MB). Draining a pool per dispatch releases them.
            objc::rc::autoreleasepool(|| {
                let cmd = self.ctx.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline.pso);
                for (index, buffer) in &bound {
                    enc.set_buffer(*index, Some(*buffer), 0);
                }
                enc.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
                enc.end_encoding();
                // An injected fault (tests only) leaves the buffer uncommitted,
                // as `crate::submit_and_wait` does.
                if !crate::fault::take() {
                    cmd.commit();
                    cmd.wait_until_completed();
                }
                ensure_command_completed(cmd.status().into())
            })
        }
    }

    impl Library {
        /// Build a compute pipeline state for the `kernel` function `name`.
        pub fn pipeline(&self, name: &str) -> Result<Pipeline, Error> {
            let f = self
                .lib
                .get_function(name, None)
                .map_err(|e| Error::Kernel(format!("no kernel function '{name}': {e}")))?;
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&f));
            let func = desc
                .compute_function()
                .ok_or_else(|| Error::Kernel(format!("kernel '{name}' lost its function")))?;
            let ctx = ctx().ok_or(Error::Unavailable)?;
            let pso = ctx
                .device
                .new_compute_pipeline_state_with_function(func)
                .map_err(Error::Kernel)?;
            Ok(Pipeline {
                pso,
                name: name.to_string(),
            })
        }
    }

    impl Pipeline {
        /// The largest total threadgroup size this pipeline supports on this
        /// device — the occupancy limit to size threadgroups against.
        pub fn max_threads_per_threadgroup(&self) -> usize {
            self.pso.max_total_threads_per_threadgroup() as usize
        }

        /// The device's SIMD width for this pipeline (32 on Apple silicon).
        /// Threadgroup sizes that are multiples of this avoid partial waves.
        pub fn thread_execution_width(&self) -> usize {
            self.pso.thread_execution_width() as usize
        }
    }

    impl SharedBuffer {
        pub(crate) fn metal_buffer(&self) -> &metal::BufferRef {
            &self.buf
        }

        /// The buffer's size in bytes.
        pub fn len_bytes(&self) -> usize {
            self.bytes
        }

        /// Copy the buffer's contents out as `f32`s. `out.len() * 4` must not
        /// exceed the buffer size.
        pub fn read_f32(&self, out: &mut [f32]) -> Result<(), Error> {
            let bytes = checked_byte_len(
                out.len(),
                std::mem::size_of::<f32>(),
                "f32 readback byte count",
            )?;
            self.check(bytes)?;
            let src =
                unsafe { std::slice::from_raw_parts(self.buf.contents() as *const f32, out.len()) };
            out.copy_from_slice(src);
            Ok(())
        }

        /// Copy the buffer's contents out as `u32`s.
        pub fn read_u32(&self, out: &mut [u32]) -> Result<(), Error> {
            let bytes = checked_byte_len(
                out.len(),
                std::mem::size_of::<u32>(),
                "u32 readback byte count",
            )?;
            self.check(bytes)?;
            let src =
                unsafe { std::slice::from_raw_parts(self.buf.contents() as *const u32, out.len()) };
            out.copy_from_slice(src);
            Ok(())
        }

        /// Copy the buffer's raw bytes out — the usual read for a packed
        /// 8-bit-per-channel output surface.
        pub fn read_u8(&self, out: &mut [u8]) -> Result<(), Error> {
            self.check(out.len())?;
            let src =
                unsafe { std::slice::from_raw_parts(self.buf.contents() as *const u8, out.len()) };
            out.copy_from_slice(src);
            Ok(())
        }

        /// Overwrite `data.len()` `f32`s in place, starting at element
        /// `offset`: a resident buffer updated without reallocating it. The
        /// range must lie inside the buffer; nothing is written otherwise.
        pub fn write_f32(&mut self, offset: usize, data: &[f32]) -> Result<(), Error> {
            self.write_bytes(offset, std::mem::size_of::<f32>(), bytes_of_f32(data))
        }

        /// Overwrite `data.len()` `u32`s in place, starting at element
        /// `offset`, as [`SharedBuffer::write_f32`].
        pub fn write_u32(&mut self, offset: usize, data: &[u32]) -> Result<(), Error> {
            self.write_bytes(offset, std::mem::size_of::<u32>(), bytes_of_u32(data))
        }

        fn write_bytes(&mut self, offset: usize, width: usize, data: &[u8]) -> Result<(), Error> {
            let start = offset
                .checked_mul(width)
                .ok_or(Error::SizeOverflow("buffer write offset"))?;
            let end = start
                .checked_add(data.len())
                .ok_or(Error::SizeOverflow("buffer write end"))?;
            if end > self.bytes {
                return Err(Error::Kernel(format!(
                    "write of bytes {start}..{end} into a {}-byte buffer",
                    self.bytes
                )));
            }
            // SAFETY: `start..end` lies inside the buffer's `self.bytes`
            // bytes of shared storage, and `data` is host memory that the
            // buffer never aliases. `&mut self` excludes every other borrow,
            // and no GPU work on this buffer outlives the call that borrowed
            // it, so nothing reads the range while it is written.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (self.buf.contents() as *mut u8).add(start),
                    data.len(),
                );
            }
            Ok(())
        }

        fn check(&self, want: usize) -> Result<(), Error> {
            if want > self.bytes {
                return Err(Error::Kernel(format!(
                    "read of {want} bytes from a {}-byte buffer",
                    self.bytes
                )));
            }
            Ok(())
        }
    }

    /// Encode `steps` in order into one serial compute encoder of one command
    /// buffer, submit it, and wait. Everything fallible is resolved before the
    /// command buffer exists, so a refusal encodes nothing.
    #[allow(
        unexpected_cfgs,
        reason = "objc 0.2's msg_send! expands a cfg(feature = \"cargo-clippy\") check"
    )]
    pub(super) fn submit_batch(
        gateway: Gateway,
        buffers: &[&SharedBuffer],
        steps: &[Step<'_>],
    ) -> Result<Receipt, Error> {
        let mut plan = Vec::with_capacity(steps.len());
        for step in steps {
            let mut bound = Vec::with_capacity(step.slots.len());
            for (index, &slot) in step.slots.iter().enumerate() {
                let index = u64::try_from(index)
                    .map_err(|_| Error::SizeOverflow("buffer argument index"))?;
                let buffer = buffers
                    .get(slot)
                    .ok_or_else(|| Error::Kernel(format!("slot {slot} is not registered")))?;
                bound.push((index, buffer.metal_buffer()));
            }
            plan.push((
                &step.pipeline.pso,
                bound,
                msize(step.grid.threadgroups, "threadgroup extent")?,
                msize(
                    step.grid.threads_per_threadgroup,
                    "threads-per-threadgroup extent",
                )?,
            ));
        }
        // The command buffer and its encoder are autoreleased. Draining a
        // pool here releases them when the submission ends rather than when
        // the calling thread exits; an uncommitted buffer (an injected fault)
        // would otherwise hold one of the queue's 64 slots for good.
        objc::rc::autoreleasepool(|| {
            let cmd = gateway.metal_queue().new_command_buffer();
            let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
            for (pso, bound, threadgroups, threads_per_threadgroup) in &plan {
                enc.set_compute_pipeline_state(pso);
                for (index, buffer) in bound {
                    enc.set_buffer(*index, Some(*buffer), 0);
                }
                enc.dispatch_thread_groups(*threadgroups, *threads_per_threadgroup);
            }
            enc.end_encoding();
            crate::submit_and_wait(cmd)?;
            // SAFETY: `GPUStartTime` and `GPUEndTime` are read-only
            // `CFTimeInterval` (f64) properties of a command buffer, valid
            // once it has completed, which `submit_and_wait` just verified.
            let (start, end): (f64, f64) =
                unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
            Ok(Receipt {
                kernels: plan.len(),
                device_seconds: (end - start).max(0.0),
            })
        })
    }

    fn msize(v: [usize; 3], what: &'static str) -> Result<MTLSize, Error> {
        let [x, y, z] = v.map(u64::try_from);
        Ok(MTLSize::new(
            x.map_err(|_| Error::SizeOverflow(what))?,
            y.map_err(|_| Error::SizeOverflow(what))?,
            z.map_err(|_| Error::SizeOverflow(what))?,
        ))
    }

    fn bytes_of_f32(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn bytes_of_u32(v: &[u32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::{Error, Grid, MathMode, Receipt, Step};

    /// Unreachable off macOS: no `Gateway`, so no `Batch`, can exist.
    pub(super) fn submit_batch(
        _gateway: Gateway,
        _buffers: &[&SharedBuffer],
        _steps: &[Step<'_>],
    ) -> Result<Receipt, Error> {
        Err(Error::Unavailable)
    }

    /// Stub: no Metal device exists off macOS, so no value of this type can be
    /// constructed — [`Gateway::open`] always fails.
    #[derive(Clone, Copy)]
    pub struct Gateway {
        _never: (),
    }

    /// Stub companion to the macOS [`Library`](super::Library).
    #[derive(Debug)]
    pub struct Library {
        _never: (),
    }

    /// Stub companion to the macOS [`Pipeline`](super::Pipeline).
    pub struct Pipeline {
        _never: (),
    }

    /// Stub companion to the macOS [`SharedBuffer`](super::SharedBuffer).
    pub struct SharedBuffer {
        _never: (),
    }

    impl Gateway {
        /// Always [`Error::Unavailable`] off macOS.
        pub fn open() -> Result<Gateway, Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn device_name(&self) -> String {
            String::new()
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn has_unified_memory(&self) -> bool {
            false
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn max_threadgroup_memory(&self) -> usize {
            0
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn max_buffer_length(&self) -> usize {
            0
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn library(&self, _source: &str) -> Result<Library, Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn library_with(&self, _source: &str, _mode: MathMode) -> Result<Library, Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn buffer_f32(&self, _data: &[f32]) -> Result<SharedBuffer, Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn buffer_u32(&self, _data: &[u32]) -> Result<SharedBuffer, Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn buffer_zeroed(&self, _bytes: usize) -> Result<SharedBuffer, Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `Gateway` can exist.
        pub fn dispatch(
            &self,
            _pipeline: &Pipeline,
            _buffers: &[&SharedBuffer],
            _grid: Grid,
        ) -> Result<(), Error> {
            Err(Error::Unavailable)
        }
    }

    impl Library {
        /// Unreachable off macOS: no `Library` can exist.
        pub fn pipeline(&self, _name: &str) -> Result<Pipeline, Error> {
            Err(Error::Unavailable)
        }
    }

    impl Pipeline {
        /// Unreachable off macOS: no `Pipeline` can exist.
        pub fn max_threads_per_threadgroup(&self) -> usize {
            0
        }
        /// Unreachable off macOS: no `Pipeline` can exist.
        pub fn thread_execution_width(&self) -> usize {
            0
        }
    }

    impl SharedBuffer {
        /// Unreachable off macOS: no `SharedBuffer` can exist.
        pub fn len_bytes(&self) -> usize {
            0
        }
        /// Unreachable off macOS: no `SharedBuffer` can exist.
        pub fn read_f32(&self, _out: &mut [f32]) -> Result<(), Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `SharedBuffer` can exist.
        pub fn read_u32(&self, _out: &mut [u32]) -> Result<(), Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `SharedBuffer` can exist.
        pub fn read_u8(&self, _out: &mut [u8]) -> Result<(), Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `SharedBuffer` can exist.
        pub fn write_f32(&mut self, _offset: usize, _data: &[f32]) -> Result<(), Error> {
            Err(Error::Unavailable)
        }
        /// Unreachable off macOS: no `SharedBuffer` can exist.
        pub fn write_u32(&mut self, _offset: usize, _data: &[u32]) -> Result<(), Error> {
            Err(Error::Unavailable)
        }
    }
}

pub use imp::{Gateway, Library, Pipeline, SharedBuffer};

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void scale_add(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint* n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n[0]) return;
    y[i] = x[i] * 2.0f + 1.0f;
}
"#;

    #[test]
    fn generic_dispatch_round_trip_or_unavailable() -> Result<(), Error> {
        let gw = match Gateway::open() {
            Ok(gw) => gw,
            // Non-macOS or headless macOS: the consumer's CPU path is the answer.
            Err(Error::Unavailable) => return Ok(()),
            Err(error) => return Err(error),
        };
        let n = 1000usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 3.0).collect();

        let lib = gw.library(SRC).expect("compile");
        let pso = lib.pipeline("scale_add").expect("pipeline");
        let bx = gw.buffer_f32(&x).expect("buffer x");
        let by = gw.buffer_zeroed(n * 4).expect("buffer y");
        let bn = gw.buffer_u32(&[n as u32]).expect("buffer n");

        let threads = 256.min(pso.max_threads_per_threadgroup());
        gw.dispatch(
            &pso,
            &[&bx, &by, &bn],
            Grid::linear(n.div_ceil(threads), threads),
        )
        .expect("dispatch");

        let mut got = vec![0.0f32; n];
        by.read_f32(&mut got).expect("readback");
        for (i, (g, xi)) in got.iter().zip(x.iter()).enumerate() {
            let want = xi * 2.0 + 1.0;
            assert!((g - want).abs() <= 1e-6, "idx {i}: {g} vs {want}");
        }
        Ok(())
    }

    /// A resident buffer rewritten in place between dispatches: each
    /// dispatch sees exactly the latest writes, a partial write leaves the
    /// rest as it was, and an out-of-range write is refused unchanged.
    #[test]
    fn in_place_writes_reach_the_next_dispatch() -> Result<(), Error> {
        let gw = match Gateway::open() {
            Ok(gw) => gw,
            Err(Error::Unavailable) => return Ok(()),
            Err(error) => return Err(error),
        };
        let n = 512usize;
        let lib = gw.library(SRC).expect("compile");
        let pso = lib.pipeline("scale_add").expect("pipeline");
        let mut bx = gw.buffer_zeroed(n * 4).expect("buffer x");
        let by = gw.buffer_zeroed(n * 4).expect("buffer y");
        let mut bn = gw.buffer_u32(&[0]).expect("buffer n");
        let grid = Grid::linear(n / 128, 128);
        let mut got = vec![0.0f32; n];

        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        bx.write_f32(0, &x).expect("write x");
        bn.write_u32(0, &[n as u32]).expect("write n");
        gw.dispatch(&pso, &[&bx, &by, &bn], grid).expect("dispatch");
        by.read_f32(&mut got).expect("readback");
        assert!(
            got.iter()
                .enumerate()
                .all(|(i, &g)| g == i as f32 * 2.0 + 1.0)
        );

        // Rewrite elements 100..110 only.
        bx.write_f32(100, &[-1.0; 10]).expect("partial write");
        gw.dispatch(&pso, &[&bx, &by, &bn], grid).expect("dispatch");
        by.read_f32(&mut got).expect("readback");
        for (i, &g) in got.iter().enumerate() {
            let want = if (100..110).contains(&i) {
                -1.0
            } else {
                i as f32 * 2.0 + 1.0
            };
            assert_eq!(g, want, "element {i}");
        }

        // Past the end: refused, and the buffer keeps its contents.
        assert!(bx.write_f32(n - 1, &[0.0, 0.0]).is_err());
        assert!(matches!(
            bx.write_f32(usize::MAX, &[0.0]),
            Err(Error::SizeOverflow(_))
        ));
        let mut back = vec![0.0f32; n];
        bx.read_f32(&mut back).expect("readback x");
        assert_eq!(back[n - 1], (n - 1) as f32);
        Ok(())
    }

    /// Kernels for the batch tests. `reverse_add_one` makes thread `i` read
    /// element `n - 1 - i`, which a far threadgroup of the previous kernel
    /// wrote, so a kernel that started before its predecessor finished would
    /// read stale values somewhere in a large grid.
    const BATCH_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void reverse_add_one(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint* n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint count = n[0];
    if (i >= count) return;
    y[i] = x[count - 1 - i] + 1.0f;
}
kernel void accumulate(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant uint* n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n[0]) return;
    y[i] += x[i];
}
kernel void add_pair(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint* n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n[0]) return;
    y[i] = a[i] + b[i];
}
"#;

    /// Sixteen reversing kernels alternating between two buffers and a
    /// read-write accumulation, in one command buffer: the result equals the
    /// same sequence on the CPU, exactly, and the receipt counts every
    /// kernel with a device time no longer than the host's wait.
    #[test]
    fn a_batch_runs_its_kernels_in_order_in_one_command_buffer() -> Result<(), Error> {
        let gw = match Gateway::open() {
            Ok(gw) => gw,
            Err(Error::Unavailable) => return Ok(()),
            Err(error) => return Err(error),
        };
        let n = 1usize << 20;
        let lib = gw.library(BATCH_SRC)?;
        let reverse = lib.pipeline("reverse_add_one")?;
        let accumulate = lib.pipeline("accumulate")?;
        let start: Vec<f32> = (0..n).map(|i| (i % 4093) as f32).collect();
        let mut a = gw.buffer_f32(&start)?;
        let mut b = gw.buffer_zeroed(n * 4)?;
        let count = gw.buffer_u32(&[n as u32])?;
        let grid = Grid::linear(n / 256, 256);

        let mut batch = gw.batch();
        let (a_out, b_out, count_in) = (
            batch.output(&mut a),
            batch.output(&mut b),
            batch.input(&count),
        );
        for step in 0..16 {
            let (from, to) = if step % 2 == 0 {
                (a_out, b_out)
            } else {
                (b_out, a_out)
            };
            batch.dispatch(&reverse, &[from.read(), to.write(), count_in.read()], grid)?;
        }
        batch.dispatch(
            &accumulate,
            &[a_out.write(), b_out.read(), count_in.read()],
            grid,
        )?;
        assert_eq!(batch.len(), 17);
        let began = std::time::Instant::now();
        let receipt = batch.submit()?;
        let waited = began.elapsed().as_secs_f64();
        assert_eq!(receipt.kernels, 17);
        assert!(
            receipt.device_seconds > 0.0 && receipt.device_seconds <= waited,
            "device {} s against a {waited} s wait",
            receipt.device_seconds
        );

        let (mut want_a, mut want_b) = (start, vec![0.0f32; n]);
        for step in 0..16 {
            let (from, to) = if step % 2 == 0 {
                (&want_a, &mut want_b)
            } else {
                (&want_b, &mut want_a)
            };
            let next: Vec<f32> = (0..n).map(|i| from[n - 1 - i] + 1.0).collect();
            *to = next;
        }
        for (y, x) in want_a.iter_mut().zip(&want_b) {
            *y += x;
        }
        let (mut got_a, mut got_b) = (vec![0.0f32; n], vec![0.0f32; n]);
        a.read_f32(&mut got_a)?;
        b.read_f32(&mut got_b)?;
        let wrong = got_a.iter().zip(&want_a).filter(|(g, w)| g != w).count()
            + got_b.iter().zip(&want_b).filter(|(g, w)| g != w).count();
        assert_eq!(wrong, 0, "{wrong} elements differ from the CPU sequence");
        Ok(())
    }

    /// Every refusal leaves the batch as it was; one buffer read twice by a
    /// kernel is compatible; an empty batch is refused; the batch then runs.
    #[test]
    fn a_batch_refuses_what_it_cannot_run_and_adds_nothing() -> Result<(), Error> {
        let gw = match Gateway::open() {
            Ok(gw) => gw,
            Err(Error::Unavailable) => return Ok(()),
            Err(error) => return Err(error),
        };
        let n = 1024usize;
        let lib = gw.library(BATCH_SRC)?;
        let add_pair = lib.pipeline("add_pair")?;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let a = gw.buffer_f32(&x)?;
        let mut y = gw.buffer_zeroed(n * 4)?;
        let count = gw.buffer_u32(&[n as u32])?;
        let other_count = gw.buffer_u32(&[n as u32])?;
        let grid = Grid::linear(n / 128, 128);

        assert!(matches!(
            gw.batch().submit(),
            Err(Error::Kernel(message)) if message == "a batch of no kernels"
        ));

        let mut other = gw.batch();
        let foreign = other.input(&other_count);
        let mut batch = gw.batch();
        let (a_in, y_out, count_in) = (batch.input(&a), batch.output(&mut y), batch.input(&count));
        let refusals = [
            batch.dispatch(
                &add_pair,
                &[a_in.read(), a_in.read(), y_out.write(), count_in.read()],
                Grid::linear(0, 128),
            ),
            batch.dispatch(
                &add_pair,
                &[a_in.read(), a_in.read(), y_out.write(), count_in.read()],
                Grid::linear(1, add_pair.max_threads_per_threadgroup() + 1),
            ),
            batch.dispatch(
                &add_pair,
                &[a_in.read(), a_in.read(), y_out.write(), foreign.read()],
                grid,
            ),
            batch.dispatch(
                &add_pair,
                &[a_in.read(), y_out.read(), y_out.write(), count_in.read()],
                grid,
            ),
            batch.dispatch(
                &add_pair,
                &[y_out.write(), a_in.read(), y_out.write(), count_in.read()],
                grid,
            ),
            batch.dispatch(&add_pair, &[a_in.read(); MAX_ARGUMENTS + 1], grid),
        ];
        for (at, refusal) in refusals.into_iter().enumerate() {
            assert!(refusal.is_err(), "refusal {at} was admitted");
        }
        assert!(batch.is_empty(), "a refused kernel was added");
        drop(other);

        batch.dispatch(
            &add_pair,
            &[a_in.read(), a_in.read(), y_out.write(), count_in.read()],
            grid,
        )?;
        assert_eq!(batch.submit()?.kernels, 1);
        let mut got = vec![0.0f32; n];
        y.read_f32(&mut got)?;
        assert!(got.iter().enumerate().all(|(i, &g)| g == 2.0 * i as f32));
        Ok(())
    }

    #[test]
    fn argument_checks_name_the_alias_and_the_foreign_batch() {
        let read = |slot| Argument {
            batch: 7,
            slot,
            writes: false,
        };
        let write = |slot| Argument {
            batch: 7,
            slot,
            writes: true,
        };
        assert_eq!(
            check_arguments(7, 3, &[read(0), read(0), write(1), read(2)]),
            Ok(vec![0, 0, 1, 2])
        );
        assert_eq!(
            check_arguments(7, 3, &[read(1), read(0), write(1)]),
            Err(Error::Kernel(
                "arguments 0 and 2 bind one buffer and one of them writes it: an alias".into()
            ))
        );
        assert_eq!(
            check_arguments(7, 3, &[write(2), write(2)]),
            Err(Error::Kernel(
                "arguments 0 and 1 bind one buffer and one of them writes it: an alias".into()
            ))
        );
        let foreign = Argument {
            batch: 8,
            slot: 0,
            writes: false,
        };
        assert_eq!(
            check_arguments(7, 3, &[read(0), foreign]),
            Err(Error::Kernel(
                "argument 1 names a buffer of another batch".into()
            ))
        );
        assert!(check_arguments(7, 3, &[read(0); MAX_ARGUMENTS]).is_ok());
        assert!(check_arguments(7, 3, &[read(0); MAX_ARGUMENTS + 1]).is_err());
    }

    /// A batch whose command buffer fails reports its state, counts the
    /// failure, and is not re-run: its output keeps what it held. Seventy
    /// failed batches in a row (more than the queue's 64 command buffers)
    /// release their buffers, so the next batch still runs.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_failed_batch_reports_its_state_and_releases_its_command_buffer() -> Result<(), Error> {
        let gw = match Gateway::open() {
            Ok(gw) => gw,
            Err(Error::Unavailable) => return Ok(()),
            Err(error) => return Err(error),
        };
        let n = 256usize;
        let lib = gw.library(BATCH_SRC)?;
        let add_pair = lib.pipeline("add_pair")?;
        let a = gw.buffer_f32(&vec![1.0; n])?;
        let mut y = gw.buffer_f32(&vec![-5.0; n])?;
        let count = gw.buffer_u32(&[n as u32])?;
        let grid = Grid::linear(n / 64, 64);
        let run = |y: &mut SharedBuffer| {
            let mut batch = gw.batch();
            let (a_in, y_out, count_in) = (batch.input(&a), batch.output(y), batch.input(&count));
            batch.dispatch(
                &add_pair,
                &[a_in.read(), a_in.read(), y_out.write(), count_in.read()],
                grid,
            )?;
            batch.submit()
        };

        let before = crate::command_buffer_failures();
        crate::fault::arm(70);
        for attempt in 0..70 {
            assert_eq!(
                run(&mut y),
                Err(Error::CommandBuffer(crate::CommandBufferState::NotEnqueued)),
                "attempt {attempt}"
            );
        }
        assert_eq!(crate::fault::pending(), 0);
        // The counter is process-wide, and other tests' injected faults
        // land in it concurrently.
        assert!(crate::command_buffer_failures() >= before + 70);
        let mut got = vec![0.0f32; n];
        y.read_f32(&mut got)?;
        assert!(got.iter().all(|&g| g == -5.0), "a failed batch wrote");

        assert_eq!(run(&mut y)?.kernels, 1);
        y.read_f32(&mut got)?;
        assert!(got.iter().all(|&g| g == 2.0));
        Ok(())
    }

    /// Seventy dispatches whose command buffers fail (more than the queue's
    /// 64) each report their state and leave their output unwritten, and the
    /// next dispatch still runs: each dispatch releases its command buffer
    /// when it returns rather than when its thread exits.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_dispatch_releases_its_command_buffer_when_it_returns() -> Result<(), Error> {
        let gw = match Gateway::open() {
            Ok(gw) => gw,
            Err(Error::Unavailable) => return Ok(()),
            Err(error) => return Err(error),
        };
        let n = 256usize;
        let pso = gw.library(SRC)?.pipeline("scale_add")?;
        let x = gw.buffer_f32(&vec![3.0; n])?;
        let y = gw.buffer_f32(&vec![-5.0; n])?;
        let count = gw.buffer_u32(&[n as u32])?;
        let grid = Grid::linear(n / 64, 64);

        crate::fault::arm(70);
        for attempt in 0..70 {
            assert_eq!(
                gw.dispatch(&pso, &[&x, &y, &count], grid),
                Err(Error::CommandBuffer(crate::CommandBufferState::NotEnqueued)),
                "attempt {attempt}"
            );
        }
        assert_eq!(crate::fault::pending(), 0);
        let mut got = vec![0.0f32; n];
        y.read_f32(&mut got)?;
        assert!(got.iter().all(|&g| g == -5.0), "a failed dispatch wrote");

        gw.dispatch(&pso, &[&x, &y, &count], grid)?;
        y.read_f32(&mut got)?;
        assert!(got.iter().all(|&g| g == 7.0));
        Ok(())
    }

    #[test]
    fn a_shader_syntax_error_reads_as_a_shader_syntax_error() -> Result<(), String> {
        let Ok(gw) = Gateway::open() else {
            return Ok(());
        };
        let err = gw
            .library("kernel void broken( { }")
            .expect_err("must reject");
        match err {
            Error::Kernel(msg) => assert!(!msg.is_empty(), "compiler diagnostics must survive"),
            other => return Err(format!("a compile error has the wrong type: {other}")),
        }
        Ok(())
    }

    #[test]
    fn oversized_threadgroups_are_rejected_before_dispatch() {
        let Ok(gw) = Gateway::open() else { return };
        let lib = gw.library(SRC).expect("compile");
        let pso = lib.pipeline("scale_add").expect("pipeline");
        let b = gw.buffer_f32(&[1.0, 2.0]).expect("buffer");
        let bn = gw.buffer_u32(&[2]).expect("buffer n");
        let too_many = pso.max_threads_per_threadgroup() + 1;
        let err = gw
            .dispatch(&pso, &[&b, &b, &bn], Grid::linear(1, too_many))
            .expect_err("must reject");
        assert!(matches!(err, Error::Kernel(_)));
    }

    #[test]
    fn a_zero_extent_grid_is_an_error_not_a_silent_no_op() {
        let g = Grid {
            threadgroups: [4, 0, 1],
            threads_per_threadgroup: [8, 8, 1],
        };
        assert!(g.validate().is_err());
    }

    #[test]
    fn pathological_grid_products_fail_before_dispatch() {
        let group_overflow = Grid {
            threadgroups: [usize::MAX, 2, 1],
            threads_per_threadgroup: [1, 1, 1],
        };
        assert_eq!(
            group_overflow.validate(),
            Err(Error::SizeOverflow("threadgroup count"))
        );

        let threadgroup_overflow = Grid {
            threadgroups: [1, 1, 1],
            threads_per_threadgroup: [usize::MAX, 2, 1],
        };
        assert_eq!(
            threadgroup_overflow.validate(),
            Err(Error::SizeOverflow("threads-per-threadgroup count"))
        );

        let total_overflow = Grid {
            threadgroups: [usize::MAX, 1, 1],
            threads_per_threadgroup: [2, 1, 1],
        };
        assert_eq!(
            total_overflow.validate(),
            Err(Error::SizeOverflow("dispatch thread count"))
        );
    }

    #[test]
    fn typed_byte_products_and_device_limits_fail_closed() {
        assert_eq!(
            checked_byte_len(usize::MAX, 4, "test byte count"),
            Err(Error::SizeOverflow("test byte count"))
        );
        assert_eq!(
            validate_buffer_size(1025, 1024),
            Err(Error::BufferTooLarge {
                requested: 1025,
                maximum: 1024,
            })
        );
    }

    #[test]
    fn injected_terminal_command_states_are_typed() {
        assert_eq!(
            ensure_command_completed(CommandBufferState::Completed),
            Ok(())
        );
        for state in [
            CommandBufferState::NotEnqueued,
            CommandBufferState::Enqueued,
            CommandBufferState::Committed,
            CommandBufferState::Scheduled,
            CommandBufferState::Error,
        ] {
            assert_eq!(
                ensure_command_completed(state),
                Err(Error::CommandBuffer(state))
            );
        }
    }

    #[test]
    fn device_buffer_limit_is_checked_before_allocation_or_unavailable() {
        let Ok(gateway) = Gateway::open() else { return };
        let maximum = gateway.max_buffer_length();
        assert!(maximum > 0, "Metal reported a zero buffer limit");
        let requested = maximum
            .checked_add(1)
            .expect("Metal buffer limit must leave a representable refusal");
        assert_eq!(
            gateway.buffer_zeroed(requested).err(),
            Some(Error::BufferTooLarge { requested, maximum })
        );
    }
}
