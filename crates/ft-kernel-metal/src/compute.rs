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
//! Anything richer — multiple encoders per command buffer, async completion
//! handlers, resource heaps — is deliberately absent until a consumer proves it
//! needs it. [`fused::Batch`](crate::fused::Batch) already demonstrates the
//! batched-encoder shape for the ops that ship here.
//!
//! Off macOS the whole module is a stub whose [`Gateway::open`] returns
//! [`Error::Unavailable`], so consumers compile everywhere and fall back to
//! their CPU path on non-Apple targets.

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
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::{Error, Grid, MathMode};
    use metal::{
        CompileOptions, ComputePipelineDescriptor, ComputePipelineState, Device, MTLResourceOptions,
        MTLSize,
    };
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
    /// Host data is copied in at construction and copied out by the `read_*`
    /// methods; the buffer never aliases host memory, which is what keeps this
    /// API safe without lifetime plumbing.
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
            self.buffer_from_bytes(bytes_of_f32(data))
        }

        /// Allocate a shared buffer holding a copy of `data`.
        pub fn buffer_u32(&self, data: &[u32]) -> Result<SharedBuffer, Error> {
            self.buffer_from_bytes(bytes_of_u32(data))
        }

        /// Allocate a zero-filled shared buffer of `bytes` bytes — the usual
        /// shape for a kernel's output surface.
        pub fn buffer_zeroed(&self, bytes: usize) -> Result<SharedBuffer, Error> {
            if bytes == 0 {
                return Err(Error::Kernel("zero-length buffer".into()));
            }
            let buf = self.ctx.device.new_buffer(bytes as u64, SHARED);
            // `new_buffer` does not promise zeroed contents; make it explicit so
            // an output surface a kernel only partially writes is deterministic.
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            Ok(SharedBuffer { buf, bytes })
        }

        fn buffer_from_bytes(&self, data: &[u8]) -> Result<SharedBuffer, Error> {
            if data.is_empty() {
                return Err(Error::Kernel("zero-length buffer".into()));
            }
            let buf = self.ctx.device.new_buffer_with_data(
                data.as_ptr() as *const _,
                data.len() as u64,
                SHARED,
            );
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
            let want: usize = grid.threads_per_threadgroup.iter().product();
            if want > max {
                return Err(Error::Kernel(format!(
                    "kernel '{}': {want} threads/threadgroup exceeds the pipeline maximum {max}",
                    pipeline.name
                )));
            }

            let cmd = self.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline.pso);
            for (i, b) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(&b.buf), 0);
            }
            enc.dispatch_thread_groups(msize(grid.threadgroups), msize(grid.threads_per_threadgroup));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            Ok(())
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
        /// The buffer's size in bytes.
        pub fn len_bytes(&self) -> usize {
            self.bytes
        }

        /// Copy the buffer's contents out as `f32`s. `out.len() * 4` must not
        /// exceed the buffer size.
        pub fn read_f32(&self, out: &mut [f32]) -> Result<(), Error> {
            self.check(out.len() * 4)?;
            let src =
                unsafe { std::slice::from_raw_parts(self.buf.contents() as *const f32, out.len()) };
            out.copy_from_slice(src);
            Ok(())
        }

        /// Copy the buffer's contents out as `u32`s.
        pub fn read_u32(&self, out: &mut [u32]) -> Result<(), Error> {
            self.check(out.len() * 4)?;
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

    fn msize(v: [usize; 3]) -> MTLSize {
        MTLSize::new(v[0] as u64, v[1] as u64, v[2] as u64)
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
    use super::{Error, Grid, MathMode};

    /// Stub: no Metal device exists off macOS, so no value of this type can be
    /// constructed — [`Gateway::open`] always fails.
    #[derive(Clone, Copy)]
    pub struct Gateway {
        _never: (),
    }

    /// Stub companion to the macOS [`Library`](super::Library).
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
    fn generic_dispatch_round_trip_or_unavailable() {
        let gw = match Gateway::open() {
            Ok(gw) => gw,
            // Non-macOS or headless macOS: the consumer's CPU path is the answer.
            Err(Error::Unavailable) => return,
            Err(e) => panic!("gateway open failed: {e}"),
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
    }

    #[test]
    fn a_shader_syntax_error_reads_as_a_shader_syntax_error() {
        let Ok(gw) = Gateway::open() else { return };
        let err = gw
            .library("kernel void broken( { }")
            .expect_err("must reject");
        match err {
            Error::Kernel(msg) => assert!(!msg.is_empty(), "compiler diagnostics must survive"),
            Error::Unavailable => panic!("a compile error is not an availability error"),
        }
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
}
