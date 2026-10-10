//! Metal GPU compute kernels for FrankenTorch (macOS / Apple Silicon).
//!
//! This crate is the **sanctioned unsafe boundary** for Metal FFI: it wraps
//! `metal-rs` behind a small, safe API ([`is_available`], [`sgemm`]) so that
//! `#![deny(unsafe_code)]` consumers (e.g. `franken_whisper`) can offload their
//! hot matmuls to the GPU without themselves touching `unsafe`.
//!
//! On non-macOS targets the whole thing degrades to a stub: [`is_available`]
//! returns `false` and [`sgemm`] returns [`Error::Unavailable`], so callers
//! transparently fall back to their CPU kernel. That keeps this crate — and any
//! workspace that contains it — building on Linux/Windows.
//!
//! ## Kernel
//!
//! A shared-memory + register-blocked tiled f32 GEMM (`BM=BN=64, BK=16, 4x4`
//! micro-tiles, 256 threads/threadgroup) — benchmarked at ~1.5–1.9 TFLOP/s on an
//! M4 Pro, ~5–8× the CPU. The device / command queue / compiled pipeline are
//! built once and cached; each call allocates unified-memory (`StorageModeShared`)
//! buffers and dispatches a single command buffer.

#![allow(unsafe_code)]

use std::fmt;

/// Terminal or unexpected state observed after waiting for a Metal command
/// buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferState {
    /// The command buffer was never enqueued.
    NotEnqueued,
    /// The command buffer was enqueued but not committed.
    Enqueued,
    /// The command buffer was committed but did not reach scheduling.
    Committed,
    /// The command buffer was scheduled but did not complete.
    Scheduled,
    /// Every encoded command completed successfully.
    Completed,
    /// Metal reported an execution failure.
    Error,
}

#[cfg(target_os = "macos")]
impl From<metal::MTLCommandBufferStatus> for CommandBufferState {
    fn from(status: metal::MTLCommandBufferStatus) -> Self {
        match status {
            metal::MTLCommandBufferStatus::NotEnqueued => Self::NotEnqueued,
            metal::MTLCommandBufferStatus::Enqueued => Self::Enqueued,
            metal::MTLCommandBufferStatus::Committed => Self::Committed,
            metal::MTLCommandBufferStatus::Scheduled => Self::Scheduled,
            metal::MTLCommandBufferStatus::Completed => Self::Completed,
            metal::MTLCommandBufferStatus::Error => Self::Error,
        }
    }
}

/// Reason a Metal operation could not run. Callers may fall back to a CPU
/// implementation, but should retain the typed cause in provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// No usable Metal device on this machine/target.
    Unavailable,
    /// A Metal call failed (shape rejected, pipeline error, …).
    Kernel(String),
    /// A host-side count or byte calculation overflowed.
    SizeOverflow(&'static str),
    /// A requested buffer exceeds the selected device's allocation limit.
    BufferTooLarge {
        /// Requested buffer bytes.
        requested: usize,
        /// Maximum bytes reported by `MTLDevice.maxBufferLength`.
        maximum: usize,
    },
    /// A synchronous command buffer did not complete successfully.
    CommandBuffer(CommandBufferState),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unavailable => write!(f, "Metal GPU unavailable"),
            Error::Kernel(m) => write!(f, "Metal kernel error: {m}"),
            Error::SizeOverflow(what) => write!(f, "Metal {what} exceeds this platform"),
            Error::BufferTooLarge { requested, maximum } => write!(
                f,
                "Metal buffer request is {requested} bytes; device limit is {maximum} bytes"
            ),
            Error::CommandBuffer(state) => {
                write!(f, "Metal command buffer ended in {state:?}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Command buffers this process committed that ended in a state other than
/// [`CommandBufferState::Completed`], counted by [`sgemm`] and
/// `fused::Batch::finish`. A failed command buffer's output buffers hold
/// whatever the GPU wrote before it stopped (often nothing), so these calls
/// never return them; see [`COMMAND_BUFFER_ATTEMPTS`].
static COMMAND_BUFFER_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many failed command buffers this process has seen (see
/// [`COMMAND_BUFFER_ATTEMPTS`]). Monotonic; callers diff it around a call to
/// learn whether that call needed a retry.
#[must_use]
pub fn command_buffer_failures() -> u64 {
    COMMAND_BUFFER_FAILURES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Attempts [`sgemm`] and the fused encoder make for one command buffer
/// before reporting [`Error::CommandBuffer`].
///
/// A command buffer can end in `Error` through no fault of its own: when
/// another process hangs the GPU, macOS restarts it and aborts every command
/// buffer in flight, ours included (WindowServer then logs "Flushing display
/// space to aid GPU restart recovery"). Re-encoding the same work from the same
/// resident inputs on the recovered GPU gives bit-identical results, because
/// these kernels have no data-dependent scheduling. Waits between attempts
/// grow from 50 ms so a recovery in progress can finish.
///
/// A thread whose last operation used up all its attempts makes one attempt
/// per operation, without waiting, until one of its command buffers completes
/// again, so a GPU that stays down costs each later call one failed submission
/// rather than four and 1.05 s of waits.
pub const COMMAND_BUFFER_ATTEMPTS: u32 = 4;

#[cfg(target_os = "macos")]
thread_local! {
    /// Set when an operation on this thread used up its attempts; cleared when
    /// a command buffer this thread submitted completes.
    static GPU_DOWN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Commit `cmd`, wait for it, and report its terminal state. The single
/// submission point for the command buffers whose outputs callers read back.
#[cfg(target_os = "macos")]
pub(crate) fn submit_and_wait(cmd: &metal::CommandBufferRef) -> Result<(), Error> {
    // An injected fault (tests only) skips the commit: the GPU writes nothing
    // and the buffer reports `NotEnqueued`, which is what an aborted command
    // buffer leaves behind (outputs unwritten, state not `Completed`).
    if !fault::take() {
        cmd.commit();
        cmd.wait_until_completed();
    }
    let state: CommandBufferState = cmd.status().into();
    if state == CommandBufferState::Completed {
        GPU_DOWN.with(|down| down.set(false));
        Ok(())
    } else {
        COMMAND_BUFFER_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Err(Error::CommandBuffer(state))
    }
}

/// Run one GPU operation: `encode_and_submit` encodes a command buffer from
/// inputs it does not write, submits it through [`submit_and_wait`], and reads
/// its outputs only when that succeeds. After a failed command buffer the
/// operation is re-run, 50, 200 and 800 ms apart, up to
/// [`COMMAND_BUFFER_ATTEMPTS`] times in all; then the error is returned.
///
/// The re-runs are for a GPU that recovers. One that stays down (macOS keeps
/// restarting it, or has revoked this process's GPU access after repeated
/// faults) would otherwise cost every later operation four failed submissions
/// and 1.05 s of waits before its caller falls back to the CPU, and a
/// franken_whisper large-v3-turbo window whose encoder falls back to the CPU
/// still submits about ninety Metal GEMMs from its decoding thread (measured
/// on jfk.wav). So once an operation on this thread has used up its attempts,
/// the thread's later operations make one attempt each, without waiting,
/// until one of its command buffers completes again. The memory is per
/// thread: it needs no synchronization, each thread pays the full attempts at
/// most once per outage, and one thread's state never changes what another
/// observes.
#[cfg(target_os = "macos")]
pub(crate) fn with_command_buffer_attempts<T>(
    mut encode_and_submit: impl FnMut() -> Result<T, Error>,
) -> Result<T, Error> {
    let mut failed = 0;
    loop {
        // The attempt's command buffer and encoders are autoreleased objects,
        // and a thread without a pool keeps them until it exits. A buffer
        // that was never committed (an injected fault) never completes, so
        // it keeps one of its queue's 64 slots until released, and once the
        // slots are used up the queue blocks every caller, on any thread.
        // Draining a pool per attempt releases them when the attempt ends;
        // what the attempt returns is owned, not autoreleased.
        match objc::rc::autoreleasepool(&mut encode_and_submit) {
            Err(Error::CommandBuffer(state)) => {
                failed += 1;
                let attempts = if GPU_DOWN.with(std::cell::Cell::get) {
                    1
                } else {
                    COMMAND_BUFFER_ATTEMPTS
                };
                if failed >= attempts {
                    GPU_DOWN.with(|down| down.set(true));
                    return Err(Error::CommandBuffer(state));
                }
                retry_backoff(failed);
            }
            other => return other,
        }
    }
}

/// Back off before re-submitting after the `failed`-th failure (1-based):
/// 50 ms, 200 ms, 800 ms.
#[cfg(target_os = "macos")]
pub(crate) fn retry_backoff(failed: u32) {
    let step = failed.clamp(1, 4) - 1;
    std::thread::sleep(std::time::Duration::from_millis(50u64 << (2 * step)));
}

/// Fault injection for [`submit_and_wait`]: the next `n` command buffers
/// submitted from the arming thread are not committed. Thread-local, so
/// parallel tests cannot consume each other's faults, and nothing is armed
/// unless a test calls [`inject_command_buffer_faults`].
///
/// Compiled only into this crate's own tests and builds with debug assertions
/// (`cargo test`, `cargo build`). A release build has no seam: `take` is a
/// constant `false` and nothing can arm a fault. The gate is a cfg rather than
/// a Cargo feature because a dependent naming a feature fails to resolve
/// against a sibling checkout that predates it.
#[cfg(target_os = "macos")]
pub(crate) mod fault {
    #[cfg(any(test, debug_assertions))]
    thread_local! {
        static PENDING: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// Arm `n` faults on this thread.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn arm(n: u32) {
        PENDING.with(|p| p.set(n));
    }

    /// Consume one armed fault, if any.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn take() -> bool {
        PENDING.with(|p| {
            let n = p.get();
            if n > 0 {
                p.set(n - 1);
            }
            n > 0
        })
    }

    /// Faults still armed on this thread.
    #[cfg(test)]
    pub(crate) fn pending() -> u32 {
        PENDING.with(std::cell::Cell::get)
    }

    #[cfg(not(any(test, debug_assertions)))]
    pub(crate) const fn take() -> bool {
        false
    }
}

/// Test seam (not for production use): the next `n` command buffers that
/// [`sgemm`] or the fused encoder submit from the calling thread end without
/// running and report `NotEnqueued`, leaving their outputs unwritten, as a GPU
/// restart leaves aborted work. `0` disarms.
///
/// Returns whether the seam exists in this build: only with debug assertions
/// on macOS. A release build compiles it out, so this arms nothing and returns
/// `false`; a dependent's test must check the result rather than assume its
/// faults were injected. To exercise it from a release-profile test run, pass
/// `--config 'profile.release.package.ft-kernel-metal.debug-assertions=true'`.
/// The function itself always exists, so a dependent's tests build against
/// any checkout of this crate that has it, without naming a Cargo feature.
#[doc(hidden)]
pub fn inject_command_buffer_faults(n: u32) -> bool {
    #[cfg(all(target_os = "macos", any(test, debug_assertions)))]
    {
        fault::arm(n);
        true
    }
    #[cfg(not(all(target_os = "macos", any(test, debug_assertions))))]
    {
        let _ = n;
        false
    }
}

/// GPU-resident fused encoder ops (macOS only): keeps activations on the GPU and
/// batches a run of ops into one command buffer / one sync. See [`fused::Batch`].
#[cfg(target_os = "macos")]
pub mod fused;

/// The **generic Metal compute gateway**: compile caller-supplied MSL, bind
/// unified-memory buffers, dispatch, read back — for consumers whose GPU work
/// is their own kernel rather than one of this crate's tensor ops. Present on
/// every target (a stub off macOS), so callers compile everywhere and fall back
/// to their CPU path where no device exists. See [`compute::Gateway`].
pub mod compute;

/// A native macOS preview surface backed by `CAMetalLayer`.
///
/// [`presentation::NativePresenter`] accepts a unified-memory
/// [`compute::SharedBuffer`] containing RGBA8 pixels and presents it without
/// copying the pixels back into host-owned memory. AppKit, Objective-C,
/// drawables, and command-buffer lifetimes remain inside this crate's sanctioned
/// unsafe boundary. Off macOS, construction reports
/// [`presentation::PresentationError::Unavailable`].
pub mod presentation;

#[cfg(target_os = "macos")]
mod imp {
    use super::Error;
    use metal::{
        CompileOptions, ComputePipelineDescriptor, ComputePipelineState, Device,
        MTLResourceOptions, MTLSize,
    };
    use std::sync::OnceLock;

    const TILED_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
kernel void matmul_tiled(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]])
{
    const uint M=dims[0],K=dims[1],N=dims[2];
    threadgroup float As[BK][BM]; threadgroup float Bs[BK][BN];
    const uint tRow=lid/(BN/TN), tCol=lid%(BN/TN);
    const uint rowBase=gid.y*BM, colBase=gid.x*BN;
    float acc[TM][TN]; for (uint i=0;i<TM;i++) for (uint j=0;j<TN;j++) acc[i][j]=0.0f;
    for (uint k0=0;k0<K;k0+=BK) {
        for (uint i=lid;i<BM*BK;i+=256){uint m=i/BK,k=i%BK;uint gr=rowBase+m,gk=k0+k;
            As[k][m]=(gr<M&&gk<K)?A[gr*K+gk]:0.0f;}
        for (uint i=lid;i<BK*BN;i+=256){uint k=i/BN,n=i%BN;uint gk=k0+k,gc=colBase+n;
            Bs[k][n]=(gk<K&&gc<N)?B[gk*N+gc]:0.0f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k=0;k<BK;k++){float aReg[TM],bReg[TN];
            for (uint i=0;i<TM;i++) aReg[i]=As[k][tRow*TM+i];
            for (uint j=0;j<TN;j++) bReg[j]=Bs[k][tCol*TN+j];
            for (uint i=0;i<TM;i++) for (uint j=0;j<TN;j++) acc[i][j]+=aReg[i]*bReg[j];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i=0;i<TM;i++){uint gr=rowBase+tRow*TM+i; if(gr>=M) continue;
        for (uint j=0;j<TN;j++){uint gc=colBase+tCol*TN+j; if(gc<N) C[gr*N+gc]=acc[i][j];}}
}
"#;

    /// Cached GPU context. Metal `Device`/`CommandQueue` are internally
    /// thread-safe and `ComputePipelineState` is immutable after creation, so
    /// sharing one instance across threads is sound; the raw objc pointers just
    /// aren't auto-`Send`/`Sync`, which we assert here (the sanctioned boundary).
    struct Ctx {
        device: Device,
        queue: metal::CommandQueue,
        pso: ComputePipelineState,
    }
    unsafe impl Send for Ctx {}
    unsafe impl Sync for Ctx {}

    static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

    fn ctx() -> Option<&'static Ctx> {
        CTX.get_or_init(|| {
            let device = Device::system_default()?;
            let lib = device
                .new_library_with_source(TILED_SRC, &CompileOptions::new())
                .ok()?;
            let f = lib.get_function("matmul_tiled", None).ok()?;
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&f));
            let pso = device
                .new_compute_pipeline_state_with_function(desc.compute_function()?)
                .ok()?;
            let queue = device.new_command_queue();
            Some(Ctx { device, queue, pso })
        })
        .as_ref()
    }

    pub fn is_available() -> bool {
        ctx().is_some()
    }

    pub fn sgemm(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), Error> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(Error::Kernel(format!(
                "shape mismatch: a={} (want {}), b={} (want {}), c={} (want {})",
                a.len(),
                m * k,
                b.len(),
                k * n,
                c.len(),
                m * n
            )));
        }
        if m == 0 || k == 0 || n == 0 {
            return Ok(());
        }
        let ctx = ctx().ok_or(Error::Unavailable)?;
        let opts = MTLResourceOptions::StorageModeShared;
        let ba =
            ctx.device
                .new_buffer_with_data(a.as_ptr() as *const _, (a.len() * 4) as u64, opts);
        let bb =
            ctx.device
                .new_buffer_with_data(b.as_ptr() as *const _, (b.len() * 4) as u64, opts);
        let bc = ctx.device.new_buffer((m * n * 4) as u64, opts);
        let dims = [m as u32, k as u32, n as u32];
        let bd = ctx
            .device
            .new_buffer_with_data(dims.as_ptr() as *const _, 12, opts);

        // Inputs are read-only to the kernel and every output element is
        // rewritten, so a failed command buffer is re-run from the same
        // buffers (see `with_command_buffer_attempts`).
        super::with_command_buffer_attempts(|| {
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.pso);
            enc.set_buffer(0, Some(&ba), 0);
            enc.set_buffer(1, Some(&bb), 0);
            enc.set_buffer(2, Some(&bc), 0);
            enc.set_buffer(3, Some(&bd), 0);
            let tg = MTLSize::new(n.div_ceil(64) as u64, m.div_ceil(64) as u64, 1);
            enc.dispatch_thread_groups(tg, MTLSize::new(256, 1, 1));
            enc.end_encoding();
            super::submit_and_wait(cmd)
        })?;

        // Unified memory: read the result straight out of the shared buffer.
        let out = unsafe { std::slice::from_raw_parts(bc.contents() as *const f32, m * n) };
        c.copy_from_slice(out);
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::Error;
    pub fn is_available() -> bool {
        false
    }
    pub fn sgemm(
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<(), Error> {
        Err(Error::Unavailable)
    }
}

/// `true` iff a usable Metal GPU + compiled GEMM pipeline exist on this machine.
/// Always `false` off macOS. Cheap after the first call (result is cached).
pub fn is_available() -> bool {
    imp::is_available()
}

/// Compute `C[m,n] = A[m,k] * B[k,n]` (row-major, f32) on the GPU.
///
/// `a` must be `m*k`, `b` must be `k*n`, `c` must be `m*n`; `c` is fully
/// overwritten on success and left untouched on error. Returns
/// [`Error::Unavailable`] if there's no GPU (callers should fall back to a CPU
/// kernel), [`Error::Kernel`] on a shape error, or [`Error::CommandBuffer`] when
/// the GPU did not complete the work in its attempts (see
/// [`COMMAND_BUFFER_ATTEMPTS`]).
pub fn sgemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), Error> {
    imp::sgemm(a, b, c, m, k, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for e in 0..k {
                    acc += a[i * k + e] * b[e * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn sgemm_matches_cpu_or_unavailable() -> Result<(), Error> {
        // Shapes incl. non-multiples of 64 and whisper encoder-ish dims.
        let shapes = [(2, 3, 2), (64, 64, 64), (65, 33, 129), (300, 384, 512)];
        for &(m, k, n) in &shapes {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32) * 0.1 - 0.6)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 5 % 11) as f32) * 0.1 - 0.5)
                .collect();
            let mut c = vec![0.0f32; m * n];
            match sgemm(&a, &b, &mut c, m, k, n) {
                Ok(()) => {
                    let want = cpu_ref(&a, &b, m, k, n);
                    for (idx, (g, w)) in c.iter().zip(want.iter()).enumerate() {
                        let tol = 1e-3 * (1.0 + w.abs());
                        assert!(
                            (g - w).abs() <= tol,
                            "shape {m}x{k}x{n} idx {idx}: gpu {g} vs cpu {w}"
                        );
                    }
                }
                Err(Error::Unavailable) => { /* non-macOS or no GPU: fine */ }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sgemm_reruns_failed_command_buffers_or_unavailable() {
        if !is_available() {
            return;
        }
        let (m, k, n) = (65, 33, 129);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
        let mut clean = vec![0.0f32; m * n];
        sgemm(&a, &b, &mut clean, m, k, n).unwrap();
        for faults in 1..COMMAND_BUFFER_ATTEMPTS {
            let before = command_buffer_failures();
            fault::arm(faults);
            let mut got = vec![f32::NAN; m * n];
            let result = sgemm(&a, &b, &mut got, m, k, n);
            fault::arm(0);
            assert_eq!(result, Ok(()));
            let differing = got
                .iter()
                .zip(&clean)
                .filter(|(g, c)| g.to_bits() != c.to_bits())
                .count();
            assert_eq!(differing, 0, "{faults} failed buffer(s) leaked into c");
            assert!(command_buffer_failures() >= before + u64::from(faults));
        }
        // Every attempt failing is an error, and `c` is left untouched.
        fault::arm(COMMAND_BUFFER_ATTEMPTS);
        let mut got = vec![f32::NAN; m * n];
        let result = sgemm(&a, &b, &mut got, m, k, n);
        fault::arm(0);
        assert_eq!(
            result,
            Err(Error::CommandBuffer(CommandBufferState::NotEnqueued))
        );
        assert!(got.iter().all(|v| v.is_nan()), "a failed sgemm wrote c");
    }

    /// After an operation on this thread used up its attempts, later ones
    /// make a single attempt (no re-run, so no waits) until a command buffer
    /// completes; then the re-runs are back.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_thread_whose_gpu_stayed_down_stops_retrying_until_a_buffer_completes_or_unavailable() {
        if !is_available() {
            return;
        }
        let (m, k, n) = (65, 33, 129);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
        let mut clean = vec![0.0f32; m * n];
        sgemm(&a, &b, &mut clean, m, k, n).unwrap();
        let down = Err(Error::CommandBuffer(CommandBufferState::NotEnqueued));

        // The GPU goes down: one operation uses up all its attempts.
        fault::arm(COMMAND_BUFFER_ATTEMPTS);
        let mut got = vec![f32::NAN; m * n];
        assert_eq!(sgemm(&a, &b, &mut got, m, k, n), down);
        assert_eq!(fault::pending(), 0);

        // Each later operation on this thread tries once: of two armed
        // faults it consumes one and reports the failure.
        for _ in 0..3 {
            fault::arm(2);
            assert_eq!(sgemm(&a, &b, &mut got, m, k, n), down);
            assert_eq!(fault::pending(), 1, "a re-run after the GPU stayed down");
        }
        fault::arm(0);
        assert!(got.iter().all(|v| v.is_nan()), "a failed sgemm wrote c");

        // A command buffer that completes restores the re-runs.
        sgemm(&a, &b, &mut got, m, k, n).unwrap();
        fault::arm(1);
        let mut rerun = vec![f32::NAN; m * n];
        assert_eq!(sgemm(&a, &b, &mut rerun, m, k, n), Ok(()));
        assert_eq!(fault::pending(), 0);
        for (what, out) in [("recovered", &got), ("re-run", &rerun)] {
            let differing = out
                .iter()
                .zip(&clean)
                .filter(|(g, c)| g.to_bits() != c.to_bits())
                .count();
            assert_eq!(differing, 0, "{what} output differs from the clean run");
        }
    }

    /// An injected failure must not keep its slot in the command queue. A
    /// queue blocks `commandBuffer` while 64 of its buffers are uncompleted,
    /// and an injected fault leaves its buffer uncommitted, which never
    /// completes; unless each attempt's buffer is released when the attempt
    /// ends, the 65th injected failure leaves every later caller of that
    /// queue, on any thread, blocked for good.
    #[cfg(target_os = "macos")]
    #[test]
    fn failed_command_buffers_do_not_keep_command_queue_slots_or_unavailable() {
        if !is_available() {
            return;
        }
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (m, k, n) = (8, 8, 8);
            let (a, b) = (vec![1.0f32; m * k], vec![1.0f32; k * n]);
            let mut c = vec![0.0f32; m * n];
            // Three times the queue's 64 uncompleted buffers.
            fault::arm(3 * 64 + COMMAND_BUFFER_ATTEMPTS);
            let mut failed_calls = 0;
            while fault::pending() > 0 {
                if sgemm(&a, &b, &mut c, m, k, n).is_err() {
                    failed_calls += 1;
                }
            }
            fault::arm(0);
            let clean = sgemm(&a, &b, &mut c, m, k, n);
            let _ = done_tx.send((failed_calls, clean, c));
        });
        let (failed_calls, clean, c) = done_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("sgemm blocked: failed command buffers used up the command queue");
        assert_eq!(
            failed_calls,
            1 + 3 * 64,
            "one exhausted call, then one attempt each"
        );
        assert_eq!(clean, Ok(()));
        assert!(c.iter().all(|v| *v == 8.0), "{c:?}");
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_stub_is_explicitly_unavailable() {
        assert!(!is_available());
        let mut output = [0.0_f32];
        assert_eq!(
            sgemm(&[1.0], &[1.0], &mut output, 1, 1, 1),
            Err(Error::Unavailable)
        );
    }
}
