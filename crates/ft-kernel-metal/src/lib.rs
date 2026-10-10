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
pub const COMMAND_BUFFER_ATTEMPTS: u32 = 4;

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
        Ok(())
    } else {
        COMMAND_BUFFER_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Err(Error::CommandBuffer(state))
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
#[cfg(target_os = "macos")]
pub(crate) mod fault {
    thread_local! {
        static PENDING: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// Arm `n` faults on this thread.
    pub(crate) fn arm(n: u32) {
        PENDING.with(|p| p.set(n));
    }

    /// Consume one armed fault, if any.
    pub(crate) fn take() -> bool {
        PENDING.with(|p| {
            let n = p.get();
            if n > 0 {
                p.set(n - 1);
            }
            n > 0
        })
    }
}

/// Test seam (not for production use): the next `n` command buffers that
/// [`sgemm`] or the fused encoder submit from the calling thread end without
/// running and report `NotEnqueued`, leaving their outputs unwritten, as a GPU
/// restart leaves aborted work. `0` disarms. A no-op off macOS, where nothing
/// is submitted. Always compiled rather than feature-gated so a dependent's
/// tests need no feature that an older sibling checkout would lack.
#[doc(hidden)]
pub fn inject_command_buffer_faults(n: u32) {
    #[cfg(target_os = "macos")]
    fault::arm(n);
    #[cfg(not(target_os = "macos"))]
    let _ = n;
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
        // buffers (see `COMMAND_BUFFER_ATTEMPTS`).
        let mut failed = 0;
        loop {
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
            match super::submit_and_wait(cmd) {
                Ok(()) => break,
                Err(error) => {
                    failed += 1;
                    if failed >= super::COMMAND_BUFFER_ATTEMPTS {
                        return Err(error);
                    }
                    super::retry_backoff(failed);
                }
            }
        }

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
/// the GPU did not complete the work in [`COMMAND_BUFFER_ATTEMPTS`] tries.
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
