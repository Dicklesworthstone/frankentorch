//! GPU-resident fused compute for transformer encoders (macOS / Apple Silicon).
//!
//! The v0.3.0 GEMM-only offload leaves the CPU and GPU ping-ponging: the CPU runs
//! layernorm/attention/gelu, then blocks on the GPU for each matmul, then reads the
//! result back. This module removes that by keeping activations **resident on the
//! GPU** ([`GpuTensor`]) and encoding a whole run of ops into **one command buffer**
//! ([`Batch`]) with a single CPU↔GPU sync at [`Batch::finish`].
//!
//! Ops are encoded as separate compute command encoders; Metal orders encoders
//! within a command buffer and hazard-tracks the (default-tracked) buffers, so each
//! op's writes are visible to the next — the activations flow GPU→GPU with no
//! readback until `finish`.
//!
//! Kernels match franken_whisper's CPU encoder: layernorm `eps = 1e-5`, tanh-GELU,
//! numerically-stable row softmax. All shapes are row-major 2-D `[rows, cols]`.

use super::Error;
use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputePipelineDescriptor, ComputePipelineState,
    Device, MTLResourceOptions, MTLSize,
};
use std::sync::OnceLock;

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

// C[M,N] = A[M,K] * B[K,N] (+ bias[N] if HAS_BIAS). Tiled, shared-mem + register block.
kernel void matmul_bias(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]], device float* C [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],   // M,K,N,has_bias
    uint2 gid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]])
{
    const uint M=dims.x,K=dims.y,N=dims.z; const uint hb=dims.w;
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
        for (uint j=0;j<TN;j++){uint gc=colBase+tCol*TN+j; if(gc<N){
            float v=acc[i][j]; if(hb) v+=bias[gc]; C[gr*N+gc]=v;}}}
}

// Row layernorm over `cols`, affine: y = (x-mean)/sqrt(var+eps)*gamma + beta. One threadgroup per row.
kernel void layernorm(
    device const float* X [[buffer(0)]], device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]], device float* Y [[buffer(3)]],
    constant uint2& dims [[buffer(4)]], constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    const uint cols=dims.y; device const float* x=X+row*cols; device float* y=Y+row*cols;
    threadgroup float red[256];
    float s=0.0f; for (uint c=lid;c<cols;c+=tgsz) s+=x[c];
    red[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o=tgsz/2;o>0;o>>=1){ if(lid<o) red[lid]+=red[lid+o]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float mean=red[0]/float(cols); threadgroup_barrier(mem_flags::mem_threadgroup);
    float v=0.0f; for (uint c=lid;c<cols;c+=tgsz){float d=x[c]-mean; v+=d*d;}
    red[lid]=v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o=tgsz/2;o>0;o>>=1){ if(lid<o) red[lid]+=red[lid+o]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv=rsqrt(red[0]/float(cols)+eps);
    for (uint c=lid;c<cols;c+=tgsz) y[c]=(x[c]-mean)*inv*gamma[c]+beta[c];
}

// tanh-GELU, elementwise (matches whisper.cpp GGML_GELU tanh form).
kernel void gelu(device const float* X [[buffer(0)]], device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if(i>=n) return; float x=X[i];
    const float c=0.7978845608028654f; // sqrt(2/pi)
    // Match ggml GGML_GELU_FP16 clamp (x>=10 -> x, x<=-10 -> 0). This is also
    // essential numerically: for x >~ 10.7 the tanh argument exceeds ~44 and MSL
    // tanh overflows to inf/inf = NaN. The clamp keeps tanh in its finite range.
    if (x >= 10.0f) { Y[i]=x; }
    else if (x <= -10.0f) { Y[i]=0.0f; }
    else { Y[i]=0.5f*x*(1.0f+tanh(c*(x+0.044715f*x*x*x))); }
}

// Elementwise add (residual): Y = A + B.
kernel void addv(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]])
{ if(i<n) Y[i]=A[i]+B[i]; }

// Row softmax over `cols`, numerically stable. One threadgroup per row.
kernel void softmax(device const float* X [[buffer(0)]], device float* Y [[buffer(1)]],
    constant uint2& dims [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]], uint tgsz [[threads_per_threadgroup]])
{
    const uint cols=dims.y; device const float* x=X+row*cols; device float* y=Y+row*cols;
    threadgroup float red[256];
    float m=-INFINITY; for (uint c=lid;c<cols;c+=tgsz) m=max(m,x[c]);
    red[lid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o=tgsz/2;o>0;o>>=1){ if(lid<o) red[lid]=max(red[lid],red[lid+o]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    bool ok = isfinite(mx);
    float s=0.0f; for (uint c=lid;c<cols;c+=tgsz){float e=ok?exp(x[c]-mx):0.0f; e=isfinite(e)?e:0.0f; y[c]=e; s+=e;}
    red[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o=tgsz/2;o>0;o>>=1){ if(lid<o) red[lid]+=red[lid+o]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv=(red[0]>0.0f)?1.0f/red[0]:0.0f; for (uint c=lid;c<cols;c+=tgsz) y[c]*=inv;
}

// Attention scores: S[h*seq+i, j] = scale * dot(Q[i, h*hd:h*hd+hd], K[j, h*hd:h*hd+hd]).
// dims = (seq, d_model, n_heads, head_dim). grid = (j, i, h).
kernel void attn_scores(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device float* S [[buffer(2)]], constant uint4& dims [[buffer(3)]],
    constant float& scale [[buffer(4)]], uint3 gid [[thread_position_in_grid]])
{
    uint seq=dims.x,d=dims.y,nh=dims.z,hd=dims.w; uint j=gid.x,i=gid.y,h=gid.z;
    if (i>=seq||j>=seq||h>=nh) return;
    float acc=0.0f; uint qo=i*d+h*hd, ko=j*d+h*hd;
    for (uint e=0;e<hd;e++) acc+=Q[qo+e]*K[ko+e];
    float s=acc*scale; S[(h*seq+i)*seq+j]=isfinite(s)?s:-1e30f;
}

// Attention context: O[i, h*hd+e] = sum_j S[h*seq+i, j] * V[j, h*hd+e].
// dims = (seq, d_model, n_heads, head_dim). grid = (e, i, h).
kernel void attn_context(
    device const float* S [[buffer(0)]], device const float* V [[buffer(1)]],
    device float* O [[buffer(2)]], constant uint4& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint seq=dims.x,d=dims.y,nh=dims.z,hd=dims.w; uint e=gid.x,i=gid.y,h=gid.z;
    if (e>=hd||i>=seq||h>=nh) return;
    float acc=0.0f; uint so=(h*seq+i)*seq;
    for (uint jj=0;jj<seq;jj++) acc+=S[so+jj]*V[jj*d+h*hd+e];
    O[i*d+h*hd+e]=acc;
}

// FlashAttention (whisper head_dim = 64). One threadgroup per (query-block, head);
// FBQ threads each own one query. Streams K/V in FBK-blocks through threadgroup
// memory with an online (running max/sum) softmax and register accumulation — the
// [n_heads*seq, seq] scores matrix is never materialized. dims=(seq,d,n_heads,64).
#define FBQ 32
#define FBK 32
#define FHD 64
kernel void attn_flash(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device float* O [[buffer(3)]],
    constant uint4& dims [[buffer(4)]], constant float& scale [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]])
{
    const uint seq=dims.x, d=dims.y; const uint h=tg.y; const uint qi=tg.x*FBQ+lid;
    float q[FHD]; for (uint e=0;e<FHD;e++) q[e]=(qi<seq)?Q[qi*d+h*FHD+e]:0.0f;
    float m=-INFINITY, l=0.0f; float acc[FHD]; for (uint e=0;e<FHD;e++) acc[e]=0.0f;
    threadgroup float Ks[FBK][FHD]; threadgroup float Vs[FBK][FHD];
    for (uint k0=0;k0<seq;k0+=FBK) {
        for (uint idx=lid; idx<FBK*FHD; idx+=FBQ) {
            uint kk=idx/FHD, e=idx%FHD; uint kj=k0+kk;
            Ks[kk][e]=(kj<seq)?K[kj*d+h*FHD+e]:0.0f;
            Vs[kk][e]=(kj<seq)?V[kj*d+h*FHD+e]:0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint kmax=min((uint)FBK, seq-k0);
        for (uint kk=0;kk<kmax;kk++) {
            float s=0.0f; for (uint e=0;e<FHD;e++) s+=q[e]*Ks[kk][e]; s*=scale;
            float mn=max(m,s); float p=exp(s-mn); float corr=exp(m-mn);
            l=l*corr+p; for (uint e=0;e<FHD;e++) acc[e]=acc[e]*corr+p*Vs[kk][e];
            m=mn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (qi<seq){ float inv=(l>0.0f)?1.0f/l:0.0f; for (uint e=0;e<FHD;e++) O[qi*d+h*FHD+e]=acc[e]*inv; }
}
"#;

struct Pipes {
    device: Device,
    queue: metal::CommandQueue,
    matmul_bias: ComputePipelineState,
    layernorm: ComputePipelineState,
    gelu: ComputePipelineState,
    addv: ComputePipelineState,
    softmax: ComputePipelineState,
    attn_scores: ComputePipelineState,
    attn_context: ComputePipelineState,
    attn_flash: ComputePipelineState,
}
unsafe impl Send for Pipes {}
unsafe impl Sync for Pipes {}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes() -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let device = Device::system_default()?;
            let lib = device
                .new_library_with_source(SRC, &CompileOptions::new())
                .ok()?;
            let p = |n: &str| -> Option<ComputePipelineState> {
                let f = lib.get_function(n, None).ok()?;
                let d = ComputePipelineDescriptor::new();
                d.set_compute_function(Some(&f));
                device
                    .new_compute_pipeline_state_with_function(d.compute_function()?)
                    .ok()
            };
            Some(Pipes {
                matmul_bias: p("matmul_bias")?,
                layernorm: p("layernorm")?,
                gelu: p("gelu")?,
                addv: p("addv")?,
                softmax: p("softmax")?,
                attn_scores: p("attn_scores")?,
                attn_context: p("attn_context")?,
                attn_flash: p("attn_flash")?,
                queue: device.new_command_queue(),
                device,
            })
        })
        .as_ref()
}

/// A `[rows, cols]` row-major f32 tensor resident in GPU (unified) memory.
pub struct GpuTensor {
    buf: Buffer,
    pub rows: usize,
    pub cols: usize,
}

impl GpuTensor {
    fn new_uninit(dev: &Device, rows: usize, cols: usize) -> GpuTensor {
        GpuTensor {
            buf: dev.new_buffer((rows * cols * 4).max(4) as u64, SHARED),
            rows,
            cols,
        }
    }

    /// Upload a row-major `[rows, cols]` slice to a resident GPU buffer.
    pub fn upload(data: &[f32], rows: usize, cols: usize) -> Result<GpuTensor, Error> {
        let p = pipes().ok_or(Error::Unavailable)?;
        if data.len() != rows * cols {
            return Err(Error::Kernel(format!(
                "upload shape: {} != {rows}x{cols}",
                data.len()
            )));
        }
        Ok(GpuTensor {
            buf: p.device.new_buffer_with_data(
                data.as_ptr() as *const _,
                (data.len() * 4).max(4) as u64,
                SHARED,
            ),
            rows,
            cols,
        })
    }

    /// Read this tensor back into a `Vec<f32>` (row-major).
    pub fn download(&self) -> Vec<f32> {
        let n = self.rows * self.cols;
        let out = unsafe { std::slice::from_raw_parts(self.buf.contents() as *const f32, n) };
        out.to_vec()
    }
}

/// True iff the fused-op pipelines are available (Apple Silicon macOS).
pub fn is_available() -> bool {
    pipes().is_some()
}

/// A sequence of GPU ops encoded into one command buffer; committed once at
/// [`finish`](Batch::finish). Each op appends a compute encoder whose output is a
/// fresh resident [`GpuTensor`] that later ops read — no CPU readback between ops.
pub struct Batch<'a> {
    p: &'a Pipes,
    cmd: &'a CommandBufferRef,
}

impl<'a> Batch<'a> {
    /// Begin a batch. Ops encode until [`finish`](Batch::finish).
    pub fn new() -> Result<Batch<'static>, Error> {
        let p = pipes().ok_or(Error::Unavailable)?;
        let cmd = p.queue.new_command_buffer();
        Ok(Batch { p, cmd })
    }

    fn u32buf(&self, v: &[u32]) -> Buffer {
        self.p
            .device
            .new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, SHARED)
    }

    fn dispatch1d(&self, pso: &ComputePipelineState, bufs: &[(&Buffer, u64)], n: usize) {
        let enc = self.cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, (b, off)) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), *off);
        }
        let w = pso.thread_execution_width().max(1);
        let tg = ((n as u64).div_ceil(w) * w).max(w);
        enc.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(tg.min(w * 8), 1, 1),
        );
        enc.end_encoding();
    }

    /// `out[rows_a, cols_b] = a[rows_a, cols_a] · w[cols_a, cols_b] (+ bias)`.
    pub fn matmul_bias(&self, a: &GpuTensor, w: &GpuTensor, bias: Option<&GpuTensor>) -> GpuTensor {
        let (m, k, n) = (a.rows, a.cols, w.cols);
        let out = GpuTensor::new_uninit(&self.p.device, m, n);
        let dims = self.u32buf(&[m as u32, k as u32, n as u32, bias.is_some() as u32]);
        let zero = self.p.device.new_buffer(4, SHARED);
        let bb = bias.map_or(&zero, |b| &b.buf);
        let enc = self.cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.p.matmul_bias);
        enc.set_buffer(0, Some(&a.buf), 0);
        enc.set_buffer(1, Some(&w.buf), 0);
        enc.set_buffer(2, Some(bb), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_buffer(4, Some(&dims), 0);
        let tg = MTLSize::new(n.div_ceil(64) as u64, m.div_ceil(64) as u64, 1);
        enc.dispatch_thread_groups(tg, MTLSize::new(256, 1, 1));
        enc.end_encoding();
        out
    }

    /// Row layernorm with affine `gamma`/`beta` (length `cols`), `eps`.
    pub fn layernorm(
        &self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        eps: f32,
    ) -> GpuTensor {
        let out = GpuTensor::new_uninit(&self.p.device, x.rows, x.cols);
        let dims = self.u32buf(&[x.rows as u32, x.cols as u32]);
        let epsb = self
            .p
            .device
            .new_buffer_with_data((&eps as *const f32) as *const _, 4, SHARED);
        let enc = self.cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.p.layernorm);
        enc.set_buffer(0, Some(&x.buf), 0);
        enc.set_buffer(1, Some(&gamma.buf), 0);
        enc.set_buffer(2, Some(&beta.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_buffer(4, Some(&dims), 0);
        enc.set_buffer(5, Some(&epsb), 0);
        enc.dispatch_thread_groups(MTLSize::new(x.rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        out
    }

    /// tanh-GELU, elementwise.
    pub fn gelu(&self, x: &GpuTensor) -> GpuTensor {
        let out = GpuTensor::new_uninit(&self.p.device, x.rows, x.cols);
        let n = (x.rows * x.cols) as u32;
        let nb = self.u32buf(&[n]);
        self.dispatch1d(
            &self.p.gelu,
            &[(&x.buf, 0), (&out.buf, 0), (&nb, 0)],
            n as usize,
        );
        out
    }

    /// Elementwise residual add `a + b`.
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        let out = GpuTensor::new_uninit(&self.p.device, a.rows, a.cols);
        let n = (a.rows * a.cols) as u32;
        let nb = self.u32buf(&[n]);
        self.dispatch1d(
            &self.p.addv,
            &[(&a.buf, 0), (&b.buf, 0), (&out.buf, 0), (&nb, 0)],
            n as usize,
        );
        out
    }

    /// Numerically-stable row softmax over `cols`.
    pub fn softmax(&self, x: &GpuTensor) -> GpuTensor {
        let out = GpuTensor::new_uninit(&self.p.device, x.rows, x.cols);
        let dims = self.u32buf(&[x.rows as u32, x.cols as u32]);
        let enc = self.cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.p.softmax);
        enc.set_buffer(0, Some(&x.buf), 0);
        enc.set_buffer(1, Some(&out.buf), 0);
        enc.set_buffer(2, Some(&dims), 0);
        enc.dispatch_thread_groups(MTLSize::new(x.rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        out
    }

    /// Multi-head self-attention: per-token `q`/`k`/`v` `[seq, d_model]` →
    /// `concat_h( softmax(q_h·k_hᵀ / √head_dim) · v_h )` → `[seq, d_model]`.
    pub fn mha(&self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, n_heads: usize) -> GpuTensor {
        let (seq, d) = (q.rows, q.cols);
        let hd = d / n_heads;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let dims = self.u32buf(&[seq as u32, d as u32, n_heads as u32, hd as u32]);
        let scaleb =
            self.p
                .device
                .new_buffer_with_data((&scale as *const f32) as *const _, 4, SHARED);
        // FlashAttention fast path (whisper head_dim = 64): one fused dispatch that
        // streams K/V with an online softmax — the scores matrix is never materialized.
        if hd == 64 {
            let out = GpuTensor::new_uninit(&self.p.device, seq, d);
            let enc = self.cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.p.attn_flash);
            enc.set_buffer(0, Some(&q.buf), 0);
            enc.set_buffer(1, Some(&k.buf), 0);
            enc.set_buffer(2, Some(&v.buf), 0);
            enc.set_buffer(3, Some(&out.buf), 0);
            enc.set_buffer(4, Some(&dims), 0);
            enc.set_buffer(5, Some(&scaleb), 0);
            let tg = MTLSize::new(seq.div_ceil(32) as u64, n_heads as u64, 1);
            enc.dispatch_thread_groups(tg, MTLSize::new(32, 1, 1));
            enc.end_encoding();
            return out;
        }
        let scores = GpuTensor::new_uninit(&self.p.device, n_heads * seq, seq);
        {
            let enc = self.cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.p.attn_scores);
            enc.set_buffer(0, Some(&q.buf), 0);
            enc.set_buffer(1, Some(&k.buf), 0);
            enc.set_buffer(2, Some(&scores.buf), 0);
            enc.set_buffer(3, Some(&dims), 0);
            enc.set_buffer(4, Some(&scaleb), 0);
            enc.dispatch_threads(
                MTLSize::new(seq as u64, seq as u64, n_heads as u64),
                MTLSize::new(8, 8, 1),
            );
            enc.end_encoding();
        }
        let sm = self.softmax(&scores); // per (h,i) row over j
        let out = GpuTensor::new_uninit(&self.p.device, seq, d);
        {
            let enc = self.cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.p.attn_context);
            enc.set_buffer(0, Some(&sm.buf), 0);
            enc.set_buffer(1, Some(&v.buf), 0);
            enc.set_buffer(2, Some(&out.buf), 0);
            enc.set_buffer(3, Some(&dims), 0);
            enc.dispatch_threads(
                MTLSize::new(hd as u64, seq as u64, n_heads as u64),
                MTLSize::new(8, 8, 1),
            );
            enc.end_encoding();
        }
        out
    }

    /// Commit the command buffer and block until the GPU finishes — the single
    /// CPU↔GPU sync for every op encoded since [`new`](Batch::new).
    pub fn finish(self) {
        self.cmd.commit();
        self.cmd.wait_until_completed();
    }
}

/// Layer-norm epsilon (matches franken_whisper / whisper.cpp `hparams.eps`).
pub const LN_EPS: f32 = 1e-5;

/// One encoder layer's weights, resident on the GPU. Attention weights `wq/wk/wv/wo`
/// and MLP `w1/w2` are `[in, out]` row-major (pre-transposed, ready for `x @ w`).
struct LayerWeightsGpu {
    ln1_g: GpuTensor,
    ln1_b: GpuTensor,
    wq: GpuTensor,
    bq: GpuTensor,
    wk: GpuTensor,
    wv: GpuTensor,
    bv: GpuTensor,
    wo: GpuTensor,
    bo: GpuTensor,
    ln2_g: GpuTensor,
    ln2_b: GpuTensor,
    w1: GpuTensor,
    b1: GpuTensor,
    w2: GpuTensor,
    b2: GpuTensor,
}

/// Borrowed weights for one encoder layer (row-major; `wk` has no key bias, per
/// whisper). Weights are `[in, out]` (pre-transposed); biases are length `out`.
pub struct LayerWeightsRef<'a> {
    pub ln1_g: &'a [f32],
    pub ln1_b: &'a [f32],
    pub wq: &'a [f32],
    pub bq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub bv: &'a [f32],
    pub wo: &'a [f32],
    pub bo: &'a [f32],
    pub ln2_g: &'a [f32],
    pub ln2_b: &'a [f32],
    pub w1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [f32],
    pub b2: &'a [f32],
}

/// A GPU-resident transformer encoder: layer weights are uploaded **once**, and
/// [`forward`](EncoderGpu::forward) runs every layer on the GPU with the
/// activations resident between layers — one command buffer (one sync) per layer,
/// replacing the CPU↔GPU ping-pong of the GEMM-only path.
pub struct EncoderGpu {
    layers: Vec<LayerWeightsGpu>,
    d_model: usize,
    n_heads: usize,
}

// The resident weight buffers are immutable after upload (GPU read-only), the
// command queue is thread-safe, and `forward` allocates only per-call local
// activation buffers — so an EncoderGpu can be shared/cached across threads.
unsafe impl Send for EncoderGpu {}
unsafe impl Sync for EncoderGpu {}

impl EncoderGpu {
    /// Upload all layer weights to the GPU. `d_ff` is the MLP hidden width
    /// (`4*d_model` for whisper). Returns [`Error::Unavailable`] with no GPU.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        layers: &[LayerWeightsRef],
    ) -> Result<EncoderGpu, Error> {
        if !is_available() {
            return Err(Error::Unavailable);
        }
        let mut gl = Vec::with_capacity(layers.len());
        for l in layers {
            gl.push(LayerWeightsGpu {
                ln1_g: GpuTensor::upload(l.ln1_g, 1, d_model)?,
                ln1_b: GpuTensor::upload(l.ln1_b, 1, d_model)?,
                wq: GpuTensor::upload(l.wq, d_model, d_model)?,
                bq: GpuTensor::upload(l.bq, 1, d_model)?,
                wk: GpuTensor::upload(l.wk, d_model, d_model)?,
                wv: GpuTensor::upload(l.wv, d_model, d_model)?,
                bv: GpuTensor::upload(l.bv, 1, d_model)?,
                wo: GpuTensor::upload(l.wo, d_model, d_model)?,
                bo: GpuTensor::upload(l.bo, 1, d_model)?,
                ln2_g: GpuTensor::upload(l.ln2_g, 1, d_model)?,
                ln2_b: GpuTensor::upload(l.ln2_b, 1, d_model)?,
                w1: GpuTensor::upload(l.w1, d_model, d_ff)?,
                b1: GpuTensor::upload(l.b1, 1, d_ff)?,
                w2: GpuTensor::upload(l.w2, d_ff, d_model)?,
                b2: GpuTensor::upload(l.b2, 1, d_model)?,
            });
        }
        Ok(EncoderGpu {
            layers: gl,
            d_model,
            n_heads,
        })
    }

    /// Run every layer on the GPU. `input` is `[seq, d_model]` row-major; returns
    /// the encoder-stack output `[seq, d_model]` (pre-`ln_post`).
    pub fn forward(&self, input: &[f32], seq: usize) -> Result<Vec<f32>, Error> {
        if input.len() != seq * self.d_model {
            return Err(Error::Kernel(format!(
                "forward input {} != {seq}x{}",
                input.len(),
                self.d_model
            )));
        }
        let mut x = GpuTensor::upload(input, seq, self.d_model)?;
        for l in &self.layers {
            // One command buffer per layer: all ops chain GPU-resident, one sync.
            let b = Batch::new()?;
            let n1 = b.layernorm(&x, &l.ln1_g, &l.ln1_b, LN_EPS);
            let q = b.matmul_bias(&n1, &l.wq, Some(&l.bq));
            let k = b.matmul_bias(&n1, &l.wk, None);
            let v = b.matmul_bias(&n1, &l.wv, Some(&l.bv));
            let attn = b.mha(&q, &k, &v, self.n_heads);
            let ao = b.matmul_bias(&attn, &l.wo, Some(&l.bo));
            let x1 = b.add(&x, &ao);
            let n2 = b.layernorm(&x1, &l.ln2_g, &l.ln2_b, LN_EPS);
            let fc = b.matmul_bias(&n2, &l.w1, Some(&l.b1));
            let g = b.gelu(&fc);
            let proj = b.matmul_bias(&g, &l.w2, Some(&l.b2));
            let x2 = b.add(&x1, &proj);
            b.finish();
            x = x2; // resident output carried to the next layer
        }
        Ok(x.download())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what} len");
        for (i, (g, w)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "{what} idx {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn fused_ops_match_cpu_or_unavailable() {
        if !is_available() {
            return;
        }
        let (m, k, n) = (40usize, 33usize, 48usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.3 - 1.0).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let ga = GpuTensor::upload(&a, m, k).unwrap();
        let gw = GpuTensor::upload(&w, k, n).unwrap();
        let gb = GpuTensor::upload(&bias, 1, n).unwrap();
        let b = Batch::new().unwrap();
        let mm = b.matmul_bias(&ga, &gw, Some(&gb));
        let ln = b.layernorm(
            &mm,
            &GpuTensor::upload(&vec![1.0; n], 1, n).unwrap(),
            &GpuTensor::upload(&vec![0.0; n], 1, n).unwrap(),
            1e-5,
        );
        let ge = b.gelu(&mm);
        let sm = b.softmax(&mm);
        b.finish(); // commit + wait, THEN read results back
        let (mm_o, ln_o, ge_o, sm_o) = (mm.download(), ln.download(), ge.download(), sm.download());

        // CPU refs.
        let mut cmm = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = bias[j];
                for e in 0..k {
                    acc += a[i * k + e] * w[e * n + j];
                }
                cmm[i * n + j] = acc;
            }
        }
        approx(&mm_o, &cmm, 1e-3, "matmul_bias");
        // layernorm(gamma=1,beta=0)
        let mut cln = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &cmm[i * n..(i + 1) * n];
            let mean: f32 = row.iter().sum::<f32>() / n as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..n {
                cln[i * n + j] = (row[j] - mean) * inv;
            }
        }
        approx(&ln_o, &cln, 1e-2, "layernorm");
        let c = 0.7978845608028654f32;
        let cge: Vec<f32> = cmm
            .iter()
            .map(|&x| 0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh()))
            .collect();
        approx(&ge_o, &cge, 1e-3, "gelu");
        let mut csm = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &cmm[i * n..(i + 1) * n];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let s: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            for j in 0..n {
                csm[i * n + j] = (row[j] - mx).exp() / s;
            }
        }
        approx(&sm_o, &csm, 1e-3, "softmax");
    }

    fn cpu_mha(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize, nh: usize) -> Vec<f32> {
        let hd = d / nh;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; seq * d];
        for h in 0..nh {
            for i in 0..seq {
                let mut sc = vec![0.0f32; seq];
                for j in 0..seq {
                    let mut acc = 0.0f32;
                    for e in 0..hd {
                        acc += q[i * d + h * hd + e] * k[j * d + h * hd + e];
                    }
                    sc[j] = acc * scale;
                }
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let s: f32 = sc.iter().map(|x| (x - mx).exp()).sum();
                for j in 0..seq {
                    sc[j] = (sc[j] - mx).exp() / s;
                }
                for e in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..seq {
                        acc += sc[j] * v[j * d + h * hd + e];
                    }
                    out[i * d + h * hd + e] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn mha_matches_cpu_or_unavailable() {
        if !is_available() {
            return;
        }
        let (seq, d, nh) = (12usize, 16usize, 4usize);
        let q: Vec<f32> = (0..seq * d).map(|i| ((i % 7) as f32) * 0.2 - 0.6).collect();
        let k: Vec<f32> = (0..seq * d)
            .map(|i| ((i % 5) as f32) * 0.15 - 0.3)
            .collect();
        let v: Vec<f32> = (0..seq * d).map(|i| ((i % 9) as f32) * 0.1 - 0.4).collect();
        let gq = GpuTensor::upload(&q, seq, d).unwrap();
        let gk = GpuTensor::upload(&k, seq, d).unwrap();
        let gv = GpuTensor::upload(&v, seq, d).unwrap();
        let b = Batch::new().unwrap();
        let o = b.mha(&gq, &gk, &gv, nh);
        b.finish();
        approx(&o.download(), &cpu_mha(&q, &k, &v, seq, d, nh), 1e-3, "mha");
    }

    #[test]
    fn mha_flash_matches_cpu_or_unavailable() {
        if !is_available() {
            return;
        }
        let (seq, d, nh) = (40usize, 128usize, 2usize); // head_dim = 64 -> flash path
        let q: Vec<f32> = (0..seq * d).map(|i| ((i % 7) as f32) * 0.2 - 0.6).collect();
        let k: Vec<f32> = (0..seq * d)
            .map(|i| ((i % 5) as f32) * 0.15 - 0.3)
            .collect();
        let v: Vec<f32> = (0..seq * d).map(|i| ((i % 9) as f32) * 0.1 - 0.4).collect();
        let gq = GpuTensor::upload(&q, seq, d).unwrap();
        let gk = GpuTensor::upload(&k, seq, d).unwrap();
        let gv = GpuTensor::upload(&v, seq, d).unwrap();
        let b = Batch::new().unwrap();
        let o = b.mha(&gq, &gk, &gv, nh);
        b.finish();
        approx(
            &o.download(),
            &cpu_mha(&q, &k, &v, seq, d, nh),
            1e-3,
            "mha_flash",
        );
    }

    fn cpu_ln(x: &[f32], g: &[f32], be: &[f32], seq: usize, d: usize) -> Vec<f32> {
        let mut o = vec![0.0f32; seq * d];
        for i in 0..seq {
            let row = &x[i * d..(i + 1) * d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            for j in 0..d {
                o[i * d + j] = (row[j] - mean) * inv * g[j] + be[j];
            }
        }
        o
    }
    fn cpu_mmb(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut o = vec![0.0f32; m * n];
        for i in 0..m {
            for jn in 0..n {
                let mut acc = bias.map_or(0.0, |b| b[jn]);
                for e in 0..k {
                    acc += x[i * k + e] * w[e * n + jn];
                }
                o[i * n + jn] = acc;
            }
        }
        o
    }
    fn cpu_gelu(x: &[f32]) -> Vec<f32> {
        let c = 0.7978845608028654f32;
        x.iter()
            .map(|&v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()))
            .collect()
    }
    fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    #[test]
    fn encoder_layer_matches_cpu_or_unavailable() {
        if !is_available() {
            return;
        }
        let (seq, d, nh, dff) = (10usize, 16usize, 4usize, 32usize);
        let mk = |n: usize, s: f32, o: f32| -> Vec<f32> {
            (0..n).map(|i| ((i % 13) as f32) * s - o).collect()
        };
        let (ln1_g, ln1_b) = (vec![1.0f32; d], vec![0.0f32; d]);
        let (ln2_g, ln2_b) = (vec![1.0f32; d], vec![0.0f32; d]);
        let wq = mk(d * d, 0.03, 0.2);
        let bq = mk(d, 0.01, 0.05);
        let wk = mk(d * d, 0.02, 0.15);
        let wv = mk(d * d, 0.025, 0.18);
        let bv = mk(d, 0.01, 0.04);
        let wo = mk(d * d, 0.02, 0.16);
        let bo = mk(d, 0.01, 0.03);
        let w1 = mk(d * dff, 0.02, 0.15);
        let b1 = mk(dff, 0.005, 0.02);
        let w2 = mk(dff * d, 0.02, 0.15);
        let b2 = mk(d, 0.01, 0.03);
        let lref = LayerWeightsRef {
            ln1_g: &ln1_g,
            ln1_b: &ln1_b,
            wq: &wq,
            bq: &bq,
            wk: &wk,
            wv: &wv,
            bv: &bv,
            wo: &wo,
            bo: &bo,
            ln2_g: &ln2_g,
            ln2_b: &ln2_b,
            w1: &w1,
            b1: &b1,
            w2: &w2,
            b2: &b2,
        };
        let x = mk(seq * d, 0.04, 0.6);

        let enc = EncoderGpu::new(d, nh, dff, std::slice::from_ref(&lref)).unwrap();
        let got = enc.forward(&x, seq).unwrap();

        // CPU reference layer (same op order as EncoderGpu::forward).
        let n1 = cpu_ln(&x, &ln1_g, &ln1_b, seq, d);
        let q = cpu_mmb(&n1, &wq, Some(&bq), seq, d, d);
        let k = cpu_mmb(&n1, &wk, None, seq, d, d);
        let v = cpu_mmb(&n1, &wv, Some(&bv), seq, d, d);
        let attn = cpu_mha(&q, &k, &v, seq, d, nh);
        let ao = cpu_mmb(&attn, &wo, Some(&bo), seq, d, d);
        let x1 = cpu_add(&x, &ao);
        let n2 = cpu_ln(&x1, &ln2_g, &ln2_b, seq, d);
        let fc = cpu_mmb(&n2, &w1, Some(&b1), seq, d, dff);
        let g = cpu_gelu(&fc);
        let proj = cpu_mmb(&g, &w2, Some(&b2), seq, dff, d);
        let want = cpu_add(&x1, &proj);

        approx(&got, &want, 2e-2, "encoder_layer");
    }
}
