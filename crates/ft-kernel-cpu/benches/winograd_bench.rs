//! Winograd F(2x2,3x3) f32 conv vs im2col+sgemm baseline. Same binary, same
//! worker — the GEMM dominates so the (serial) im2col gather is negligible and
//! this isolates the Winograd algorithmic lever (16 transform-domain products/
//! tile vs 36 direct). Representative 3x3 stride-1 conv shapes.
//!
//!   cargo bench -p ft-kernel-cpu --bench winograd_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{matmul_tensor_contiguous_f32, winograd_conv2d_3x3_s1_f32};

/// Serial im2col panel `[flat, patch_width]` for a 3x3 stride-1 conv, f32.
fn im2col_3x3_s1_f32(
    padded: &[f32],
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let patch_width = in_ch * 9;
    let patch_count = oh * ow;
    let flat = batch * patch_count;
    let mut panel = vec![0.0f32; flat * patch_width];
    for b in 0..batch {
        for oy in 0..oh {
            for ox in 0..ow {
                let row = (b * patch_count) + oy * ow + ox;
                let ro = row * patch_width;
                for ic in 0..in_ch {
                    let ibase = (b * in_ch + ic) * ph * pw;
                    let po = ro + ic * 9;
                    for ky in 0..3 {
                        let iy = oy + ky;
                        for kx in 0..3 {
                            panel[po + ky * 3 + kx] = padded[ibase + iy * pw + (ox + kx)];
                        }
                    }
                }
            }
        }
    }
    panel
}

/// Baseline fused 3x3 stride-1 f32 conv: im2col + sgemm_bt (mirrors the f64 fused
/// path `conv2d_forward_f64`). `weight` is `[out_ch, in_ch, 3, 3]` row-major.
fn conv2d_3x3_s1_im2col_f32(
    padded: &[f32],
    weight: &[f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let oh = ph - 2;
    let ow = pw - 2;
    let patch_width = in_ch * 9;
    let flat = batch * oh * ow;
    let panel = im2col_3x3_s1_f32(padded, batch, in_ch, ph, pw, oh, ow);
    // weight [out_ch, patch_width] -> weight_t [patch_width, out_ch] so the
    // standard matmul computes panel[flat,pw] @ weight_t[pw,out_ch] = [flat,out_ch],
    // matching the production f32 conv path (im2col + tensor_matmul over sgemm).
    let mut weight_t = vec![0.0f32; patch_width * out_ch];
    for oc in 0..out_ch {
        for k in 0..patch_width {
            weight_t[k * out_ch + oc] = weight[oc * patch_width + k];
        }
    }
    let pm = TensorMeta::from_shape(vec![flat, patch_width], DType::F32, Device::Cpu);
    let wm = TensorMeta::from_shape(vec![patch_width, out_ch], DType::F32, Device::Cpu);
    matmul_tensor_contiguous_f32(&panel, &weight_t, &pm, &wm).unwrap()
}

fn bench_winograd(c: &mut Criterion) {
    // (batch, in_ch, out_ch, H=W) — H/W is the unpadded spatial size; pad=1 -> ph=pw=H+2.
    let shapes = [
        (8usize, 64usize, 64usize, 32usize),
        (4, 128, 128, 16),
        (1, 16, 16, 128),
        (16, 32, 32, 28),
    ];
    for &(batch, in_ch, out_ch, hw) in &shapes {
        let ph = hw + 2;
        let pw = hw + 2;
        let padded: Vec<f32> = (0..batch * in_ch * ph * pw)
            .map(|i| ((i % 97) as f32) * 0.013 - 0.5)
            .collect();
        let weight: Vec<f32> = (0..out_ch * in_ch * 9)
            .map(|i| ((i % 31) as f32) * 0.017 - 0.25)
            .collect();
        let tag = format!("b{batch}_ic{in_ch}_oc{out_ch}_hw{hw}");
        c.bench_function(&format!("im2col_{tag}"), |bch| {
            bch.iter(|| {
                black_box(conv2d_3x3_s1_im2col_f32(
                    black_box(&padded),
                    black_box(&weight),
                    batch,
                    in_ch,
                    out_ch,
                    ph,
                    pw,
                ))
            })
        });
        c.bench_function(&format!("winograd_{tag}"), |bch| {
            bch.iter(|| {
                black_box(winograd_conv2d_3x3_s1_f32(
                    black_box(&padded),
                    black_box(&weight),
                    batch,
                    in_ch,
                    out_ch,
                    ph,
                    pw,
                ))
            })
        });
    }
}

criterion_group!(benches, bench_winograd);
criterion_main!(benches);
