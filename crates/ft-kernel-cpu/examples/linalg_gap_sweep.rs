//! Honest dense-linalg gap sweep vs torch/LAPACK. Times the symmetric eig
//! family + qr + svd on a deterministic-LCG random matrix (identical matrix
//! reproducible in torch) so we can size where we LOSE. frankentorch-l9xod /
//! t0b4l (blocked dsytrd) / x53r3.
//!
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example linalg_gap_sweep

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    eigh_contiguous_f64, eigvalsh_contiguous_f64, qr_contiguous_f64, svd_contiguous_f64,
    svdvals_contiguous_f64,
};
use std::time::Instant;

fn lcg(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    let mut x: u64 = 0x9E3779B97F4A7C15;
    for slot in a.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (x >> 11) as f64 / 9007199254740992.0;
        *slot = u * 2.0 - 1.0;
    }
    a
}

fn symmetrize(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
}

fn bench<F: FnMut()>(mut f: F, it: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..it {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / it as f64
}

fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[256usize, 512, 1024] {
        let mut sym = lcg(n);
        symmetrize(&mut sym, n);
        let gmat = lcg(n);
        let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let it = if n <= 512 { 4 } else { 2 };
        let evh = bench(
            || {
                let _ = eigvalsh_contiguous_f64(&sym, &m).unwrap();
            },
            it,
        );
        let egh = bench(
            || {
                let _ = eigh_contiguous_f64(&sym, &m).unwrap();
            },
            it,
        );
        let qr = bench(
            || {
                let _ = qr_contiguous_f64(&gmat, &m, true).unwrap();
            },
            it,
        );
        let sv = bench(
            || {
                let _ = svdvals_contiguous_f64(&gmat, &m).unwrap();
            },
            it,
        );
        let svf = bench(
            || {
                let _ = svd_contiguous_f64(&gmat, &m, false).unwrap();
            },
            it,
        );
        println!(
            "n={n:5} eigvalsh={evh:8.2}ms eigh={egh:8.2}ms qr={qr:8.2}ms svdvals={sv:8.2}ms svd={svf:8.2}ms"
        );
    }
}
