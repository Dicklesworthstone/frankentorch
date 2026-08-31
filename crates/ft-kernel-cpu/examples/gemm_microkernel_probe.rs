//! Feasibility probe for the packed-panel GEMM (bead frankentorch-z6sjf): can a
//! hand-written safe-Rust register-blocked microkernel match matrixmultiply's per-core
//! throughput? If not (per-core << matrixmultiply's ~52 GF/s/core f64), then even
//! perfect multi-core scaling would net a loss and the rewrite is not worth it.
//! Compares a naive triple loop, an mr x nr register-blocked microkernel (straight-k
//! accumulation, so tolerance-parity-safe), and the production matmul — all SINGLE
//! THREAD (per-core). 1024^3 by default.
//!
//! The `--reverse-apply [reps]` mode is the `frankentorch-5rnsq` raw-entry isolation
//! probe. It measures the three compact-WY reverse-application dgemm shapes through
//! the internal f64 dgemm entry under one and eight Rayon threads, then compares the
//! latter with the kernel's checked/allocating contiguous-matmul wrapper.
//!   cargo run -q --release -p ft-kernel-cpu --example gemm_microkernel_probe -- --reverse-apply 64
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{matmul_tensor_contiguous_f64, probe_dgemm, probe_dgemm_tb};
use std::time::Instant;

#[inline(never)]
fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[inline(never)]
fn blocked(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    const MR: usize = 4;
    const NR: usize = 8;
    let mut c = vec![0.0; m * n];
    let mut i0 = 0;
    while i0 < m {
        let ib = (i0 + MR).min(m) - i0;
        let mut j0 = 0;
        while j0 < n {
            let jb = (j0 + NR).min(n) - j0;
            let mut acc = [[0.0f64; NR]; MR];
            for p in 0..k {
                let brow = &b[p * n + j0..p * n + j0 + jb];
                for ii in 0..ib {
                    let av = a[(i0 + ii) * k + p];
                    let arow = &mut acc[ii];
                    for jj in 0..jb {
                        arow[jj] += av * brow[jj];
                    }
                }
            }
            for ii in 0..ib {
                for jj in 0..jb {
                    c[(i0 + ii) * n + j0 + jj] = acc[ii][jj];
                }
            }
            j0 += NR;
        }
        i0 += MR;
    }
    c
}

fn gflops(m: usize, k: usize, n: usize, ms: f64) -> f64 {
    2.0 * (m as f64) * (k as f64) * (n as f64) / (ms / 1e3) / 1e9
}

fn fill_f64(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|index| ((index + seed) % 29) as f64 * 0.03125 - 0.4375)
        .collect()
}

fn raw_best_ms(
    pool: &rayon::ThreadPool,
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    reps: usize,
) -> (f64, f64) {
    let mut c = vec![0.0; m * n];
    for _ in 0..16 {
        pool.install(|| probe_dgemm(m, k, n, a, b, &mut c));
    }

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        c.fill(0.0);
        let start = Instant::now();
        pool.install(|| probe_dgemm(m, k, n, a, b, &mut c));
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(c[n - 1]);
    }
    (best, c.iter().sum())
}

fn raw_tb_best_ms(
    pool: &rayon::ThreadPool,
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    reps: usize,
) -> (f64, f64) {
    let mut c = vec![0.0; m * n];
    for _ in 0..16 {
        pool.install(|| probe_dgemm_tb(m, k, n, a, b, &mut c));
    }

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        c.fill(0.0);
        let start = Instant::now();
        pool.install(|| probe_dgemm_tb(m, k, n, a, b, &mut c));
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(c[n - 1]);
    }
    (best, c.iter().sum())
}

fn wrapper_best_ms(
    pool: &rayon::ThreadPool,
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    reps: usize,
) -> (f64, f64) {
    let a_meta = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
    let b_meta = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
    for _ in 0..16 {
        pool.install(|| matmul_tensor_contiguous_f64(a, b, &a_meta, &b_meta).unwrap());
    }

    let mut best = f64::INFINITY;
    let mut checksum = 0.0;
    for _ in 0..reps {
        let start = Instant::now();
        let c = pool.install(|| matmul_tensor_contiguous_f64(a, b, &a_meta, &b_meta).unwrap());
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        checksum = c.iter().sum();
        std::hint::black_box(c[n - 1]);
    }
    (best, checksum)
}

fn reverse_apply_probe(reps: usize) {
    const SHAPES: [(usize, usize, usize); 3] = [(32, 512, 512), (32, 32, 512), (512, 32, 512)];
    const PAR_MIN_FMA: usize = 1 << 24;

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("serial rayon pool");
    let eight_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("eight-thread rayon pool");

    println!(
        "reverse_apply_raw_dgemm host={} reps={reps} serial_rayon={} eight_rayon={} par_min_fma={PAR_MIN_FMA}",
        std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
        serial_pool.install(rayon::current_num_threads),
        eight_pool.install(rayon::current_num_threads),
    );
    println!(
        "shape        fma       raw_1t ms/GF/s      raw_8t ms/GF/s   8t/1t  wrapper_8t ms/GF/s wrapper/raw checksum"
    );

    for (shape_index, (m, k, n)) in SHAPES.into_iter().enumerate() {
        let a = fill_f64(m * k, 3 + shape_index);
        let b = fill_f64(k * n, 11 + shape_index);
        let fma = m * k * n;
        let (raw_1t_ms, raw_1t_sum) = raw_best_ms(&serial_pool, m, k, n, &a, &b, reps);
        let (raw_8t_ms, raw_8t_sum) = raw_best_ms(&eight_pool, m, k, n, &a, &b, reps);
        let (wrapper_ms, wrapper_sum) = wrapper_best_ms(&eight_pool, m, k, n, &a, &b, reps);
        assert_eq!(raw_1t_sum.to_bits(), raw_8t_sum.to_bits(), "raw checksums differ");
        assert_eq!(raw_1t_sum.to_bits(), wrapper_sum.to_bits(), "wrapper checksum differs");
        println!(
            "{m:>3}x{k:<3}x{n:<3} {fma:>8} {raw_1t_ms:>7.3}/{:>5.1} {raw_8t_ms:>7.3}/{:>5.1} {:>6.3} {wrapper_ms:>7.3}/{:>5.1} {:>7.3} {raw_1t_sum:.6}",
            gflops(m, k, n, raw_1t_ms),
            gflops(m, k, n, raw_8t_ms),
            raw_8t_ms / raw_1t_ms,
            gflops(m, k, n, wrapper_ms),
            wrapper_ms / raw_8t_ms,
        );
    }
    println!(
        "Interpretation: every listed shape has 8.39M FMA, below the 16.78M FMA policy floor; raw_8t therefore exercises the same serial dgemm branch under an eight-thread pool."
    );
}

fn ormqr_left_tb_probe(reps: usize) {
    let (m, k, n) = (32usize, 512usize, 512usize);
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("serial rayon pool");
    let eight_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("eight-thread rayon pool");
    let a = fill_f64(k * m, 47);
    let b = fill_f64(k * n, 59);
    let (serial_ms, serial_sum) = raw_tb_best_ms(&serial_pool, m, k, n, &a, &b, reps);
    let (eight_ms, eight_sum) = raw_tb_best_ms(&eight_pool, m, k, n, &a, &b, reps);
    assert_eq!(serial_sum.to_bits(), eight_sum.to_bits(), "TB checksums differ");
    println!(
        "ormqr_left_tb_raw host={} reps={reps} shape={m}x{k}x{n} raw_1t={serial_ms:.4}ms/{:.1}GFLOP/s raw_8t={eight_ms:.4}ms/{:.1}GFLOP/s 8t/1t={:.3} checksum={eight_sum:.6}",
        std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
        gflops(m, k, n, serial_ms),
        gflops(m, k, n, eight_ms),
        eight_ms / serial_ms,
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).is_some_and(|argument| argument == "--reverse-apply") {
        let reps = argv.get(2).and_then(|argument| argument.parse().ok()).unwrap_or(64);
        assert!(reps > 0, "reverse-apply reps must be positive");
        reverse_apply_probe(reps);
        return;
    }
    if argv.get(1).is_some_and(|argument| argument == "--ormqr-left-tb") {
        let reps = argv.get(2).and_then(|argument| argument.parse().ok()).unwrap_or(64);
        assert!(reps > 0, "ormqr-left-tb reps must be positive");
        ormqr_left_tb_probe(reps);
        return;
    }

    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let a: Vec<f64> = (0..m * k).map(|i| (i % 101) as f64 * 0.01).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i % 103) as f64 * 0.01).collect();
    let am = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
    let bm = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);

    // Force single-threaded for a per-core comparison.
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let bench = |f: &dyn Fn() -> Vec<f64>| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let r = std::hint::black_box(f());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&r);
        }
        best
    };

    let bn = bench(&|| naive(&a, &b, m, k, n));
    let bb = bench(&|| blocked(&a, &b, m, k, n));
    let bp = pool1.install(|| bench(&|| matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap()));

    // correctness sanity vs naive (tolerance)
    let rn = naive(&a, &b, m, k, n);
    let rb = blocked(&a, &b, m, k, n);
    let maxdiff = rn
        .iter()
        .zip(rb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);

    eprintln!("gemm 1024^3 (1 core):");
    eprintln!(
        "  naive triple loop : {bn:.1} ms  {:.1} GF/s",
        gflops(m, k, n, bn)
    );
    eprintln!(
        "  blocked 4x8       : {bb:.1} ms  {:.1} GF/s  (blocked-vs-naive maxdiff {maxdiff:.2e})",
        gflops(m, k, n, bb)
    );
    eprintln!(
        "  matrixmultiply    : {bp:.1} ms  {:.1} GF/s",
        gflops(m, k, n, bp)
    );
    eprintln!("  blocked/matrixmultiply per-core ratio: {:.2}x", bp / bb);
}

#[cfg(test)]
mod tests {
    use super::fill_f64;
    use ft_core::{DType, Device, TensorMeta};
    use ft_kernel_cpu::{matmul_tensor_contiguous_f64, probe_dgemm};

    #[test]
    fn reverse_apply_shapes_match_raw_dgemm_and_wrapper_bit_exact() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("eight-thread rayon pool");
        for (shape_index, (m, k, n)) in [(32, 512, 512), (32, 32, 512), (512, 32, 512)]
            .into_iter()
            .enumerate()
        {
            let a = fill_f64(m * k, 3 + shape_index);
            let b = fill_f64(k * n, 11 + shape_index);
            let mut raw = vec![0.0; m * n];
            pool.install(|| probe_dgemm(m, k, n, &a, &b, &mut raw));
            let a_meta = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
            let b_meta = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
            let wrapper = pool
                .install(|| matmul_tensor_contiguous_f64(&a, &b, &a_meta, &b_meta).unwrap());
            assert_eq!(raw.len(), wrapper.len());
            for (index, (raw_value, wrapper_value)) in raw.iter().zip(wrapper).enumerate() {
                assert_eq!(raw_value.to_bits(), wrapper_value.to_bits(), "shape {m}x{k}x{n}, index {index}");
            }
        }
    }
}
