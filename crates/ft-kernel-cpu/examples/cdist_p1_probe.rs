//! Bit-exact golden + timing for the cdist/pdist p=1 (Manhattan) kernels
//! (cdist_forward_f64 / pdist_forward_f64). The p=1 path currently pays a libm
//! `powf(1.0)` per element (an exact no-op: pow(x,1)==x) plus a final powf; the
//! fast path elides both. Digest MUST be unchanged (bit-exact). frankentorch-cdist-p1.
//!
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example cdist_p1_probe
use ft_kernel_cpu::{cdist_forward_f64, pdist_forward_f64};
use std::time::Instant;
fn fnv(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
fn bench<F: FnMut() -> u64>(mut f: F, it: usize) -> (f64, u64) {
    let d = f();
    let t = Instant::now();
    for _ in 0..it {
        std::hint::black_box(f());
    }
    (t.elapsed().as_secs_f64() * 1e3 / it as f64, d)
}
fn main() {
    // cdist: batch=1, P x R rows of dim M.
    for &(pp, rr, mm) in &[(256usize, 256usize, 64usize), (512, 512, 128)] {
        let x1: Vec<f64> = (0..pp * mm)
            .map(|i| ((i * 7 % 101) as f64) * 0.013 - 0.5)
            .collect();
        let x2: Vec<f64> = (0..rr * mm)
            .map(|i| ((i * 5 % 97) as f64) * 0.011 - 0.4)
            .collect();
        let (ms, dg) = bench(|| fnv(&cdist_forward_f64(&x1, &x2, 1, pp, rr, mm, 1.0)), 5);
        println!("cdist_p1 P={pp} R={rr} M={mm}  {ms:8.3}ms  digest={dg:016x}");
    }
    // pdist: n rows of dim m.
    for &(nn, mm) in &[(512usize, 64usize), (768, 128usize)] {
        let inp: Vec<f64> = (0..nn * mm)
            .map(|i| ((i * 3 % 89) as f64) * 0.017 - 0.6)
            .collect();
        let (ms, dg) = bench(|| fnv(&pdist_forward_f64(&inp, nn, mm, 1.0)), 5);
        println!("pdist_p1 N={nn} M={mm}  {ms:8.3}ms  digest={dg:016x}");
    }
}
