//! Anchored A/B: is a dim=0 (outer_size==1) f32 cat a true bandwidth wall, or
//! does parallel-split memcpy beat the single-threaded copy_from_slice?
//! Pure kernel probe — no torch, no session. min-of-N, same process.
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let n = 8_000_000usize; // each input
    let a: Vec<f32> = (0..n).map(|i| (i % 9973) as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 7919) as f32 * 0.25).collect();
    let out_numel = 2 * n;
    let threads = rayon::current_num_threads();

    // Strategy S: current serial path — two copy_from_slice into one row.
    let serial = || {
        let mut out = vec![0.0f32; out_numel];
        out[0..n].copy_from_slice(&a);
        out[n..2 * n].copy_from_slice(&b);
        out
    };
    // Strategy P: split the output into fixed chunks, each chunk copies from
    // whichever source it overlaps (disjoint dst via par_chunks_mut). Bit-identical.
    let blocks: Vec<(&[f32], usize)> = vec![(&a[..], 0usize), (&b[..], n)]; // (src, dst_start)
    let parallel = || {
        let mut out = vec![0.0f32; out_numel];
        const CHUNK: usize = 1 << 18; // 256K elems = 1MB
        out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, oc)| {
            let g0 = ci * CHUNK;
            let g1 = g0 + oc.len();
            // copy the slice of each block that intersects [g0,g1)
            for &(src, dst_start) in &blocks {
                let bs = dst_start;
                let be = dst_start + src.len();
                let lo = g0.max(bs);
                let hi = g1.min(be);
                if lo < hi {
                    let oc_lo = lo - g0;
                    let oc_hi = hi - g0;
                    let s_lo = lo - bs;
                    let s_hi = hi - bs;
                    oc[oc_lo..oc_hi].copy_from_slice(&src[s_lo..s_hi]);
                }
            }
        });
        out
    };

    // correctness
    let s = serial();
    let p = parallel();
    assert!(s == p, "parallel cat not bit-identical to serial");

    let bench = |f: &dyn Fn() -> Vec<f32>| {
        let mut best = f64::INFINITY;
        for _ in 0..9 {
            let t = Instant::now();
            let o = f();
            let e = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&o);
            if e < best { best = e; }
        }
        best
    };
    let ts = bench(&serial);
    let tp = bench(&parallel);
    let mb = (out_numel * 4) as f64 / 1e6;
    println!("cat dim0 outer=1, out={out_numel} ({mb:.0}MB), threads={threads}, min-of-9");
    println!("  serial   {ts:7.3} ms  ({:.1} GB/s w)", mb / 1e3 / (ts / 1e3));
    println!("  parallel {tp:7.3} ms  ({:.1} GB/s w)  => {:.2}x", mb / 1e3 / (tp / 1e3), ts / tp);
}
