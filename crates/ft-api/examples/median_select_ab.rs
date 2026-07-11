//! Same-process A/B: f32 median selection via total_cmp closure vs sortable-u32 native Ord.
//! Asserts identical result. min-of-9.
use std::time::Instant;

#[inline]
fn key(v: f32) -> u32 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}
#[inline]
fn unkey(k: u32) -> f32 {
    let b = if k & 0x8000_0000 != 0 {
        k & 0x7fff_ffff
    } else {
        !k
    };
    f32::from_bits(b)
}

fn main() {
    let n = 16_000_000usize;
    // CLUSTERED positive data (matches norm_select_sweep): worst case for filter-radix
    // (sign bit constant, exponent clustered -> active set barely shrinks).
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 1_000_003) as f32 * 0.001)
        .collect();
    let mid = (n - 1) / 2;

    // A: total_cmp closure (current impl)
    let a = || {
        let mut s = data.clone();
        *s.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
    };
    // B: sortable-u32 native Ord
    let b = || {
        let mut keys: Vec<u32> = data.iter().map(|&v| key(v)).collect();
        let k = *keys.select_nth_unstable(mid).1;
        unkey(k)
    };
    // C: parallel u32 transform + native select
    let c = || {
        use rayon::prelude::*;
        let mut keys: Vec<u32> = data.par_iter().map(|&v| key(v)).collect();
        let k = *keys.select_nth_unstable(mid).1;
        unkey(k)
    };

    // D: parallel radix-select on u32 keys (no full sort). rank-th order statistic.
    let d = || {
        use rayon::prelude::*;
        let mut active: Vec<u32> = data.par_iter().map(|&v| key(v)).collect();
        let mut rank = mid;
        let mut answer = 0u32;
        for byte in (0..4).rev() {
            let shift = byte * 8;
            // chunk-local 256-bucket histograms, summed (parallel).
            let hist = active
                .par_chunks(1 << 16)
                .map(|ch| {
                    let mut h = [0usize; 256];
                    for &k in ch {
                        h[((k >> shift) & 0xff) as usize] += 1;
                    }
                    h
                })
                .reduce(
                    || [0usize; 256],
                    |mut a, b| {
                        for i in 0..256 {
                            a[i] += b[i];
                        }
                        a
                    },
                );
            // find the bucket holding `rank`.
            let mut cum = 0usize;
            let mut chosen = 0u8;
            for (bk, &c) in hist.iter().enumerate() {
                if cum + c > rank {
                    chosen = bk as u8;
                    break;
                }
                cum += c;
            }
            answer |= (chosen as u32) << shift;
            rank -= cum;
            if byte == 0 {
                break;
            }
            active = active
                .par_iter()
                .copied()
                .filter(|&k| ((k >> shift) & 0xff) as u8 == chosen)
                .collect();
        }
        unkey(answer)
    };

    // E: masked-histogram radix-select — full-array scan each pass, count only
    // prefix-matching elements (NO filter allocation). Distribution-independent.
    let e = || {
        use rayon::prelude::*;
        let keys: Vec<u32> = data.par_iter().map(|&v| key(v)).collect();
        let mut rank = mid;
        let mut prefix = 0u32;
        let mut pmask = 0u32;
        for byte in (0..4).rev() {
            let shift = byte * 8;
            let hist = keys
                .par_chunks(1 << 16)
                .map(|ch| {
                    let mut h = [0usize; 256];
                    for &k in ch {
                        if k & pmask == prefix {
                            h[((k >> shift) & 0xff) as usize] += 1;
                        }
                    }
                    h
                })
                .reduce(
                    || [0usize; 256],
                    |mut a, b| {
                        for i in 0..256 {
                            a[i] += b[i];
                        }
                        a
                    },
                );
            let mut cum = 0usize;
            let mut chosen = 0u8;
            for (bk, &c) in hist.iter().enumerate() {
                if cum + c > rank {
                    chosen = bk as u8;
                    break;
                }
                cum += c;
            }
            prefix |= (chosen as u32) << shift;
            pmask |= 0xffu32 << shift;
            rank -= cum;
        }
        unkey(prefix)
    };

    let ra = a();
    let rb = b();
    let rc = c();
    let rd = d();
    let re = e();
    assert_eq!(
        ra.to_bits(),
        re.to_bits(),
        "masked-histogram median differs"
    );
    assert_eq!(
        ra.to_bits(),
        rb.to_bits(),
        "u32-key median differs from total_cmp"
    );
    assert_eq!(
        ra.to_bits(),
        rc.to_bits(),
        "parallel u32-key median differs"
    );
    assert_eq!(ra.to_bits(), rd.to_bits(), "radix-select median differs");

    let bench = |f: &dyn Fn() -> f32| {
        let mut best = f64::INFINITY;
        for _ in 0..9 {
            let t = Instant::now();
            let v = f();
            let e = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(v);
            if e < best {
                best = e;
            }
        }
        best
    };
    let ta = bench(&a);
    let tb = bench(&b);
    let tc = bench(&c);
    let td = bench(&d);
    println!(
        "f32 median select, n={n}, median={ra}, threads={}, min-of-9",
        rayon::current_num_threads()
    );
    println!("  A total_cmp closure   {ta:7.3} ms");
    println!(
        "  B u32-key native      {tb:7.3} ms  ({:.2}x vs A)",
        ta / tb
    );
    println!(
        "  C u32-key par-xform   {tc:7.3} ms  ({:.2}x vs A)",
        ta / tc
    );
    println!(
        "  D radix-filter par    {td:7.3} ms  ({:.2}x vs A)",
        ta / td
    );
    let te = bench(&e);
    println!(
        "  E radix-masked par    {te:7.3} ms  ({:.2}x vs A)",
        ta / te
    );
}
