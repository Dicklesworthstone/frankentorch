//! What dense-write bandwidth is actually achievable on this host?
//! `frankentorch-zoqws`, step 0.
//!
//! `frankentorch-87sz8` measured 84% of `max_pool3d`'s backward as the cost of
//! materialising a dense 8 MiB f64 gradient, at ~6.3 GB/s, against PyTorch's
//! ~26 GB/s for its whole op. But that 6.3 GB/s was the floor for **one** write
//! implementation — `par_chunks_mut` over 64 planes with scalar `f64` stores —
//! not a proven optimum. Attacking a kernel before knowing the achievable number
//! would be choosing a lever without bounding the phase.
//!
//! So this establishes the ceiling first. Same 8 MiB f64 buffer the pooling
//! backward produces, filled every way worth trying, reported in GB/s.
//!
//! Nothing here touches a kernel. The output is a number that tells the next
//! person whether `zoqws` has headroom at all, and if so which pattern to reach
//! for.
//!
//! Run: `cargo run --release -p ft-api --features fair-alloc --example dense_write_bandwidth_ladder`

use std::process::Command;
use std::time::Instant;

use rayon::prelude::*;
use wide::f64x4;

/// 1 Mi f64 = 8 MiB — the gradient `max_pool3d [2,32,16,32,32]` returns.
const N: usize = 1 << 20;
const BYTES: usize = N * 8;
const REPS: usize = 21;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn executable_sha256() -> String {
    let executable = std::env::current_exe().expect("current executable must be available");
    let output = Command::new("sha256sum")
        .arg(executable)
        .output()
        .expect("sha256sum must be available");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256sum output must be UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum must print a digest")
        .to_owned()
}

fn bench<F: FnMut()>(label: &str, mut f: F, rows: &mut Vec<(String, f64, f64)>) {
    for _ in 0..3 {
        f();
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        f();
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let ms = median(samples);
    #[allow(clippy::cast_precision_loss)]
    let gib_s = BYTES as f64 / (ms / 1_000.0) / (1024.0 * 1024.0 * 1024.0);
    rows.push((label.to_owned(), ms, gib_s));
}

fn main() {
    println!("executing_elf_sha256={}", executable_sha256());
    println!(
        "allocator={}",
        if cfg!(feature = "fair-alloc") {
            "mimalloc (--features fair-alloc)"
        } else {
            "system (glibc malloc)"
        }
    );
    println!(
        "buffer={} MiB f64 ({N} elements), reps={REPS} median, threads={}\n",
        BYTES / (1024 * 1024),
        rayon::current_num_threads()
    );

    let mut rows: Vec<(String, f64, f64)> = Vec::new();

    // Lower bound: allocate zeroed and never touch. Pages stay unfaulted, so this
    // is allocator bookkeeping only — NOT a write rate, included as the floor.
    bench(
        "alloc_zeroed_no_touch (not a write; floor)",
        || {
            std::hint::black_box(vec![0.0f64; N]);
        },
        &mut rows,
    );

    // The pattern the pooling backward actually uses today: chunk per plane
    // (16384 f64 = 128 KiB), scalar stores.
    bench(
        "par_chunks_mut 128KiB, scalar stores  [CURRENT KERNEL PATTERN]",
        || {
            let mut v = vec![0.0f64; N];
            v.par_chunks_mut(16_384).for_each(|row| {
                for slot in row.iter_mut() {
                    *slot = 1.0;
                }
            });
            std::hint::black_box(v);
        },
        &mut rows,
    );

    // Same chunking, but let the compiler emit a memset instead of a scalar loop.
    bench(
        "par_chunks_mut 128KiB, slice::fill",
        || {
            let mut v = vec![0.0f64; N];
            v.par_chunks_mut(16_384).for_each(|row| row.fill(1.0));
            std::hint::black_box(v);
        },
        &mut rows,
    );

    // Chunk-size sweep with fill, to separate task granularity from store width.
    for (elems, label) in [
        (2_048usize, "par_chunks_mut  16KiB, slice::fill"),
        (8_192, "par_chunks_mut  64KiB, slice::fill"),
        (65_536, "par_chunks_mut 512KiB, slice::fill"),
        (262_144, "par_chunks_mut   2MiB, slice::fill"),
    ] {
        bench(
            label,
            || {
                let mut v = vec![0.0f64; N];
                v.par_chunks_mut(elems).for_each(|row| row.fill(1.0));
                std::hint::black_box(v);
            },
            &mut rows,
        );
    }

    // Explicit 256-bit stores, in case neither the scalar loop nor fill vectorises.
    bench(
        "par_chunks_mut 128KiB, wide f64x4 stores",
        || {
            let mut v = vec![0.0f64; N];
            v.par_chunks_mut(16_384).for_each(|row| {
                let ones = f64x4::splat(1.0);
                let mut lane = 0;
                while lane + 4 <= row.len() {
                    row[lane..lane + 4].copy_from_slice(&ones.to_array());
                    lane += 4;
                }
                for slot in &mut row[lane..] {
                    *slot = 1.0;
                }
            });
            std::hint::black_box(v);
        },
        &mut rows,
    );

    // Single-threaded fill, so the parallel rows have something to beat.
    bench(
        "serial slice::fill (1 thread)",
        || {
            let mut v = vec![0.0f64; N];
            v.fill(1.0);
            std::hint::black_box(v);
        },
        &mut rows,
    );

    // Construct already-filled: lets the allocator+init path pick its own strategy.
    bench(
        "vec![1.0; N] (non-zero init at construction)",
        || {
            std::hint::black_box(vec![1.0f64; N]);
        },
        &mut rows,
    );

    // The pooling backward's ACTUAL access shape: touch 1 element in 8, scattered,
    // which still faults every page. Included because a dense fill is not what the
    // kernel does — this is the honest comparison for it.
    bench(
        "par_chunks_mut 128KiB, write 1-in-8 (pool scatter shape)",
        || {
            let mut v = vec![0.0f64; N];
            v.par_chunks_mut(16_384).for_each(|row| {
                let mut i = 0;
                while i < row.len() {
                    row[i] = 1.0;
                    i += 8;
                }
            });
            std::hint::black_box(v);
        },
        &mut rows,
    );

    let mut best = 0.0f64;
    let mut best_label = String::new();
    println!("{:<58} {:>9}  {:>10}", "pattern", "ms", "GiB/s");
    for (label, ms, gib) in &rows {
        println!("  {label:<56} {ms:7.3}  {gib:8.2}");
        if *gib > best && !label.starts_with("alloc_zeroed_no_touch") {
            best = *gib;
            best_label.clone_from(label);
        }
    }

    let current = rows
        .iter()
        .find(|(l, _, _)| l.contains("CURRENT KERNEL PATTERN"))
        .map(|(_, _, g)| *g)
        .unwrap_or(0.0);
    println!(
        "\nBest real write pattern: {best_label} at {best:.2} GiB/s.\n\
         Current kernel pattern: {current:.2} GiB/s -> headroom {:.2}x.",
        best / current.max(f64::MIN_POSITIVE)
    );
    println!(
        "\nREAD THIS BEFORE ACTING: headroom near 1.0x means the dense write is already at this\n\
         host's achievable rate and zoqws has no kernel-side lever — the gap to PyTorch would then\n\
         be structural (fewer/other writes, not faster ones), and that is a REJECT for the obvious\n\
         lever. Headroom well above 1.0x means the pattern is the problem and names its replacement."
    );
}
