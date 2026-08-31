//! `frankentorch-f32-inplace-accessor-gap-5fxq2` — ONE falsifiable probe of the bandwidth
//! hypothesis for the f32 unary in-place residual.
//!
//! # What is being decided
//!
//! After the clone (ledger 289) and the split borrow (289b) landed, `inplace_neg_f32` still read
//! ~2.18x slower than torch, and routing the arithmetic to native f32 was measured and REFUTED
//! (289c) — it did not move our arm at all. Effective bandwidth reframed the question:
//!
//!                    FT                PyTorch
//!     neg_      80-83 GB/s     110 / 174 / 211 GB/s
//!     mul_      68-71 GB/s      52 /  70 /  83 GB/s
//!
//! We are bandwidth-CONSISTENT across both lanes. Torch matches us on `mul_` and then does two to
//! three times ITS OWN `mul_` bandwidth on `neg_`. So the residual is not "our neg_ is slow", it is
//! "torch's neg_ is not moving 51 MB of DRAM traffic".
//!
//! This host is a Threadripper PRO 5975WX: L2 16 MiB aggregate, **L3 128 MiB** (4 x 32 MiB). The
//! existing lane's 24.5 MiB buffer is therefore comfortably L3-RESIDENT, which is the single fact
//! that makes the hypothesis testable — push the buffer past 128 MiB and cache residency has to
//! stop being available to either arm.
//!
//! # PREDICTIONS, REGISTERED BEFORE THE RUN
//!
//! The point of a sweep is that rival explanations predict different CURVE SHAPES, so the result
//! cannot be narrated after the fact. `feedback_size_gate_is_the_result` records a lever that was
//! +1.147x at one size and -0.828x at another; a single shape is not a measurement.
//!
//! | size    | (a) CACHE RESIDENCY | (c) PER-ELEMENT OVERHEAD | (d) FIXED PER-CALL COST |
//! |---------|---------------------|--------------------------|-------------------------|
//! | 2 MiB   | large gap           | ~2.2x                    | VERY large gap          |
//! | 24 MiB  | large gap (~2.2x)   | ~2.2x                    | ~2.2x                   |
//! | 128 MiB | shrinking           | ~2.2x                    | small                   |
//! | 256 MiB | **-> ~1.0x**        | **~2.2x**                | **-> ~1.0x**            |
//!
//! (a) and (d) both collapse at the top end, so the RATIO alone cannot separate them. The
//! discriminator is the ABSOLUTE difference, printed beside every ratio: (d) is a constant number
//! of milliseconds at every size, (a) is a difference that GROWS with size while the ratio shrinks.
//! (c) is the null result — our loop is simply slower per element and none of this is about memory.
//!
//! A fourth candidate, non-temporal stores, is deliberately NOT in the table. An in-place
//! read-modify-write already owns the line from its load, so there is no read-for-ownership left
//! for a streaming store to avoid; it predicts nothing distinguishable here and I am not going to
//! pretend it does.
//!
//! # Why a standalone harness
//!
//! These fixtures total ~410 MiB per arm. Putting them in `gauntlet_lane_sweep_h2h` would make
//! every peer's board run allocate that at startup for lanes they did not ask for. Same reason
//! `pad_h2h` and `pdist_f32_h2h` are separate, and this file borrows their protocol wholesale.
//!
//!   RAYON_NUM_THREADS=16 PYTORCH_PYTHON=/path/to/python \
//!     cargo run --release -p frankentorch-api --example inplace_size_sweep_h2h

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    BALANCED_SQUARE, QUIT_REQUEST, READY_MARKER, SAMPLE_LOOP_PY, interpreter_args,
    parse_sample_line, sample_request,
};
use ft_core::ExecutionMode;

/// Sizes in f32 ELEMENTS, against this host's 16 MiB L2 aggregate and 128 MiB L3.
/// The last one is twice L3, which is the whole point: neither arm can be cache-resident there.
const SIZES: [(&str, usize); 4] = [
    ("l2_2MiB", 524_288),
    ("l3_24MiB", 6_422_528),
    ("l3_128MiB", 33_554_432),
    ("dram_256MiB", 67_108_864),
];

const NULL_MIN: f64 = 0.97;
const NULL_MAX: f64 = 1.03;

/// The same generator both arms use, so the checksums compare like for like.
fn seq_f32(n: usize) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation)]
    (0..n)
        .map(|i| (((i % 251) as f64) * 0.001 - 0.12) as f32)
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timings"));
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        f64::midpoint(values[n / 2 - 1], values[n / 2])
    }
}

fn paired_slot_median(values: [f64; 2]) -> f64 {
    f64::midpoint(values[0], values[1])
}

/// One in-place `neg_` on a freshly built no-grad leaf. The leaf is built OUTSIDE the clock, which
/// is not a nicety here: the op MUTATES its input, so a reused leaf would hand each sample the
/// previous sample's output. The incumbent's `base.detach().clone()` is the same shape of work in
/// the same place, so neither arm is timing its allocation.
fn timed_neg_f32(values: &[f32]) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable_f32(values.to_vec(), vec![values.len()], false)
        .expect("f32 leaf");
    let started = Instant::now();
    session.tensor_neg_(x).expect("neg_ f32");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = session
        .tensor_values_f32(x)
        .expect("values")
        .iter()
        .map(|v| f64::from(v.abs()))
        .sum::<f64>();
    (elapsed, checksum)
}

fn incumbent_sample(
    stdin: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
    lane: &str,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    writeln!(stdin, "{}", sample_request(lane))?;
    stdin.flush()?;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!("PyTorch arm closed before answering {lane}").into());
        }
        if let Some(sample) = parse_sample_line(line.trim()) {
            if sample.lane == lane {
                return Ok((sample.milliseconds, sample.gradient_checksum));
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reps: usize = std::env::var("FT_H2H_REPS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(12);
    let python =
        std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "/data/tmp/torchvenv-2121/bin/python".to_owned());

    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} nproc={} rayon={} reps={reps} loadavg={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_owned()),
        load.split_whitespace().take(3).collect::<Vec<_>>().join(","),
    );
    println!("HOST CACHE  L2 16 MiB aggregate, L3 128 MiB (4 x 32 MiB) — Threadripper PRO 5975WX");
    println!(
        "PREDICTION REGISTERED BEFORE MEASURING:\n  \
         (a) CACHE RESIDENCY  ratio collapses toward 1.0 at 256 MiB, and the ABSOLUTE ms gap GROWS with size\n  \
         (c) PER-ELEMENT      ratio stays ~2.2x at EVERY size\n  \
         (d) FIXED PER-CALL   ratio collapses toward 1.0 AND the absolute ms gap stays CONSTANT\n  \
         ratio and absolute gap are both printed so (a) and (d) cannot be conflated after the fact."
    );

    // Fixtures: `seq` on both arms, and the incumbent accumulates its checksum in float64 so the
    // parity column compares two f64 sums rather than an f32 sum against an f64 one. That is
    // teardown on both sides, after the clock.
    let mut py_setup = String::from(
        "import torch, time\n\
         print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)\n\
         torch.set_num_threads(8)\n\
         print('PT_TIMED_STEPS inplace_op', flush=True)\n\
         def seq(n):\n\
         \x20   return ((((torch.arange(n,dtype=torch.int64)%251).double())*0.001-0.12)).float()\n\
         def run(base, fn):\n\
         \x20   x=base.detach().clone()\n\
         \x20   s=time.perf_counter()\n\
         \x20   fn(x)\n\
         \x20   elapsed=(time.perf_counter()-s)*1e3\n\
         \x20   return elapsed, x.abs().sum(dtype=torch.float64).item()\n",
    );
    for (name, len) in SIZES {
        py_setup.push_str(&format!("t_{name}=seq({len})\n"));
    }
    py_setup.push_str("LANES = {\n");
    for (name, _) in SIZES {
        py_setup.push_str(&format!("  \"{name}\": (t_{name}, lambda x: x.neg_()),\n"));
    }
    py_setup.push_str("}\n");
    let program = format!("{py_setup}{SAMPLE_LOOP_PY}");

    let mut child = Command::new(&python)
        .args(interpreter_args(&program))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    let mut reader = BufReader::new(child.stdout.take().ok_or("no stdout")?);

    let mut preamble = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!("PyTorch arm ({python}) exited before {READY_MARKER}: {preamble}").into());
        }
        if line.trim() == READY_MARKER {
            break;
        }
        preamble.push_str(&line);
    }
    print!("{preamble}");

    println!(
        "\n{:<13} {:>10} {:>10} {:>10} {:>12} {:>9} {:>9} {:>7}",
        "size", "FT ms", "PT ms", "verdict", "abs gap ms", "PT null", "FT null", "parity"
    );

    let mut rows: Vec<(String, f64, f64, f64, bool)> = Vec::new();
    for (name, len) in SIZES {
        let data = seq_f32(len);
        // Warm both arms at THIS size before timing it.
        for _ in 0..3 {
            std::hint::black_box(timed_neg_f32(&data));
            incumbent_sample(&mut stdin, &mut reader, name)?;
        }
        let (mut ft_times, mut pt_times) = (Vec::new(), Vec::new());
        let (mut ft_a, mut ft_b, mut pt_a, mut pt_b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut ft_checksum, mut pt_checksum) = (0.0, 0.0);
        for _ in 0..reps {
            let (mut ft_slots, mut pt_slots) = (Vec::with_capacity(4), Vec::with_capacity(4));
            for incumbent_slot in BALANCED_SQUARE {
                if incumbent_slot {
                    let (milliseconds, checksum) = incumbent_sample(&mut stdin, &mut reader, name)?;
                    pt_slots.push(milliseconds);
                    pt_checksum = checksum;
                } else {
                    let (milliseconds, checksum) = timed_neg_f32(&data);
                    ft_slots.push(milliseconds);
                    ft_checksum = checksum;
                }
            }
            ft_a.push(paired_slot_median([ft_slots[0], ft_slots[1]]));
            ft_b.push(paired_slot_median([ft_slots[2], ft_slots[3]]));
            pt_a.push(paired_slot_median([pt_slots[0], pt_slots[1]]));
            pt_b.push(paired_slot_median([pt_slots[2], pt_slots[3]]));
            ft_times.push(median(ft_slots));
            pt_times.push(median(pt_slots));
        }
        let ft = median(ft_times);
        let pt = median(pt_times);
        let pt_null = median(pt_a) / median(pt_b);
        let ft_null = median(ft_a) / median(ft_b);
        let parity = (ft_checksum - pt_checksum).abs() <= 1e-6 * pt_checksum.abs().max(1.0);
        let slower = ft / pt;
        let quotable =
            (NULL_MIN..=NULL_MAX).contains(&pt_null) && (NULL_MIN..=NULL_MAX).contains(&ft_null);
        println!(
            "{name:<13} {ft:>10.3} {pt:>10.3} {:>10} {:>12.3} {pt_null:>9.3} {ft_null:>9.3} {:>7}",
            format!("{slower:.2}x slow"),
            ft - pt,
            if parity { "match" } else { "MISMATCH" },
        );
        if !quotable {
            println!("    ^ NOT QUOTABLE: a null is outside {NULL_MIN:.2}..{NULL_MAX:.2}");
        }
        rows.push((name.to_owned(), ft, pt, slower, quotable && parity));
    }

    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    child.wait()?;

    // The verdict is read off the SHAPE, against the table registered above.
    println!("\nVERDICT");
    if let (Some(first), Some(last)) = (rows.first(), rows.last()) {
        let ratio_collapse = first.3 / last.3;
        let gap_first = first.1 - first.2;
        let gap_last = last.1 - last.2;
        println!(
            "  ratio {:.2}x at {} -> {:.2}x at {}  (collapse factor {ratio_collapse:.2})",
            first.3, first.0, last.3, last.0
        );
        println!("  absolute gap {gap_first:.3} ms -> {gap_last:.3} ms");
        if last.3 < 1.25 && gap_last > gap_first * 2.0 {
            println!("  => (a) CACHE RESIDENCY: ratio collapsed past L3 while the absolute gap grew.");
        } else if last.3 < 1.25 && (gap_last - gap_first).abs() <= gap_first.abs().max(0.05) {
            println!("  => (d) FIXED PER-CALL COST: ratio collapsed and the absolute gap held constant.");
        } else if ratio_collapse < 1.3 {
            println!(
                "  => (c) PER-ELEMENT: the ratio did NOT collapse past L3, so this is not about \
                 cache residency. The bandwidth framing is REFUTED."
            );
        } else {
            println!("  => NONE OF THE REGISTERED SHAPES. Report the curve, do not narrate it.");
        }
    }
    Ok(())
}
