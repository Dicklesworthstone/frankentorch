//! Re-baseline the gauntlet lanes that never got a live-PyTorch comparison under
//! `--features fair-alloc` — `frankentorch-kgs4`.
//!
//! WHY THIS SHAPE OF MEASUREMENT. `frankentorch-ujw3g` established that the
//! gauntlet's headline ratios are dominated by each lane's per-iteration input
//! rebuild: for `avg_pool1d`, 57% of the step was the caller's 32 MiB `to_vec()`
//! and only ~12% was the pooling forward, and FrankenTorch's *op work* turned out
//! to be at ~1.18x while the whole-step ratio read 1.5-2.0x. A whole-step ratio
//! therefore mostly measures buffer-copy cost, which is allocator-shaped and
//! already settled by `frankentorch-1ji9l` option C.
//!
//! So this sweep times **op work only** — forward + backward with the input built
//! OUTSIDE the timed region on both sides. That is the number that says whether a
//! kernel is competitive, and it is the one a lever could actually move.
//!
//! WHY THE ARMS INTERLEAVE (`frankentorch-6atx2`). This harness used to run its
//! ENTIRE PyTorch arm to completion before the first FrankenTorch lane started,
//! so the two arms were sampled tens of seconds apart and any load shift in that
//! gap landed entirely, and undetectably, in the ratio. The contention preflight
//! could not cover it: it certifies only that nothing heavy sat on the placement
//! CPUs at the instant sampling began, not one second later. Repetition plus a
//! median averaged that effect down; nothing bounded it.
//!
//! The incumbent is therefore driven as a **co-process** — it sets up, warms up,
//! announces readiness, then returns exactly one timed sample per request — and
//! each round takes an incumbent sample immediately beside our samples for the
//! same lane. See `ft_api::harness_interleave` for the protocol and the schedule.
//!
//! THE INCUMBENT'S ESTIMATOR IS UNCHANGED. Interleaving alters *when* samples are
//! taken, not how many or which statistic summarises them. PyTorch is still
//! min-of-`PT_SAMPLES` after 4 warmups; those samples are merely spread evenly
//! across the rounds instead of taken in one block. Had this also switched to
//! min-of-16 the ratio's level would have moved for reasons unrelated to
//! interleaving, and the before/after set could not isolate this change.
//!
//! Four lanes, shapes copied from `pytorch_gauntlet_bench` so the two describe the
//! same workloads: `max_pool1d`, `avg_pool2d`, `conv3d`, `max_pool3d`.
//!
//! Run (must be local; rch workers have no PyTorch):
//! ```text
//! PYTORCH_PYTHON=/path/to/python \
//!   cargo run --release -p ft-api --features fair-alloc --example gauntlet_lane_sweep_h2h
//! ```

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    ArmOrdering, LEGACY_BLOCK_ARMS_ENV, MAX_NULL_CI_WIDTH, QUIT_REQUEST, READY_MARKER, TIMED_STEPS,
    TIMED_STEPS_MARKER, adjudicate_null, arm_ordering_from_env, incumbent_sample_rounds,
    parse_sample_line, parse_timed_steps, sample_request, timed_region_disagreement,
};
use ft_core::ExecutionMode;

/// MUST BE EVEN (`frankentorch-svabf`). Each round runs two timed calls and
/// assigns them to the A/A arms by round parity, which cancels a constant
/// first-call-vs-second-call offset *only if the two positions are used equally
/// often*. At the previous odd value of 15, arm `a` took the first position 8
/// times and the second 7 — a 1-in-15 imbalance that leaks the position effect
/// straight into the null ratio. That is what made `max_pool1d` and `conv3d`
/// report A/A CIs excluding 1.0 (`[0.843,0.986]`, `[0.770,0.992]`) on a busy host
/// while identical code ran in both arms.
const REPS: usize = 16;
/// Incumbent samples per lane. Held at 7 to keep PyTorch's estimator exactly the
/// min-of-7 the banked non-interleaved set used — see the module note on why the
/// estimator must not move in the same change that introduces interleaving.
const PT_SAMPLES: usize = 7;
const BOOTSTRAP_REPS: usize = 2_000;

// Shapes lifted verbatim from pytorch_gauntlet_bench.
const MP1_N: usize = 8;
const MP1_C: usize = 64;
const MP1_L: usize = 8192;

const AP2_N: usize = 8;
const AP2_C: usize = 64;
const AP2_H: usize = 64;
const AP2_W: usize = 64;

const C3_N: usize = 2;
const C3_CI: usize = 32;
const C3_CO: usize = 32;
const C3_D: usize = 8;
const C3_H: usize = 16;
const C3_W: usize = 16;
const C3_K: usize = 3;

const MP3_N: usize = 2;
const MP3_C: usize = 32;
const MP3_D: usize = 16;
const MP3_H: usize = 32;
const MP3_W: usize = 32;

/// One FrankenTorch lane: runs a single timed forward+backward, returning
/// (milliseconds, gradient checksum).
type LaneRun<'a> = Box<dyn Fn() -> (f64, f64) + 'a>;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn minimum(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median_ratio_ci(numerator: &[f64], denominator: &[f64]) -> (f64, f64, f64) {
    assert_eq!(numerator.len(), denominator.len());
    let point = median(numerator.to_vec()) / median(denominator.to_vec());
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut state = 0x2f1e_9b47_c30d_a851_u64;
    for _ in 0..BOOTSTRAP_REPS {
        let mut left = Vec::with_capacity(numerator.len());
        let mut right = Vec::with_capacity(denominator.len());
        for _ in 0..numerator.len() {
            let index = (next_random(&mut state) as usize) % numerator.len();
            left.push(numerator[index]);
            right.push(denominator[index]);
        }
        samples.push(median(left) / median(right));
    }
    samples.sort_by(f64::total_cmp);
    (
        point,
        samples[BOOTSTRAP_REPS * 25 / 1_000],
        samples[BOOTSTRAP_REPS * 975 / 1_000],
    )
}

fn seq(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 251) as f64) * 0.001 - 0.12).collect()
}

/// Time forward+backward with the leaf already built. Returns (ms, grad checksum).
fn timed_op<F>(values: &[f64], shape: Vec<usize>, build: F) -> (f64, f64)
where
    F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId) -> ft_autograd::TensorNodeId,
{
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), shape, true)
        .expect("leaf");
    // Leaf construction is deliberately outside the timer — see the module note.
    let started = Instant::now();
    let out = build(&mut session, x);
    let loss = session.tensor_sum(out).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    // frankentorch-574cu: STOP THE CLOCK HERE. The timed region is exactly
    // `harness_interleave::TIMED_STEPS` — forward, loss_sum, backward — which is
    // what the PyTorch arm times. The gradient checksum below is teardown: it
    // proves both sides computed the same thing, and on the avg_pool2d lane it
    // cost 1.599 ms (a serial dependent-add chain over 2M f64, 24% of that
    // lane's session) that the incumbent's timer never paid. Timing it on one arm
    // only biased every repetition in the same direction, which repetition cannot
    // average out.
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
    (elapsed, checksum)
}

/// Ask the incumbent co-process for exactly one timed sample of `lane`.
///
/// Chatter the child may emit (warnings, notices) is skipped rather than parsed,
/// but a closed stdout is a hard failure: a silently-short arm would otherwise
/// leave the lane's remaining rounds measuring only our side.
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
            return Err(format!(
                "the PyTorch co-process closed its stdout while lane `{lane}` was being sampled; \
                 a partially-measured arm cannot carry a vs-PyTorch claim"
            )
            .into());
        }
        if let Some(sample) = parse_sample_line(&line) {
            assert_eq!(
                sample.lane, lane,
                "co-process answered for the wrong lane; replies would be misfiled"
            );
            return Ok((sample.milliseconds, sample.gradient_checksum));
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The A/A null gate is only meaningful if the two arms use the first- and
    // second-call positions equally often; see the note on REPS.
    assert!(
        REPS.is_multiple_of(2),
        "REPS must be even or the A/A arms are position-imbalanced and the null gate leaks bias"
    );
    // Compile-time: the incumbent cannot take more samples than there are rounds
    // to spread them across, or the schedule would clamp and silently change the
    // estimator this harness is careful to hold fixed.
    const {
        assert!(PT_SAMPLES <= REPS);
    }

    let mp1 = seq(MP1_N * MP1_C * MP1_L);
    let ap2 = seq(AP2_N * AP2_C * AP2_H * AP2_W);
    let c3x = seq(C3_N * C3_CI * C3_D * C3_H * C3_W);
    let c3w = seq(C3_CO * C3_CI * C3_K * C3_K * C3_K);
    let mp3 = seq(MP3_N * MP3_C * MP3_D * MP3_H * MP3_W);

    // Interleaved is the default; the legacy block ordering is an explicit opt-in.
    let legacy_env = std::env::var(LEGACY_BLOCK_ARMS_ENV).ok();
    let ordering = arm_ordering_from_env(legacy_env.as_deref());

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // Setup only. The request/serve/quit loop is appended from the library so the
    // protocol markers cannot drift away from the parser that reads them.
    let py_setup = r#"
import time, torch
import torch.nn.functional as Fn
# frankentorch-wnku0: the arm self-reports its version, in this same invocation,
# BEFORE any timing — so a run that dies mid-measurement still leaves provenance.
print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)
torch.set_num_threads(8)
def seq(n):
    return ((torch.arange(n,dtype=torch.int64)%251).double())*0.001-0.12
mp1=seq(8*64*8192).reshape(8,64,8192)
ap2=seq(8*64*64*64).reshape(8,64,64,64)
c3x=seq(2*32*8*16*16).reshape(2,32,8,16,16)
c3w=seq(32*32*3*3*3).reshape(32,32,3,3,3)
mp3=seq(2*32*16*32*32).reshape(2,32,16,32,32)
# frankentorch-574cu: this arm declares the region it times, so a change to
# `run` below that is not mirrored here fails the run instead of silently
# biasing every ratio. Written as an independent literal rather than generated
# from the Rust constant, which would make the agreement check tautological.
print('PT_TIMED_STEPS forward,loss_sum,backward', flush=True)
def run(base, fn):
    # leaf built OUTSIDE the timed region, matching the FrankenTorch side
    x=base.detach().clone().requires_grad_(True)
    s=time.perf_counter()
    fn(x).sum().backward()          # <- forward, loss_sum, backward: the timed region
    elapsed=(time.perf_counter()-s)*1e3
    # teardown, deliberately AFTER the clock stops on BOTH arms
    return elapsed, x.grad.sum().item()
LANES = {
    "max_pool1d": (mp1, lambda x: Fn.max_pool1d(x,2,2)),
    "avg_pool2d": (ap2, lambda x: Fn.avg_pool2d(x,(2,2),(2,2))),
    "conv3d":     (c3x, lambda x: Fn.conv3d(x,c3w,None,(1,1,1),(1,1,1))),
    "max_pool3d": (mp3, lambda x: Fn.max_pool3d(x,(2,2,2),(2,2,2))),
    # frankentorch-9pafs: the same torch op under a second name. The FrankenTorch
    # side runs this lane with its buffer pool DISABLED, so the pair isolates the
    # pool against one live incumbent inside one invocation. Because the incumbent
    # code is byte-identical for the two names, PT(max_pool3d_nopool)/PT(max_pool3d)
    # is a free control that must come out ~1.0; if it does not, the host moved
    # during the run and the FT comparison is not readable either.
    "max_pool3d_nopool": (mp3, lambda x: Fn.max_pool3d(x,(2,2,2),(2,2,2))),
}
"#;
    let py = format!("{py_setup}{}", ft_api::harness_interleave::SAMPLE_LOOP_PY);

    // `-c`, never `-`: the latter reads the program from stdin until EOF, which
    // deadlocks a co-process whose stdin must stay open for requests.
    let mut child = Command::new(&python)
        .args(ft_api::harness_interleave::interpreter_args(&py))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "could not start the PyTorch arm (`{python}`): {error}. Set PYTORCH_PYTHON to an \
                 interpreter with torch installed; a FrankenTorch-only run cannot carry a \
                 vs-PyTorch claim."
            )
        })?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
    let mut reader = BufReader::new(
        child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("no stdout"))?,
    );
    // Block until the arm has imported torch, built its tensors and warmed every
    // lane. Anything it prints before that (version line, warnings) is preamble.
    let mut preamble = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!(
                "the PyTorch arm exited before announcing `{READY_MARKER}`; set PYTORCH_PYTHON to \
                 an interpreter with torch installed. A FrankenTorch-only run cannot carry a \
                 vs-PyTorch claim. Its output was:\n{preamble}"
            )
            .into());
        }
        if line.trim() == READY_MARKER {
            break;
        }
        preamble.push_str(&line);
    }

    // frankentorch-wnku0: hard-fails if the arm did not self-report, so this
    // harness cannot emit ratios without the version they were measured against.
    let torch_version = ft_api::harness_provenance::require_reported_version(&preamble)?;

    // frankentorch-574cu: both arms must have timed the SAME region, or the
    // ratios below are biased in one direction in every repetition and nothing
    // here is quotable. Fail the run rather than print a caveat nobody reads.
    let incumbent_steps = parse_timed_steps(&preamble).ok_or_else(|| {
        format!(
            "the PyTorch arm did not declare its timed region: no `{TIMED_STEPS_MARKER}` line. \
             Without it there is no evidence the two arms measured the same work."
        )
    })?;
    if let Some(message) = timed_region_disagreement(TIMED_STEPS, &incumbent_steps) {
        return Err(message.into());
    }

    println!(
        "executing_elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "{}",
        ft_api::harness_provenance::incumbent_provenance_block(torch_version, 8)
    );
    println!(
        "allocator={}",
        if cfg!(feature = "fair-alloc") {
            "mimalloc (--features fair-alloc)"
        } else {
            "system (glibc malloc) — re-run with --features fair-alloc before quoting"
        }
    );
    println!(
        "measurement=OP WORK ONLY (forward+backward; leaf built outside the timer on BOTH sides)"
    );
    println!(
        "sampling={} (frankentorch-6atx2); PyTorch min-of-{PT_SAMPLES} after 4 warmups, spread \
         evenly across {REPS} rounds, torch threads=8",
        ordering.label()
    );
    if !ordering.is_quotable() {
        println!(
            "WARNING: {LEGACY_BLOCK_ARMS_ENV} is set, so the whole PyTorch arm ran before the first\n\
             FrankenTorch lane. Any load shift in that gap lands entirely in the ratio. These rows\n\
             exist to be compared against a default run, NOT to be quoted."
        );
    }
    println!();
    println!("lane          FT(ms)    PT(ms)   standing            A/A gate           parity");

    let lanes: Vec<(&str, LaneRun<'_>)> = vec![
        (
            "max_pool1d",
            Box::new(|| {
                timed_op(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_max_pool1d(x, 2, 2).expect("max_pool1d")
                })
            }),
        ),
        (
            "avg_pool2d",
            Box::new(|| {
                timed_op(&ap2, vec![AP2_N, AP2_C, AP2_H, AP2_W], |s, x| {
                    s.functional_avg_pool2d_sum(x, (2, 2), (2, 2), (0, 0), false, true)
                        .expect("avg_pool2d")
                })
            }),
        ),
        (
            "max_pool3d",
            Box::new(|| {
                timed_op(&mp3, vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W], |s, x| {
                    s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))
                        .expect("max_pool3d")
                })
            }),
        ),
        (
            // frankentorch-9pafs: the SAME lane with `ft_core::buffer_pool` off.
            // Two things make this the readable form of the comparison. It is one
            // binary, so the two arms cannot differ by a compilation; and both are
            // sampled against one live incumbent inside one invocation, so the
            // ratio-of-ratios against `max_pool3d` is immune to the host drift
            // that makes cross-run ratios unquotable here.
            "max_pool3d_nopool",
            Box::new(|| {
                ft_core::buffer_pool::set_enabled(false);
                let sample = timed_op(&mp3, vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W], |s, x| {
                    s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))
                        .expect("max_pool3d")
                });
                ft_core::buffer_pool::set_enabled(true);
                sample
            }),
        ),
        (
            "conv3d",
            Box::new(|| {
                timed_op(&c3x, vec![C3_N, C3_CI, C3_D, C3_H, C3_W], |s, x| {
                    let w = s
                        .tensor_variable(c3w.clone(), vec![C3_CO, C3_CI, C3_K, C3_K, C3_K], false)
                        .expect("weight");
                    s.functional_conv3d(x, w, None, (1, 1, 1), (1, 1, 1))
                        .expect("conv3d")
                })
            }),
        ),
    ];

    // Our side warms here; the incumbent warmed itself before announcing ready.
    for (_, run_lane) in &lanes {
        let mut warm = 0.0;
        for _ in 0..3 {
            warm += run_lane().0;
        }
        std::hint::black_box(warm);
    }

    let schedule = incumbent_sample_rounds(REPS, PT_SAMPLES);
    let mut arm_a: Vec<Vec<f64>> = vec![Vec::with_capacity(REPS); lanes.len()];
    let mut arm_b: Vec<Vec<f64>> = vec![Vec::with_capacity(REPS); lanes.len()];
    let mut pt_times: Vec<Vec<f64>> = vec![Vec::with_capacity(PT_SAMPLES); lanes.len()];
    let mut pt_grads: Vec<Option<f64>> = vec![None; lanes.len()];
    let mut checksums: Vec<f64> = vec![0.0; lanes.len()];

    // frankentorch-6atx2 option (a): under the legacy ordering the ENTIRE
    // incumbent arm is drained up front, exactly as this harness behaved before
    // interleaving. Sample counts, estimator and lane order are identical to the
    // interleaved path — only WHEN the incumbent is sampled differs — so a
    // BEFORE/AFTER pair of one binary in one window isolates arm ordering and
    // nothing else. That is the comparison the banked set can no longer support.
    if ordering == ArmOrdering::LegacyBlock {
        for (index, (name, _)) in lanes.iter().enumerate() {
            for _ in 0..PT_SAMPLES {
                let (ms, grad) = incumbent_sample(&mut stdin, &mut reader, name)?;
                pt_times[index].push(ms);
                pt_grads[index] = Some(grad);
            }
        }
    }

    for round in 0..REPS {
        let incumbent_round =
            ordering == ArmOrdering::Interleaved && schedule.binary_search(&round).is_ok();
        for (index, (name, run_lane)) in lanes.iter().enumerate() {
            // The incumbent sample sits immediately beside our samples for the
            // SAME lane, so both arms see the same instant of machine state.
            if incumbent_round {
                let (ms, grad) = incumbent_sample(&mut stdin, &mut reader, name)?;
                pt_times[index].push(ms);
                pt_grads[index] = Some(grad);
            }
            let (first, checksum) = run_lane();
            let (second, _) = run_lane();
            checksums[index] = checksum;
            if round.is_multiple_of(2) {
                arm_a[index].push(first);
                arm_b[index].push(second);
            } else {
                arm_b[index].push(first);
                arm_a[index].push(second);
            }
        }
    }

    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    child.wait()?;

    for (index, (name, _)) in lanes.iter().enumerate() {
        let (nr, nlo, nhi) = median_ratio_ci(&arm_a[index], &arm_b[index]);
        // frankentorch-8ieqm: bracketing 1.0 is not enough — contention WIDENS
        // the null's CI, and a wider CI brackets 1.0 more easily, so the old
        // gate got easier to pass exactly when the host got noisier.
        let null = adjudicate_null(nlo, nhi, MAX_NULL_CI_WIDTH);
        let ft_ms = median(
            arm_a[index]
                .iter()
                .chain(arm_b[index].iter())
                .copied()
                .collect(),
        );
        let (Some(pt_grad), false) = (pt_grads[index], pt_times[index].is_empty()) else {
            println!("  {name:<12} {ft_ms:8.3}       --   PyTorch row missing");
            continue;
        };
        // Estimator preserved from the banked set: min over the lane's samples.
        let pt_ms = minimum(&pt_times[index]);
        let ratio = pt_ms / ft_ms;
        let standing = if ratio >= 1.0 {
            format!("FT {ratio:.2}x FASTER")
        } else {
            format!("FT {:.2}x SLOWER", 1.0 / ratio)
        };
        // Gradient-sum agreement: the two sides must be computing the same thing.
        let tolerance = 1e-6 * pt_grad.abs().max(1.0);
        let parity = if (checksums[index] - pt_grad).abs() <= tolerance {
            "match"
        } else {
            "MISMATCH"
        };
        println!(
            "  {name:<12} {ft_ms:8.3} {pt_ms:8.3}   {standing:<19} {} [{nlo:.3},{nhi:.3}] {:<5} {parity}",
            null.label(),
            format!("{nr:.3}"),
        );
    }
    println!(
        "\nQuote a lane only if its A/A gate says PASS and parity is `match`. WIDE means the null's\n\
         CI exceeded {MAX_NULL_CI_WIDTH:.2} — the sample was too noisy to support ANY verdict, so the row is\n\
         undecidable rather than a win or a loss (frankentorch-8ieqm). Op-work ratios are NOT\n\
         comparable to the gauntlet's whole-step ratios, which include each lane's input rebuild."
    );

    // frankentorch-v92uh: prove the pool was actually SERVING this run rather than
    // sitting inert. Without this line a pooled/unpooled A/B that came out flat
    // would be indistinguishable from one where every `take` missed, and the
    // paired verdict below would be reporting on a lever that never engaged.
    // Requests below `MIN_POOLED_LEN`, and every request made while the pool is
    // disabled, are not counted at all — so these are pooled-lane numbers.
    let pool = ft_core::buffer_pool::stats();
    println!(
        "\nbuffer_pool: hits={} misses={} parked={} buffers / {:.1} MiB",
        pool.hits,
        pool.misses,
        pool.parked_buffers,
        pool.parked_bytes as f64 / (1024.0 * 1024.0)
    );

    // ── frankentorch-v92uh: PAIRED analysis for `X` vs `X_nopool` ───────────
    //
    // The table above compares two lanes through their independent medians, and
    // on a shared host that is the weakest reading available: a load excursion
    // lands in one lane's median and not the other's, and the CI has to be wide
    // enough to cover that. The pair is sampled ADJACENTLY inside each round, so
    // the excursion is very nearly common to both — differencing per round
    // cancels it, and only then is a busy host survivable.
    //
    // Per round the lane's two calls are reduced by MIN, not mean: under peer
    // load the fast sample is the one that measured the work and the slow one
    // measured the interference (frankentorch peer-bench contention note).
    //
    // Reported as `nopool / pooled`, so > 1.0 means the pool is FASTER. The
    // incumbent rows carry their own control: PT is byte-identical code under
    // both names, so PT(nopool)/PT(pooled) must land near 1.0 or the run is not
    // readable at all.
    for (index, (name, _)) in lanes.iter().enumerate() {
        let Some(base) = name.strip_suffix("_nopool") else {
            continue;
        };
        let Some(base_index) = lanes.iter().position(|(other, _)| *other == base) else {
            continue;
        };
        let rounds = arm_a[index].len().min(arm_a[base_index].len());
        let mut treated = Vec::with_capacity(rounds);
        let mut control = Vec::with_capacity(rounds);
        for round in 0..rounds {
            control.push(arm_a[index][round].min(arm_b[index][round]));
            treated.push(arm_a[base_index][round].min(arm_b[base_index][round]));
        }
        let paired: Vec<f64> = control
            .iter()
            .zip(treated.iter())
            .map(|(off, on)| off / on)
            .collect();
        let (point, lo, hi) = median_ratio_ci(&control, &treated);
        let wins = paired.iter().filter(|ratio| **ratio > 1.0).count();
        let pt_control = match (pt_times[index].is_empty(), pt_times[base_index].is_empty()) {
            (false, false) => minimum(&pt_times[index]) / minimum(&pt_times[base_index]),
            _ => f64::NAN,
        };
        println!(
            "\nPAIRED {base}: pool ON vs OFF, one binary, one invocation, per-round min-of-2\n  \
             ratio (off/on) = {point:.3}x  95% CI [{lo:.3},{hi:.3}]  {wins}/{rounds} rounds faster with the pool\n  \
             incumbent control PT(off)/PT(on) = {pt_control:.3} (must be ~1.0; the arm is identical code)\n  \
             verdict: {}",
            if !pt_control.is_finite() || (pt_control - 1.0).abs() >= 0.05 {
                "UNREADABLE — the incumbent control moved, so the host shifted between the two lanes"
            } else if lo > 1.0 {
                "the pool is FASTER by the paired CI"
            } else if hi < 1.0 {
                "the pool is SLOWER by the paired CI"
            } else {
                "UNDECIDED — the paired CI brackets 1.0"
            }
        );
    }
    Ok(())
}
