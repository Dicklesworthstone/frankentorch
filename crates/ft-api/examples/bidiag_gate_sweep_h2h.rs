//! Square-SVD forward vs PyTorch, with our own arms AND the incumbent alternated ROUND BY ROUND
//! inside one process — `frankentorch-bidiag-parallel-gate-fork-thrash-mzrnh`.
//!
//! WHAT IS MEASURED. `linalg.svd` forward only (full U, S, Vh) on a square matrix, ours against
//! PyTorch, plus as many of our own configurations as the caller asks for. A configuration is a
//! tuple: the bidiagonal PARALLEL GATE, the step-(12) kernel, the trailing-update form, and
//! whether the serial panel keeps four independent outputs in flight.
//!
//! WHY EVERY ARM IS IN ONE PROCESS. The gate used to live in a `OnceLock` and the step-(12)
//! kernel was a compile-time choice, so an A/B needed one process per arm — a whole launch, a
//! cold allocator, and a different window between the two numbers being compared. Both are now
//! runtime switches (`bidiag_parallel_gate_set`, `bidiag_rowdot_blocked_set`, and
//! `bidiag_panel_output_blocked_set`).
//!
//! THE ESTIMATOR IS THE INSTRUMENT (NEGATIVE_EVIDENCE item 255). The first version of this lane
//! timed all of arm A, then all of arm B. Its A/A null — two arms with identical settings, whose
//! difference is therefore pure noise — read 1.02x to 1.19x across four invocations on this host,
//! the size of the effects being chased, and three runs of the same n=512 comparison disagreed on
//! the ordering. Now every arm AND the incumbent are sampled once per ROUND, arm order reversed
//! on odd rounds, first round discarded, and every ratio is the median over rounds of the PAIRED
//! per-round ratio. The null fell to 1.001-1.05x on the same host minutes later. The window was
//! never the binding constraint; the pairing was.
//!
//! WHY THE INCUMBENT IS A CO-PROCESS. Item 256 banked a row whose FT figure was a min over nine
//! rounds and whose PyTorch figure was a min over five samples taken in one block seconds away —
//! an estimator asymmetry biased in our favour, and a gap in time nothing bounded. A child that
//! computes everything and exits cannot be interleaved, so the incumbent is driven as a
//! request/response co-process (`ft_api::harness_interleave`), one timed sample per round, the
//! same warmup count as ours.
//!
//! HOW TO GET A NULL. Repeat an arm in `FT_GATE_VALUES`: two identical arms differ only by the
//! window's own noise, and **no effect smaller than that is readable**. A row whose effect does
//! not clear its own null is unresolved, not a result.
//!
//! THE ROUTE PROOF. Each arm reports how many times a gated call site actually took its parallel
//! branch, split by site. The inference this replaced — that two arms producing bit-identical
//! singular values must have taken the same route — is UNSOUND, and this lane's own data refuted
//! it: at n=256 two arms timed 1.32x apart produced identical singular-value sums, because the QR
//! sweep converges to the same rounded values from slightly different bidiagonal input.
//!
//! Run (must be local; rch workers have no PyTorch):
//! ```text
//! RAYON_NUM_THREADS=8 PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
//!   cargo run --release -p frankentorch-api --example bidiag_gate_sweep_h2h
//! ```
//! `FT_GATE_SIZES` (default `128,136,256,512`), `FT_GATE_VALUES` (default the shipped gate and
//! always-serial), `FT_ROWDOT` (`1` = the four-row step-(12) kernel, `0` = the one-row loop it
//! replaced), `FT_PANEL_OUTPUT` (`1` = four exact-order panel outputs, `0` = scalar),
//! `FT_ROUNDS` (default 9) and `FT_H2H_WARMUP` (default 8, read by BOTH arms).

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    QUIT_REQUEST, READY_MARKER, SAMPLE_LOOP_PY, interpreter_args, parse_sample_line, sample_request,
};
use ft_core::ExecutionMode;

/// Which dense-linalg op both arms run — `frankentorch-linalg-live-torch-arm`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LinalgOp {
    Svd,
    Svdvals,
    Eigh,
    Eigvalsh,
    Qr,
    /// Householder QR FACTORISATION only — no Q formed. Isolates panel + trailing GEMM.
    Geqrf,
    /// FORM Q from packed reflectors. The geqrf producing them runs outside the timer.
    Orgqr,
    /// APPLY packed reflectors to a matrix (Q^T C). geqrf runs outside the timer.
    Ormqr,
    /// Cholesky of an SPD matrix. Unique factorisation, so parity is unambiguous.
    Cholesky,
    /// slogdet — computed via LU, so it prices the LU path with a SCALAR checksum that is
    /// invariant to pivot order. Chosen over plain `det` because the SPD fixture's
    /// determinant (~n^n) is not representable in f64 at n=512.
    ///
    /// This once also said `lu_factor` was unusable here because row permutations can
    /// legitimately differ between implementations. That is true in general but NOT for
    /// this fixture: `_spd` is strictly diagonally dominant, so partial pivoting takes the
    /// diagonal at every step on both sides and no swap ever happens. `LuFactor` below now
    /// measures the factor directly.
    Slogdet,
    /// inv — LU-backed like slogdet, but with an O(n^3) getri tail instead of slogdet's
    /// O(n) diagonal log-product. Included to separate two readings of slogdet's 21-25x:
    /// either the whole LU family sits there (getrf is the cost), or slogdet's scalar tail
    /// is implicated. The SPD fixture is used for CONDITIONING only — torch does not know
    /// the matrix is SPD and still routes through getrf/getri, so the LU path is preserved.
    Inv,
    /// lu_factor — getrf and nothing else, so it measures our LU factorisation against
    /// torch's with no tail on either side. Included to TEST the claim that post-fix
    /// slogdet's residual ~10.8x IS our bare getrf gap: if that holds, lu_factor must land
    /// near 10.8x too. Its no-grad path already takes both outputs off one kernel call, so
    /// it is uncontaminated by the redundancy just fixed in slogdet.
    ///
    /// The SPD fixture is strictly diagonally dominant, so partial pivoting selects the
    /// diagonal at every step in BOTH implementations — no row swaps, LU is unique, and
    /// |sum| is directly comparable rather than pivot-order noise.
    LuFactor,
    /// matrix_exp — scaling-and-squaring + Pade, so almost entirely GEMM. Included to test
    /// whether the board's GEMM-bound wins (1.06-1.70x FASTER) carry to a linalg op.
    MatrixExp,
}

/// One measured configuration of our own arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Arm {
    gate: u64,
    blocked: bool,
    /// Whether the panel's two trailing updates run as ONE pass over `A22`
    /// (`frankentorch-4zjaa`, NEGATIVE_EVIDENCE item 247b). Bit-identical either way, so this
    /// pair can move time and cannot move a number — `FT_FUSED=1,0` puts both halves in one
    /// invocation against one live incumbent.
    fused: bool,
    /// The serial dlabrd steps keep four independent outputs in flight. This
    /// preserves each output's reduction order exactly (`frankentorch-75e38`).
    panel_output_blocked: bool,
    /// Run the VALUES-ONLY entry (`tensor_linalg_svdvals`) instead of the full
    /// `(U, S, Vh)` decomposition — `frankentorch-bidiag-form-q-unblocked-gl0rj`.
    ///
    /// WHY THIS ARM EXISTS. `SVD_FORM_PQ_NS` wraps `form_p` AND `form_q` in one
    /// counter, so the expansion phase cannot be sized from it — and that counter
    /// is separately untrustworthy (item 258c had it summing to 2.3x the measured
    /// wall time; a sweep of n=120..140 showed it non-monotonic, reading LOWER at
    /// n=124 than at n=120). The n>=130 `form_p` gate turned out to be a no-op, so
    /// the dispatch discontinuity cannot separate them either.
    ///
    /// `svd_blocked_bidiag_values` materialises NO reflectors, so the values-only
    /// path skips `form_p` and `form_q` entirely. `full - values` is therefore the
    /// whole expansion phase measured at the LANE level, min-of-N on both sides,
    /// same estimator — which is exactly what the phase counters are not.
    ///
    /// It is an ARM rather than a separate invocation on purpose: a cross-run
    /// subtraction on this host is worthless (the incumbent has moved 1.94x
    /// between two runs of the same ELF), so both halves must be interleaved
    /// round-by-round inside one process against one live incumbent.
    values_only: bool,
    /// Whether the U/V rotation replay transposes its row-block first — `frankentorch-i040z`,
    /// shipped in `8e077e39`. `FT_REPLAY=1,0` puts both halves in ONE invocation against one
    /// live incumbent.
    ///
    /// WHY IT NEEDS TO BE AN ARM. The change is bit-exact (same arithmetic, same order per
    /// element, only the layout it is read from differs), so it can move time and cannot move
    /// a number — which is exactly the shape that a two-binary A/B measures badly on this host
    /// and an interleaved arm measures well. It was landed on an INSTRUCTION count (1.767x
    /// fewer) and has never been priced in wall time; this is how it gets priced or reverted.
    ///
    /// It is only visible on a spectrum that actually exercises the sweep: on the default
    /// fixture the QR sweep is 0% of the lane, so this arm would read as a null there by
    /// construction. Run it with `FT_FIXTURE=generic`.
    replay_transposed: bool,
    /// Whether the tred2 reduction's `ggs` matvec keeps FOUR outputs in flight
    /// (`frankentorch-wjrqt`). Bit-identical to the per-`j` loop, so this pair can move time
    /// and cannot move a number — `FT_GGS=1,0` prices it in ONE invocation against one live
    /// incumbent. Only the eigh/eigvalsh lanes touch it.
    grouped_ggs: bool,
    /// Force the blocked trailing-update GEMM's column blocks to run SEQUENTIALLY
    /// (`frankentorch-rpytm`). Bit-identical either way, so this pair can move time and cannot
    /// move a number. `FT_SUBSER=0,1` prices lu_factor's only parallelism in ONE invocation.
    sub_serial: bool,
    /// Forced column-block WIDTH for the trailing-update GEMM; `0` = the thread-derived
    /// default (`frankentorch-rpytm`). `block_cols` divides by `rayon::current_num_threads()`,
    /// so width and thread count move together by default and a thread sweep cannot separate
    /// them — this arm varies width alone. Bit-identical either way.
    sub_cols: usize,
}

fn arm_label(arm: Arm) -> String {
    let gate = if arm.gate == u64::MAX {
        "SERIAL".to_string()
    } else {
        format!("{}", arm.gate)
    };
    format!(
        "{gate}/{}/{}/{}{}",
        if arm.blocked { "4row" } else { "1row" },
        if arm.fused { "fused" } else { "2pass" },
        if arm.panel_output_blocked {
            "4output"
        } else {
            "1output"
        },
        if arm.values_only { "/VALUES-ONLY" } else { "" }
    ) + if arm.replay_transposed {
        "/replay-T"
    } else {
        "/replay-rowmajor"
    } + if arm.grouped_ggs { "/ggs4" } else { "/ggs1" }
        + if arm.sub_serial { "/subSER" } else { "/subPAR" }
        + &(if arm.sub_cols > 0 { format!("/cols{}", arm.sub_cols) } else { "/colsAUTO".to_string() })
}

/// Deterministic and diagonally dominant, built by the SAME closed form on both arms so the
/// singular-value checksum is a real parity check rather than a coincidence of shapes.
fn fill(n: usize) -> Vec<f64> {
    if generic_fixture() {
        return fill_generic(n);
    }
    let mut a = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            let v = ((((r + 2) * (c + 3)) % 17) as f64 - 8.0) * 0.05;
            a[r * n + c] = v + if r == c { 3.0 } else { 0.0 };
        }
    }
    a
}

/// `FT_FIXTURE=generic` selects [`fill_generic`] over the default closed form.
///
/// OPT-IN, deliberately. Every banked row on this harness was taken on the default fixture,
/// so switching it silently would make new rows incomparable with the ledger while looking
/// identical. The chosen fixture is printed in the header for exactly that reason.
fn generic_fixture() -> bool {
    std::env::var("FT_FIXTURE")
        .map(|v| v == "generic")
        .unwrap_or(false)
}

/// A fixture with a GENERIC singular-value spectrum — `frankentorch-gqmws`.
///
/// WHY THIS EXISTS. The default fixture is `3*I` plus a low-rank modular term, and at n=512
/// **495 of its 512 singular values are exactly 3.0**, with 18 distinct values in the whole
/// spectrum. The implicit-shift bidiagonal QR sweep terminates by DEFLATION, so a spectrum
/// that degenerate deflates immediately and the sweep does essentially no work: it times at
/// 0.255 ms there against 24.291 ms on a generic matrix through the SAME counter, same code,
/// same n, same thread count. The lane therefore cannot see that phase at all, and any
/// "phase X is/is not the bottleneck" conclusion drawn from it is a property of the fixture.
///
/// BIT-IDENTICAL ON BOTH ARMS, which is the whole point and the thing that is easy to get
/// wrong: a fixture asymmetry between the Rust and Python sides has already voided one whole
/// lane in this campaign. Every operation here is integer until the final scale, and the
/// scale is a power of two, so both arms produce the same f64 bits rather than merely the
/// same values to some tolerance. `_mk_generic` in the Python setup is the same closed form
/// in the same order.
///
/// It is NOT an RNG: same bits on every run, every host and every build.
fn fill_generic(n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            let h = (r * 73 + c * 151 + (r * c) % 257) % 2048;
            // /2048, -1.0 and +16.0 are all exact; h is an integer below 2^11, so the result
            // is an exact f64 and no rounding separates the arms.
            //
            // The +16 diagonal is CONDITIONING, not spectrum shaping. Without it the matrix
            // measures cond 2.7e6 at n=512 (smin 9.3e-5), and there the smallest singular
            // values differ between Golub-Reinsch and LAPACK gesdd by more than the lane's
            // parity tolerance — which would report MISMATCH for a difference that is
            // algorithmic, not a defect. With it: cond 97.4 at n=512, 25.2 at n=256, and the
            // spectrum stays fully non-degenerate (512 of 512 distinct, against the default
            // fixture's 495-fold repeat of a single value).
            a[r * n + c] = (h as f64) / 2048.0 - 1.0 + if r == c { 16.0 } else { 0.0 };
        }
    }
    a
}

/// `(A + A^T) / 2`, for the eig lanes only — `frankentorch-linalg-live-torch-arm`.
///
/// `eigh` reads a single triangle. Handing it the non-symmetric `fill` would make each arm
/// answer a question about a DIFFERENT matrix, and the parity checksum would then be comparing
/// two correct answers to two different problems rather than catching a defect. The incumbent's
/// `_mk(n, sym=True)` performs the identical symmetrisation in the identical order, so both arms
/// see the same bits.
fn symmetrise(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            out[r * n + c] = (a[r * n + c] + a[c * n + r]) * 0.5;
        }
    }
    out
}

/// Cumulative iowait jiffies from `/proc/stat`'s aggregate `cpu` line.
fn iowait_jiffies() -> u64 {
    std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|text| {
            let line = text.lines().next()?;
            line.split_whitespace().nth(5)?.parse::<u64>().ok()
        })
        .unwrap_or(0)
}

fn provenance() -> (f64, f64) {
    let load = ft_api::harness_provenance::load_average_1m().unwrap_or(f64::NAN);
    let mhz = ft_api::harness_provenance::cpu_mhz_stats()
        .map_or(f64::NAN, |(_min, median, _max, _spread)| median);
    (load, mhz)
}

/// One SVD forward under `arm`, in milliseconds, plus the singular-value sum.
///
/// The timer stops before the checksum is read, exactly as the incumbent's `run` does, so both
/// arms time the same region.
fn ft_one(n: usize, data: &[f64], arm: Arm, op: LinalgOp) -> (f64, f64) {
    let previous_gate = ft_kernel_cpu::bidiag_parallel_gate_set(arm.gate);
    let previous_rowdot = ft_kernel_cpu::bidiag_rowdot_blocked_set(arm.blocked);
    let previous_fused = ft_kernel_cpu::bidiag_fused_trailing_set(arm.fused);
    let previous_panel = ft_kernel_cpu::bidiag_panel_output_blocked_set(arm.panel_output_blocked);
    let previous_replay = ft_kernel_cpu::set_svd_replay_transposed(arm.replay_transposed);
    let previous_ggs = ft_kernel_cpu::set_tred2_grouped_ggs(arm.grouped_ggs);
    let previous_subser = ft_kernel_cpu::set_dgemm_sub_serial(arm.sub_serial);
    let previous_subcols = ft_kernel_cpu::set_dgemm_sub_block_cols(arm.sub_cols);
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(data.to_vec(), vec![n, n], false)
        .expect("svd leaf");
    // ORGQR IS TIMED SEPARATELY, because its input must be produced OUTSIDE the clock.
    // `tensor_geqrf` is itself a measured 226x defect; timing it here would drown the very
    // thing this lane exists to isolate. The incumbent does the same -- its geqrf runs at
    // LANES construction time.
    if op == LinalgOp::Ormqr {
        let (packed, tau) = session.tensor_geqrf(x).expect("geqrf for ormqr");
        let c = session
            .tensor_variable(data.to_vec(), vec![n, n], false)
            .expect("ormqr C");
        let started = Instant::now();
        let out = session
            .tensor_ormqr(packed, tau, c, true, false)
            .expect("ormqr");
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let sum: f64 = session
            .tensor_values(out)
            .expect("ormqr values")
            .iter()
            .map(|v| v.abs())
            .sum();
        ft_kernel_cpu::bidiag_parallel_gate_set(previous_gate);
        ft_kernel_cpu::bidiag_rowdot_blocked_set(previous_rowdot);
        ft_kernel_cpu::bidiag_fused_trailing_set(previous_fused);
        ft_kernel_cpu::bidiag_panel_output_blocked_set(previous_panel);
        ft_kernel_cpu::set_svd_replay_transposed(previous_replay);
        ft_kernel_cpu::set_tred2_grouped_ggs(previous_ggs);
        ft_kernel_cpu::set_dgemm_sub_serial(previous_subser);
        ft_kernel_cpu::set_dgemm_sub_block_cols(previous_subcols);
        return (ms, sum);
    }
    if op == LinalgOp::Orgqr {
        let (packed, tau) = session.tensor_geqrf(x).expect("geqrf for orgqr");
        let started = Instant::now();
        let q = session.tensor_orgqr(packed, tau).expect("orgqr");
        let ms = started.elapsed().as_secs_f64() * 1e3;
        // |Q|, matching the incumbent: Q is unique only up to COLUMN SIGNS, so a raw sum
        // would mismatch on a convention difference rather than on a defect.
        let sum: f64 = session
            .tensor_values(q)
            .expect("q values")
            .iter()
            .map(|v| v.abs())
            .sum();
        ft_kernel_cpu::bidiag_parallel_gate_set(previous_gate);
        ft_kernel_cpu::bidiag_rowdot_blocked_set(previous_rowdot);
        ft_kernel_cpu::bidiag_fused_trailing_set(previous_fused);
        ft_kernel_cpu::bidiag_panel_output_blocked_set(previous_panel);
        ft_kernel_cpu::set_svd_replay_transposed(previous_replay);
        ft_kernel_cpu::set_tred2_grouped_ggs(previous_ggs);
        ft_kernel_cpu::set_dgemm_sub_serial(previous_subser);
        ft_kernel_cpu::set_dgemm_sub_block_cols(previous_subcols);
        return (ms, sum);
    }
    let started = Instant::now();
    // Both branches time exactly the decomposition and stop before the checksum is read, which
    // is what the incumbent's `run` does too. The values-only branch reaches
    // `svd_blocked_bidiag_values`, which materialises NO reflectors -- so `full - values` is the
    // form_p + form_q expansion phase, at the lane level, same estimator on both sides.
    let sv = if arm.values_only {
        session.tensor_linalg_svdvals(x).expect("svdvals")
    } else {
        match op {
            // Each branch returns the SAME quantity the incumbent's lambda returns, so the parity
            // checksum compares like with like: eigenvalues for the eig lanes, |diag(R)| for qr
            // (R's diagonal is sign-ambiguous between implementations, the magnitudes are not).
            LinalgOp::Svd => {
                let (_u, sv, _vh) = session.tensor_linalg_svd(x, true).expect("svd");
                sv
            }
            LinalgOp::Svdvals => session.tensor_linalg_svdvals(x).expect("svdvals"),
            LinalgOp::Eigh => {
                let (w, _v) = session.tensor_linalg_eigh(x).expect("eigh");
                w
            }
            LinalgOp::Eigvalsh => session.tensor_linalg_eigvalsh(x).expect("eigvalsh"),
            LinalgOp::MatrixExp => {
                let e = session.tensor_matrix_exp(x).expect("matrix_exp");
                let a = session.tensor_abs(e).expect("abs");
                session.tensor_sum(a).expect("sum")
            }
            LinalgOp::Slogdet => {
                let (_sign, logabsdet) = session.tensor_linalg_slogdet(x).expect("slogdet");
                session.tensor_abs(logabsdet).expect("abs")
            }
            LinalgOp::LuFactor => {
                // Pivots come back as a plain Vec<usize>, so only the LU factor is timed.
                let (lu, _pivots) = session.tensor_lu_factor(x).expect("lu_factor");
                let a = session.tensor_abs(lu).expect("abs");
                session.tensor_sum(a).expect("sum")
            }
            LinalgOp::Inv => {
                // inv's result is unique (no sign, pivot or basis freedom), so |sum|
                // genuinely discriminates rather than decorating.
                let a_inv = session.tensor_linalg_inv(x).expect("inv");
                let a = session.tensor_abs(a_inv).expect("abs");
                session.tensor_sum(a).expect("sum")
            }
            LinalgOp::Cholesky => {
                let l = session.tensor_linalg_cholesky(x, false).expect("cholesky");
                let d = session.tensor_diagonal(l, 0).expect("diag");
                session.tensor_abs(d).expect("abs")
            }
            LinalgOp::Qr => {
                let (_q, r) = session.tensor_linalg_qr(x, false).expect("qr");
                let d = session.tensor_diagonal(r, 0).expect("diag");
                session.tensor_abs(d).expect("abs")
            }
            LinalgOp::Orgqr | LinalgOp::Ormqr => unreachable!("timed in their own branches above"),
            LinalgOp::Geqrf => {
                // geqrf overwrites the upper triangle with R and the lower with the
                // packed reflectors; |diag| is R's diagonal, the same quantity the qr
                // lane checksums and the only part comparable across implementations.
                let (a, _tau) = session.tensor_geqrf(x).expect("geqrf");
                let d = session.tensor_diagonal(a, 0).expect("diag");
                session.tensor_abs(d).expect("abs")
            }
        }
    };
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let sum: f64 = session
        .tensor_values(sv)
        .expect("singular values")
        .iter()
        .sum();
    ft_kernel_cpu::bidiag_parallel_gate_set(previous_gate);
    ft_kernel_cpu::bidiag_rowdot_blocked_set(previous_rowdot);
    ft_kernel_cpu::bidiag_fused_trailing_set(previous_fused);
    ft_kernel_cpu::bidiag_panel_output_blocked_set(previous_panel);
    ft_kernel_cpu::set_svd_replay_transposed(previous_replay);
    ft_kernel_cpu::set_tred2_grouped_ggs(previous_ggs);
    ft_kernel_cpu::set_dgemm_sub_serial(previous_subser);
    ft_kernel_cpu::set_dgemm_sub_block_cols(previous_subcols);
    (ms, sum)
}

/// Ask the incumbent co-process for exactly one timed sample of `lane`.
///
/// A closed stdout is a hard failure rather than a skipped sample: a silently short incumbent arm
/// would leave the remaining rounds measuring only our side, which is the one failure mode a
/// vs-incumbent lane may not have.
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
                "the PyTorch co-process closed its stdout while `{lane}` was being sampled; a \
                 partially measured arm cannot carry a vs-PyTorch claim"
            )
            .into());
        }
        if let Some(sample) = parse_sample_line(&line) {
            assert_eq!(sample.lane, lane, "co-process answered for the wrong lane");
            return Ok((sample.milliseconds, sample.gradient_checksum));
        }
    }
}

/// Median of `v`, which is sorted in place. `NaN` for an empty slice.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn ratio_label(ratio: f64) -> String {
    if ratio >= 1.0 {
        format!("FT {ratio:.3}x FASTER")
    } else {
        format!("FT {:.3}x SLOWER", 1.0 / ratio)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let sizes: Vec<usize> = std::env::var("FT_GATE_SIZES")
        .unwrap_or_else(|_| "128,136,256,512".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let gate_values: Vec<u64> = std::env::var("FT_GATE_VALUES")
        .unwrap_or_else(|_| format!("262144,{}", u64::MAX))
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let rowdots: Vec<bool> = std::env::var("FT_ROWDOT")
        .unwrap_or_else(|_| "1".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    // `frankentorch-4zjaa` item 247b's lever, as a paired arm. Default "1" so an existing command
    // keeps measuring exactly what it measured before; `FT_FUSED=1,0` runs both halves in ONE
    // invocation against ONE live incumbent, which is the only form item 25 admits.
    let fuseds: Vec<bool> = std::env::var("FT_FUSED")
        .unwrap_or_else(|_| "1".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    // `frankentorch-i040z`'s lever as a paired arm. Default "1" (the shipped path) so an
    // existing command measures exactly what it measured before; `FT_REPLAY=1,0` prices it
    // against the row-major form in ONE invocation against ONE live incumbent. Pair it with
    // FT_FIXTURE=generic — on the default fixture the QR sweep is 0% of the lane, so this arm
    // is a null there by construction and would "prove" the lever worthless for the wrong
    // reason.
    let subcols: Vec<usize> = std::env::var("FT_SUBCOLS")
        .unwrap_or_else(|_| "0".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let subsers: Vec<bool> = std::env::var("FT_SUBSER")
        .unwrap_or_else(|_| "0".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    let ggs_arms: Vec<bool> = std::env::var("FT_GGS")
        .unwrap_or_else(|_| "0".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    let replays: Vec<bool> = std::env::var("FT_REPLAY")
        .unwrap_or_else(|_| "1".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    let panel_outputs: Vec<bool> = std::env::var("FT_PANEL_OUTPUT")
        .unwrap_or_else(|_| "1".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    assert!(
        !sizes.is_empty()
            && !gate_values.is_empty()
            && !rowdots.is_empty()
            && !fuseds.is_empty()
            && !panel_outputs.is_empty(),
        "empty grid"
    );
    let mut arms: Vec<Arm> =
        Vec::with_capacity(gate_values.len() * rowdots.len() * fuseds.len() * panel_outputs.len());
    for &gate in &gate_values {
        for &blocked in &rowdots {
            for &fused in &fuseds {
                for &panel_output_blocked in &panel_outputs {
                    for &replay_transposed in &replays {
                        for &grouped_ggs in &ggs_arms {
                            for &sub_serial in &subsers {
                                for &sub_cols in &subcols {
                                    arms.push(Arm {
                                        gate,
                                        blocked,
                                        fused,
                                        panel_output_blocked,
                                        values_only: false,
                                        replay_transposed,
                                        grouped_ggs,
                                        sub_serial,
                                        sub_cols,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // FT_VALUES_ARM=1 appends ONE extra arm running the values-only entry, with the shipped
    // configuration otherwise identical to `arms[0]`. `arms[0] - this` is the expansion phase
    // (form_p + form_q), measured at the lane level rather than read off the untrustworthy
    // SVD_FORM_PQ_NS counter -- see the `values_only` field for why that counter cannot be used.
    //
    // Appended LAST so every existing arm keeps its index, which matters because `paired-vs-arm0`
    // is reported against `arms[0]` and every banked row in this campaign is indexed that way.
    //
    // Its PARITY COLUMN IS THE POINT, not a formality: svdvals and svd must produce the SAME
    // singular values, so the checksum comparison is a live check that the values-only path is
    // the same decomposition and not a different (or truncated) one. A MISMATCH here means the
    // subtraction is comparing two different computations and the phase figure is meaningless.
    if std::env::var("FT_VALUES_ARM").is_ok_and(|v| v == "1") {
        let mut values_arm = arms[0];
        values_arm.values_only = true;
        arms.push(values_arm);
    }
    let rounds: usize = std::env::var("FT_ROUNDS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(9);
    assert!(rounds >= 1, "FT_ROUNDS must be at least 1");
    // Read by BOTH arms, and matched deliberately: an asymmetric warmup has a bias whose
    // direction depends on which arm is faster, which is a property no instrument may have.
    // Eight rather than the board's 32 because a single n=1024 SVD is ~0.5 s here, and 32 would
    // spend minutes per size before the first sample.
    let warmup: usize = std::env::var("FT_H2H_WARMUP")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(8);
    // The co-process reads the same count from its OWN environment, handed to it on the
    // `Command` below rather than through `std::env::set_var` — this crate forbids `unsafe`, and
    // mutating the parent's environment to talk to a child is the wrong mechanism anyway.

    // FT_OP selects WHICH dense-linalg op both arms run — `frankentorch-linalg-live-torch-arm`.
    //
    // WHY THIS EXISTS. Every standing this session banked used a torch CO-PROCESS inside the same
    // invocation, because this host has moved the incumbent 1.94x between two runs of the same
    // ELF. The eig/qr family had no such harness at all: `linalg_gap_sweep` is FT-INTERNAL and
    // its own header says torch is run separately, i.e. a cross-run comparison. So "SVD is the
    // worst op in the tree" could only ever be claimed as "the worst op among ops that HAVE a
    // live-torch harness". This closes that gap by reusing the machinery already here —
    // provenance, the A/A null via repeated arms, per-round interleaving, the parity checksum.
    //
    // The gate/rowdot/fused/panel arms are SVD-reduction knobs and do nothing for eigh/qr, so on
    // those ops the arms differ only by position: a repeated arm is then a PURE A/A null, which
    // is exactly what is wanted for a first standing.
    let op = std::env::var("FT_OP").unwrap_or_else(|_| "svd".to_owned());
    // SYMMETRY IS NOT OPTIONAL for the eig lanes. `_mk` builds a NON-symmetric matrix, and
    // `torch.linalg.eigh` reads only one triangle — it would silently return the spectrum of a
    // DIFFERENT matrix than our arm sees, and the parity checksum would compare two answers to
    // two different questions. `(A + A.T) * 0.5` on both sides is what makes the row mean
    // anything.
    let ft_op = match op.as_str() {
        "svd" => LinalgOp::Svd,
        "svdvals" => LinalgOp::Svdvals,
        "eigh" => LinalgOp::Eigh,
        "eigvalsh" => LinalgOp::Eigvalsh,
        "qr" => LinalgOp::Qr,
        "geqrf" => LinalgOp::Geqrf,
        "orgqr" => LinalgOp::Orgqr,
        "ormqr" => LinalgOp::Ormqr,
        "cholesky" => LinalgOp::Cholesky,
        "slogdet" => LinalgOp::Slogdet,
        "matrix_exp" => LinalgOp::MatrixExp,
        "inv" => LinalgOp::Inv,
        "lu_factor" => LinalgOp::LuFactor,
        other => panic!(
            "FT_OP={other:?} is not one of svd|svdvals|eigh|eigvalsh|qr|geqrf|orgqr|ormqr|cholesky|slogdet|matrix_exp|inv"
        ),
    };
    let (py_fn, sym) = match op.as_str() {
        "svd" => ("torch.linalg.svd(A)[1]", false),
        "svdvals" => ("torch.linalg.svdvals(A)", false),
        "eigh" => ("torch.linalg.eigh(A)[0]", true),
        "eigvalsh" => ("torch.linalg.eigvalsh(A)", true),
        "qr" => ("torch.linalg.qr(A)[1].diagonal().abs()", false),
        // geqrf is the FACTORISATION ALONE -- panel + trailing GEMM, with NO Q formed.
        // `qr - geqrf` therefore isolates the Q-formation half, and geqrf on its own
        // prices the panel/BLAS-2 + GEMM phase that the GEMM refutation (ee024f6e) and the
        // blocking refutation (318cd457) both pointed at without being able to measure.
        // Checksum is |diag(A)| on both arms: geqrf overwrites A's upper triangle with R,
        // so its diagonal is R's, matching the qr lane's convention and comparable across
        // implementations where the raw reflectors are not.
        "geqrf" => ("torch.geqrf(A)[0].diagonal().abs()", false),
        // orgqr: FORM Q from packed reflectors. The geqrf that produces (A, tau) runs at LANES
        // construction time, OUTSIDE the timed region, on both arms -- otherwise this would
        // re-measure geqrf's 226x defect instead of orgqr.
        //
        // CHECKSUM IS |Q|, NOT Q. Each arm forms Q from its OWN geqrf output, and the two
        // factorisations need not agree on reflector sign convention. For a full-rank matrix the
        // QR factor Q is unique only up to COLUMN SIGNS, so a raw sum would mismatch on a
        // convention difference rather than on a defect. Summing |Q| is invariant under column
        // sign flips and still catches a genuinely wrong Q.
        "orgqr" => ("torch.linalg.householder_product(A[0], A[1]).abs()", false),
        // ormqr: APPLY the packed reflectors to a matrix C (Q^T C, left, no transpose).
        // Like orgqr, the geqrf producing (A, tau) runs at LANES construction time, OUTSIDE
        // the timed region, on both arms -- otherwise this re-measures geqrf's 227x defect.
        // |result| for the same reason orgqr uses |Q|: the two factorisations need not agree
        // on reflector sign convention, and a sign difference is not a defect.
        "ormqr" => ("torch.ormqr(A[0], A[1], A[2]).abs()", false),
        // cholesky: the factorisation is UNIQUE for an SPD matrix, so diag(L) is directly
        // comparable across implementations -- no sign or pivot freedom to explain away a
        // mismatch. That is why this op was chosen over ldl_factor, whose Bunch-Kaufman
        // pivoting may legitimately differ and would make the parity column unreadable.
        "cholesky" => ("torch.linalg.cholesky(A).diagonal().abs()", false),
        // slogdet: exercises the LU path with a SCALAR, pivot-order-invariant checksum.
        // See the Rust arm for why this was chosen over lu_factor and over plain det.
        "slogdet" => ("torch.linalg.slogdet(A)[1].abs().reshape(1)", false),
        // inv: LU-backed with an O(n^3) getri tail. See the Rust arm for why the SPD
        // fixture is used here (conditioning) without diverting off the LU route.
        "inv" => ("torch.linalg.inv(A).abs().sum().reshape(1)", false),
        // lu_factor: bare getrf on both sides. SPD fixture is diagonally dominant so
        // partial pivoting takes the diagonal and LU is unique. See the Rust arm.
        "lu_factor" => ("torch.linalg.lu_factor(A)[0].abs().sum().reshape(1)", false),
        // matrix_exp: GEMM-dominated (scaling-squaring + Pade), unique result. See the
        // Rust arm for why the fixture is scaled by 1/n.
        "matrix_exp" => ("torch.linalg.matrix_exp(A).abs().sum().reshape(1)", false),
        other => panic!(
            "FT_OP={other:?} is not one of svd|svdvals|eigh|eigvalsh|qr|geqrf|orgqr|ormqr|cholesky|slogdet|matrix_exp"
        ),
    };
    let lanes: Vec<(usize, String)> = sizes.iter().map(|&n| (n, format!("{op}_{n}"))).collect();
    let lane_entries: Vec<String> = lanes
        .iter()
        .map(|(n, name)| {
            let base = if op == "orgqr" {
                format!("torch.geqrf(_mk({n}, False))")
            } else if op == "cholesky" || op == "slogdet" || op == "inv" || op == "lu_factor" {
                format!("_spd({n})")
            } else if op == "matrix_exp" {
                format!("_expm_fixture({n})")
            } else if op == "ormqr" {
                // (A, tau, C) as a flat 3-tuple: geqrf's two outputs plus the matrix to apply to.
                format!(
                    "(lambda g, c: (g[0], g[1], c))(torch.geqrf(_mk({n}, False)), _mk({n}, False))"
                )
            } else if generic_fixture() {
                // Mirrors the Rust arm's fill()/fill_generic() switch. Both selectors are
                // driven by the SAME predicate so the two arms cannot diverge on which matrix
                // they factor — an asymmetry here voided a whole lane earlier in this campaign
                // and every other gate still passed while it did.
                format!("_mk_generic({n}, {})", if sym { "True" } else { "False" })
            } else {
                format!("_mk({n}, {})", if sym { "True" } else { "False" })
            };
            format!("    \"{name}\": ({base}, lambda A: {py_fn}),")
        })
        .collect();
    let py_setup = format!(
        r#"
import time, torch
torch.set_num_threads(8)
def _mk(n, sym=False):
    r = torch.arange(n, dtype=torch.float64).reshape(n, 1)
    c = torch.arange(n, dtype=torch.float64).reshape(1, n)
    A = ((((r + 2) * (c + 3)) % 17) - 8.0) * 0.05 + torch.eye(n, dtype=torch.float64) * 3.0
    # Symmetrised for the eig lanes, matching the Rust arm exactly. eigh reads one triangle, so
    # without this the two arms would be answering different questions and the parity column
    # would be meaningless rather than merely failing.
    return (A + A.T) * 0.5 if sym else A
def _mk_generic(n, sym=False):
    # frankentorch-gqmws. Generic singular-value spectrum, against _mk's 495-of-512 degeneracy.
    # Integer arithmetic throughout and a power-of-two scale, so this is BIT-IDENTICAL to the
    # Rust arm's fill_generic rather than merely close: int64 ops are exact on both sides and
    # h/2048 - 1.0 introduces no rounding. Same closed form, same order.
    r = torch.arange(n, dtype=torch.int64).reshape(n, 1)
    c = torch.arange(n, dtype=torch.int64).reshape(1, n)
    h = (r * 73 + c * 151 + (r * c) % 257) % 2048
    # +16 on the diagonal is CONDITIONING: without it cond is 2.7e6 at n=512 and the smallest
    # singular values differ between Golub-Reinsch and gesdd by more than the parity tolerance,
    # which would report MISMATCH for an algorithmic difference rather than a defect. With it,
    # cond 97.4 at n=512 and the spectrum is still fully non-degenerate (512/512 distinct).
    A = h.to(torch.float64) / 2048.0 - 1.0 + torch.eye(n, dtype=torch.float64) * 16.0
    return (A + A.T) * 0.5 if sym else A
def _expm_fixture(n):
    # Scaled by 1/n so the spectral radius stays O(1): expm of the raw fixture overflows
    # f64 well before n=512. The Rust arm applies the identical scaling.
    return _mk(n, False) / float(n)
def _spd(n):
    # Symmetric + strictly diagonally dominant => positive definite at every n, so the
    # Cholesky factor exists and is unique. The Rust arm builds the identical matrix.
    A = _mk(n, True)
    return A + torch.eye(n, dtype=torch.float64) * float(n)
def run(base, fn):
    _t = time.perf_counter()
    _s = fn(base)
    _ms = (time.perf_counter() - _t) * 1e3
    return _ms, float(_s.sum())
LANES = {{
{}
}}
print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)
print('PT_THREADS %d' % torch.get_num_threads(), flush=True)
"#,
        lane_entries.join("\n")
    );
    let py = format!("{py_setup}{SAMPLE_LOOP_PY}");

    println!(
        "measurement=SVD FORWARD ONLY (full U,S,Vh); estimator=min over {rounds} rounds, every \
         arm AND the incumbent sampled once per round, arm order reversed on odd rounds, first \
         round discarded; every ratio is the median of the PAIRED per-round ratio"
    );
    println!(
        "elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "rayon_threads={} warmup={warmup} (both arms)  default_gate={}",
        rayon::current_num_threads(),
        ft_kernel_cpu::bidiag_parallel_gate()
    );
    // Provenance: a banked row that does not name its fixture is how a spectrally degenerate
    // input went unnoticed long enough to close a bead (frankentorch-gqmws). The default is
    // 3*I plus a low-rank modular term, whose spectrum has 495 of 512 values equal at n=512 —
    // the bidiagonal QR sweep deflates immediately on it and times ~95x lower than on a
    // generic matrix, so phase conclusions from it do not generalise.
    println!(
        "fixture={} (FT_FIXTURE=generic selects the generic-spectrum matrix)",
        if generic_fixture() {
            "generic (spread spectrum)"
        } else {
            "_mk DEFAULT: 3*I + low-rank; 495/512 singular values EQUAL at n=512 — QR sweep is invisible on it"
        }
    );
    println!(
        "arms (gate/step-12/trailing/panel-output; u64::MAX = always serial): {:?}",
        arms.iter().map(|a| arm_label(*a)).collect::<Vec<_>>()
    );
    println!(
        "null: repeat an arm in FT_GATE_VALUES — two identical arms differ only by this window's \
         noise, and no effect below that is readable"
    );

    // `-c`, never `-`: the latter reads the program from stdin until EOF, which deadlocks a
    // co-process whose stdin must stay open for requests.
    let mut child = Command::new(&python)
        .args(interpreter_args(&py))
        .env("FT_H2H_WARMUP", warmup.to_string())
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
    let mut stdin = child.stdin.take().expect("co-process stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("co-process stdout"));
    // Block until the arm has imported torch, built its tensors and warmed every lane.
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!(
                "the PyTorch arm exited before announcing `{READY_MARKER}`; a FrankenTorch-only \
                 run cannot carry a vs-PyTorch claim"
            )
            .into());
        }
        let trimmed = line.trim();
        if let Some(version) = trimmed.strip_prefix("PT_TORCH_VERSION ") {
            println!("incumbent=PyTorch {version} (self-reported, same invocation)");
        }
        if let Some(threads) = trimmed.strip_prefix("PT_THREADS ") {
            println!("incumbent threads={threads} (self-reported)");
        }
        if trimmed == READY_MARKER {
            break;
        }
    }

    for (n, lane) in &lanes {
        let n = *n;
        // Symmetrised for the eig lanes ONLY, matching the incumbent's `_mk(n, sym=True)`
        // exactly. eigh reads one triangle, so an unsymmetrised fixture would have the two arms
        // answering different questions.
        let data = if ft_op == LinalgOp::MatrixExp {
            // Identical to the incumbent's `_expm_fixture`: scale by 1/n so the spectral
            // radius stays O(1). Unscaled, expm overflows f64 long before n=512.
            let mut d = fill(n);
            for v in &mut d {
                *v /= n as f64;
            }
            d
        } else if ft_op == LinalgOp::Cholesky
            || ft_op == LinalgOp::Slogdet
            || ft_op == LinalgOp::Inv
            || ft_op == LinalgOp::LuFactor
        {
            // Identical to the incumbent's `_spd`: symmetrise, then add n to the diagonal.
            // Strictly diagonally dominant => positive definite at every n.
            let mut d = symmetrise(&fill(n), n);
            for i in 0..n {
                d[i * n + i] += n as f64;
            }
            d
        } else if sym {
            symmetrise(&fill(n), n)
        } else {
            fill(n)
        };
        let iowait_before = iowait_jiffies();
        let (load_before, mhz_before) = provenance();

        // Matched warmup, ours only: the co-process warmed its lanes before READY.
        for _ in 0..warmup {
            let _ = ft_one(n, &data, arms[0], ft_op);
        }

        let mut ft_ms: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); arms.len()];
        let mut ft_sum = vec![0.0f64; arms.len()];
        // ONE INCUMBENT SAMPLE PER ARM PER ROUND, not one per round. With six arms the earlier
        // design ran six of our SVDs against one of theirs, so the incumbent's caches were
        // disturbed six times as often as ours between its own samples — an asymmetry that
        // penalises the incumbent and grows with the number of arms we happen to be sweeping. A
        // measuring instrument whose reading depends on how many of OUR configurations are in the
        // grid is not measuring the incumbent. Each arm is now paired with the incumbent sample
        // that immediately followed it.
        let mut pt_ms: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); arms.len()];
        let mut pt_sum = 0.0f64;
        for round in 0..=rounds {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for &idx in &order {
                let (ms, sum) = ft_one(n, &data, arms[idx], ft_op);
                let (pt, pt_checksum) = incumbent_sample(&mut stdin, &mut reader, lane)?;
                if round > 0 {
                    ft_ms[idx].push(ms);
                    ft_sum[idx] = sum;
                    pt_ms[idx].push(pt);
                    pt_sum = pt_checksum;
                }
            }
        }
        let pt_all: Vec<f64> = pt_ms.iter().flatten().copied().collect();
        // One extra call per arm, untimed, with the counters cleared: the route AND the phase
        // split. Deliberately NOT part of the estimator — mixing an instrumented call into the
        // timed rounds would report a number nothing else in this campaign is comparable to.
        // THREE instrumented calls per arm, and the phase split is the per-component MEDIAN.
        // NEGATIVE_EVIDENCE item 258c read a single call whose components summed to 1058 ms
        // against a 464 ms median and suspected the counters of double-counting. They do not —
        // the blocked and NR prologues are mutually exclusive and each records once — it was one
        // contended sample. A single sample of anything on this host is not a measurement.
        const PHASE_CALLS: usize = 3;
        let mut branches = vec![(0u64, 0u64, 0u64); arms.len()];
        let mut phases = vec![(0u64, 0u64, 0u64); arms.len()];
        for (idx, &arm) in arms.iter().enumerate() {
            let mut samples: Vec<(u64, u64, u64)> = Vec::with_capacity(PHASE_CALLS);
            for _ in 0..PHASE_CALLS {
                let _ = ft_kernel_cpu::bidiag_parallel_branches_take();
                let _ = ft_kernel_cpu::svd_reduction_sweep_ns_take();
                let _ = ft_one(n, &data, arm, ft_op);
                branches[idx] = ft_kernel_cpu::bidiag_parallel_branches_take();
                samples.push(ft_kernel_cpu::svd_reduction_sweep_ns_take());
            }
            let mut reduction: Vec<u64> = samples.iter().map(|s| s.0).collect();
            let mut form_pq: Vec<u64> = samples.iter().map(|s| s.1).collect();
            let mut sweep: Vec<u64> = samples.iter().map(|s| s.2).collect();
            reduction.sort_unstable();
            form_pq.sort_unstable();
            sweep.sort_unstable();
            phases[idx] = (
                reduction[PHASE_CALLS / 2],
                form_pq[PHASE_CALLS / 2],
                sweep[PHASE_CALLS / 2],
            );
        }

        // EIGH PHASES, in the SAME invocation as the incumbent — `frankentorch-wjrqt`.
        //
        // The phase block above reads the SVD counters, which are zero on the eigh lanes, and
        // eigh has no live-call counters. So for eigh we profile the same matrix through
        // `eigh_stage_profile_f64` here, inside this process and this window, beside the
        // PyTorch figure printed below.
        //
        // WHY IT MATTERS. The claim it supports is that our REDUCTION PHASE ALONE costs several
        // times PyTorch's ENTIRE eigh. Sourcing the two halves of that ratio from different
        // windows makes it an anecdote on a host that has moved 1.94x between runs of one ELF;
        // sourcing them from one invocation makes it a measurement.
        //
        // This is a SEPARATE profiled call, not the timed one — the phase split is FT-internal
        // either way, and the timed arms above are what the vs-PT column is built from.
        if matches!(ft_op, LinalgOp::Eigh | LinalgOp::Eigvalsh) {
            let mut sym = vec![0.0f64; n * n];
            for r in 0..n {
                for c in 0..n {
                    sym[r * n + c] = (data[r * n + c] + data[c * n + r]) * 0.5;
                }
            }
            let _ = ft_kernel_cpu::eigh_stage_profile_f64(&sym, n); // warm
            let (reduce, back, tql2) = ft_kernel_cpu::eigh_stage_profile_f64(&sym, n);
            let ms = |v: u128| v as f64 / 1e6;
            println!(
                "eigh phases (same invocation, separate profiled call): reduce {:.3} ms  \
                 backtransform {:.3} ms  tql2 {:.3} ms",
                ms(reduce),
                ms(back),
                ms(tql2)
            );
        }

        let (load_after, mhz_after) = provenance();
        let pt_min = pt_all.iter().copied().fold(f64::INFINITY, f64::min);
        let pt_max = pt_all.iter().copied().fold(0.0f64, f64::max);
        println!();
        println!(
            "n={n}  PT min {pt_min:.3} ms  median {:.3} ms  spread {:.2}x  \
             load {load_before:.2}->{load_after:.2}  MHz {mhz_before:.0}/{mhz_after:.0}  \
             iowait {} jiffies",
            median(&mut pt_all.clone()),
            pt_max / pt_min,
            iowait_jiffies().saturating_sub(iowait_before)
        );
        for (idx, arm) in arms.iter().enumerate() {
            let min = ft_ms[idx].iter().copied().fold(f64::INFINITY, f64::min);
            let med = median(&mut ft_ms[idx].clone());
            let mut vs_pt: Vec<f64> = ft_ms[idx]
                .iter()
                .zip(pt_ms[idx].iter())
                .map(|(ours, theirs)| theirs / ours)
                .collect();
            let mut vs_arm0: Vec<f64> = ft_ms[idx]
                .iter()
                .zip(ft_ms[0].iter())
                .map(|(ours, reference)| reference / ours)
                .collect();
            let rel = (ft_sum[idx] - pt_sum).abs() / (pt_sum.abs() + 1e-300);
            println!(
                "  arm={:<16} min {min:8.3} ms  median {med:8.3} ms  PT-beside-it min \
                 {:8.3} ms  paired-vs-PT {}  paired-vs-arm0 {:.3}x  branches {:?}  \
                 parity rel {rel:.2e} {}",
                arm_label(*arm),
                pt_ms[idx].iter().copied().fold(f64::INFINITY, f64::min),
                ratio_label(median(&mut vs_pt)),
                median(&mut vs_arm0),
                branches[idx],
                if rel < 1e-9 { "MATCH" } else { "MISMATCH" }
            );
            let (reduction, form_pq, sweep) = phases[idx];
            let total = (reduction + form_pq + sweep).max(1) as f64;
            println!(
                "                     phases (ours only, median of 3 instrumented calls): reduction \
                 {:.3} ms {:.0}%  form_p/q {:.3} ms {:.0}%  QR sweep {:.3} ms {:.0}%",
                reduction as f64 / 1e6,
                100.0 * reduction as f64 / total,
                form_pq as f64 / 1e6,
                100.0 * form_pq as f64 / total,
                sweep as f64 / 1e6,
                100.0 * sweep as f64 / total
            );
        }
        // Same gate, different step-(12) kernel, MUST agree bit-for-bit: the four-row kernel
        // preserves each row's own summation order. An assertion rather than a print because it
        // is the only thing standing between an index bug in that kernel and a silently wrong
        // reduction at sizes the golden's small shapes never reach.
        for (i, a) in arms.iter().enumerate() {
            for (j, b) in arms.iter().enumerate() {
                if i < j && a.gate == b.gate && a.blocked != b.blocked {
                    assert_eq!(
                        ft_sum[i].to_bits(),
                        ft_sum[j].to_bits(),
                        "n={n}: {} and {} differ, but the step-(12) kernels are supposed to be \
                         bit-identical",
                        arm_label(*a),
                        arm_label(*b)
                    );
                }
            }
        }
    }

    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    let _ = child.wait();
    Ok(())
}
