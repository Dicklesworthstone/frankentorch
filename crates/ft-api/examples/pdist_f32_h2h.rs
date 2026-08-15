//! Live, balanced-square PyTorch comparison for the scored f32 no-grad pdist lane.
//!
//! The scorecard's `pdist_f32_p2_mm/512x64` route is deliberately no-grad.  It
//! therefore cannot share `gauntlet_lane_sweep_h2h`'s forward+loss+backward
//! timed region: asking the tape to backpropagate would select a different,
//! differentiable composition rather than the direct f32 kernel this benchmark
//! is meant to price.  This example keeps both arms to `forward` only and uses
//! the already-shared balanced-square request protocol.
//!
//! ```text
//! PYTORCH_PYTHON=/path/to/python \
//!   cargo run --profile release-perf -p ft-api --example pdist_f32_h2h
//! ```

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    BALANCED_SQUARE, MAX_NULL_CI_WIDTH, QUIT_REQUEST, READY_MARKER, adjudicate_null,
    parse_sample_line, sample_request,
};
use ft_core::ExecutionMode;

const N: usize = 512;
const M: usize = 64;
const DEFAULT_REPS: usize = 16;
const BOOTSTRAP_REPS: usize = 2_000;
const BALANCED_NULL_MAX_DEVIATION: f64 = 0.02;

fn reps() -> usize {
    std::env::var("FT_H2H_REPS")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .filter(|reps: &usize| *reps >= 8)
        .unwrap_or(DEFAULT_REPS)
}

fn seq(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 251) as f64 * 0.001 - 0.12) as f32)
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn paired_slot_median(mut values: [f64; 2]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[0] + values[1]) * 0.5
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

fn null_is_centered(point: f64) -> bool {
    point.is_finite() && (point - 1.0).abs() <= BALANCED_NULL_MAX_DEVIATION
}

/// Build the no-grad leaf outside the timer, then time only the direct p=2 route.
fn timed_pdist(values: &[f32], n: usize, m: usize) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable_f32(values.to_vec(), vec![n, m], false)
        .expect("pdist input");
    let started = Instant::now();
    let output = session.tensor_pdist(input, 2.0).expect("pdist");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = session
        .tensor_values_f32(output)
        .expect("pdist output")
        .iter()
        .map(|value| f64::from(*value))
        .sum();
    (elapsed, checksum)
}

fn incumbent_sample(
    stdin: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    writeln!(stdin, "{}", sample_request("pdist_f32"))?;
    stdin.flush()?;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("PyTorch co-process closed before its pdist sample".into());
        }
        if let Some(sample) = parse_sample_line(&line) {
            assert_eq!(sample.lane, "pdist_f32", "co-process returned wrong lane");
            return Ok((sample.milliseconds, sample.gradient_checksum));
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = seq(N * M);
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py_setup = r#"
import time, torch
print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)
torch.set_num_threads(8)
def seq(n):
    return ((torch.arange(n,dtype=torch.int64)%251).double()*0.001-0.12).float()
pdist=seq(512*64).reshape(512,64)
print('PT_TIMED_STEPS forward', flush=True)
def run(base, fn):
    s=time.perf_counter()
    out=fn(base)
    elapsed=(time.perf_counter()-s)*1e3
    return elapsed, out.sum().item()
LANES = {"pdist_f32": (pdist, lambda x: torch.pdist(x, p=2.0))}
"#;
    let py = format!("{py_setup}{}", ft_api::harness_interleave::SAMPLE_LOOP_PY);
    let mut child = Command::new(&python)
        .args(ft_api::harness_interleave::interpreter_args(&py))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("missing PyTorch stdin")?;
    let mut reader = BufReader::new(child.stdout.take().ok_or("missing PyTorch stdout")?);
    let mut preamble = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!("PyTorch arm exited before {READY_MARKER}: {preamble}").into());
        }
        if line.trim() == READY_MARKER {
            break;
        }
        preamble.push_str(&line);
    }
    let torch_version = ft_api::harness_provenance::require_reported_version(&preamble)?;
    assert!(
        preamble.contains("PT_TIMED_STEPS forward"),
        "both pdist arms must time forward only"
    );
    println!(
        "executing_elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "{}",
        ft_api::harness_provenance::incumbent_provenance_block(torch_version, 8)
    );
    println!(
        "measurement=FORWARD ONLY (no-grad direct p=2; leaf built outside timer on BOTH sides)"
    );

    for _ in 0..4 {
        std::hint::black_box(timed_pdist(&values, N, M));
    }
    let reps = reps();
    let mut ft_times = Vec::with_capacity(reps);
    let mut pt_times = Vec::with_capacity(reps);
    let mut ft_first_half = Vec::with_capacity(reps);
    let mut ft_second_half = Vec::with_capacity(reps);
    let mut pt_first_half = Vec::with_capacity(reps);
    let mut pt_second_half = Vec::with_capacity(reps);
    let mut ft_checksum = 0.0;
    let mut pt_checksum = 0.0;
    for _ in 0..reps {
        let mut pt_slots = Vec::with_capacity(4);
        let mut ft_slots = Vec::with_capacity(4);
        for incumbent_slot in BALANCED_SQUARE {
            if incumbent_slot {
                let (milliseconds, checksum) = incumbent_sample(&mut stdin, &mut reader)?;
                pt_slots.push(milliseconds);
                pt_checksum = checksum;
            } else {
                let (milliseconds, checksum) = timed_pdist(&values, N, M);
                ft_slots.push(milliseconds);
                ft_checksum = checksum;
            }
        }
        pt_first_half.push(paired_slot_median([pt_slots[0], pt_slots[1]]));
        pt_second_half.push(paired_slot_median([pt_slots[2], pt_slots[3]]));
        ft_first_half.push(paired_slot_median([ft_slots[0], ft_slots[1]]));
        ft_second_half.push(paired_slot_median([ft_slots[2], ft_slots[3]]));
        pt_times.push(median(pt_slots));
        ft_times.push(median(ft_slots));
    }
    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    child.wait()?;

    let (ratio, ratio_lo, ratio_hi) = median_ratio_ci(&pt_times, &ft_times);
    let (pt_null, pt_null_lo, pt_null_hi) = median_ratio_ci(&pt_first_half, &pt_second_half);
    let (ft_null, ft_null_lo, ft_null_hi) = median_ratio_ci(&ft_first_half, &ft_second_half);
    let pt_calm = adjudicate_null(pt_null_lo, pt_null_hi, MAX_NULL_CI_WIDTH).is_quotable()
        && null_is_centered(pt_null);
    let ft_calm = adjudicate_null(ft_null_lo, ft_null_hi, MAX_NULL_CI_WIDTH).is_quotable()
        && null_is_centered(ft_null);
    let parity = if (ft_checksum - pt_checksum).abs() <= 1e-5 * pt_checksum.abs().max(1.0) {
        "match"
    } else {
        "MISMATCH"
    };
    println!(
        "pdist_f32 {N}x{M}: FT {:.3} ms PT {:.3} ms = FT {:.2}x {} | PT A/A {pt_null:.3} [{pt_null_lo:.3},{pt_null_hi:.3}] FT A/A {ft_null:.3} [{ft_null_lo:.3},{ft_null_hi:.3}] ratio {ratio:.3} [{ratio_lo:.3},{ratio_hi:.3}] parity {parity}",
        median(ft_times),
        median(pt_times),
        if ratio >= 1.0 { ratio } else { 1.0 / ratio },
        if ratio >= 1.0 { "FASTER" } else { "SLOWER" },
    );
    if !(pt_calm && ft_calm && parity == "match") {
        println!(
            "NOT QUOTABLE: require both A/A nulls centred within +/-{BALANCED_NULL_MAX_DEVIATION:.2}, calm CIs, and parity match"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{median, paired_slot_median, timed_pdist};

    #[test]
    fn forward_only_pdist_keeps_leaf_construction_outside_the_timed_region() {
        let (milliseconds, checksum) = timed_pdist(&[0.0, 3.0, 4.0, 0.0], 2, 2);
        assert!(milliseconds.is_finite() && milliseconds >= 0.0);
        assert_eq!(checksum.to_bits(), 5.0_f64.to_bits());
    }

    #[test]
    fn balanced_square_medians_match_the_incumbent_definition() {
        assert_eq!(median(vec![9.0, 1.0, 3.0, 7.0]), 5.0);
        assert_eq!(paired_slot_median([9.0, 1.0]), 5.0);
    }
}
