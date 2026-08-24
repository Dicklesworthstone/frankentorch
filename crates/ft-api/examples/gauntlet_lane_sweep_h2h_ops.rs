//! Isolated whole-op H2H lanes for the training routes in `frankentorch-58zjz`.
//!
//! This deliberately does not share the leased board source.  The two arms are
//! still in one invocation, follow the shared balanced-square schedule, report
//! both median and min A/A nulls, and time only forward + weighted loss +
//! backward.  Leaves and non-uniform loss weights are built before the clock.

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    BALANCED_SQUARE, MAX_NULL_CI_WIDTH, QUIT_REQUEST, READY_MARKER, TIMED_STEPS, adjudicate_null,
    parse_sample_line, parse_timed_steps, sample_request, timed_region_disagreement,
};
use ft_core::ExecutionMode;

const REPS: usize = 16;
const BOOTSTRAP_REPS: usize = 2_000;
const NULL_CENTER_TOLERANCE: f64 = 0.02;

const LIN_BATCH: usize = 512;
const LIN_IN: usize = 1024;
const LIN_OUT_WIDE: usize = 128;
const LIN_OUT_NARROW: usize = 512;

const C2_BATCH: usize = 16;
const C2_CHANNELS: usize = 32;
const C2_HEIGHT: usize = 32;
const C2_WIDTH: usize = 32;
const C2_KERNEL: usize = 3;

const ATTN_BATCH: usize = 4;
const ATTN_HEADS: usize = 8;
const ATTN_SEQUENCE: usize = 256;
const ATTN_DIM: usize = 64;

type LaneRun<'a> = Box<dyn Fn() -> (f64, f64) + 'a>;

fn seq(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 251) as f64 * 0.001 - 0.12).collect()
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
    assert!(!numerator.is_empty());
    let point = median(numerator.to_vec()) / median(denominator.to_vec());
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut state = 0x58_7a_6a_5d_u64;
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

fn null_passes(point: f64, low: f64, high: f64) -> bool {
    adjudicate_null(low, high, MAX_NULL_CI_WIDTH).is_quotable()
        && point.is_finite()
        && (point - 1.0).abs() <= NULL_CENTER_TOLERANCE
}

fn gradient_l1(report: &ft_autograd::BackwardReport, node: ft_autograd::TensorNodeId) -> f64 {
    report
        .gradient(node)
        .expect("gradient")
        .iter()
        .map(|value| value.abs())
        .sum()
}

fn timed_linear(
    values: &[f64],
    weights: &[f64],
    mask: &[f64],
    batch: usize,
    input_features: usize,
    output_features: usize,
) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable(values.to_vec(), vec![batch, input_features], true)
        .expect("linear input");
    let weight = session
        .tensor_variable(
            weights.to_vec(),
            vec![output_features, input_features],
            true,
        )
        .expect("linear weight");
    let loss_weights = session
        .tensor_variable(mask.to_vec(), vec![batch, output_features], false)
        .expect("linear loss weights");
    let started = Instant::now();
    let output = session
        .functional_linear(input, weight, None)
        .expect("linear forward");
    let weighted = session
        .tensor_mul(output, loss_weights)
        .expect("linear mask");
    let loss = session.tensor_sum(weighted).expect("linear sum");
    let report = session.tensor_backward(loss).expect("linear backward");
    (
        started.elapsed().as_secs_f64() * 1_000.0,
        gradient_l1(&report, input),
    )
}

#[allow(clippy::too_many_arguments)]
fn timed_conv2d(
    values: &[f64],
    weights: &[f64],
    mask: &[f64],
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    height: usize,
    width: usize,
    kernel: usize,
    padding: (usize, usize),
) -> (f64, f64) {
    let output_height = height + 2 * padding.0 - kernel + 1;
    let output_width = width + 2 * padding.1 - kernel + 1;
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable(
            values.to_vec(),
            vec![batch, input_channels, height, width],
            true,
        )
        .expect("conv2d input");
    let weight = session
        .tensor_variable(
            weights.to_vec(),
            vec![output_channels, input_channels, kernel, kernel],
            true,
        )
        .expect("conv2d weight");
    let loss_weights = session
        .tensor_variable(
            mask.to_vec(),
            vec![batch, output_channels, output_height, output_width],
            false,
        )
        .expect("conv2d loss weights");
    let started = Instant::now();
    let output = session
        .functional_conv2d(input, weight, None, (1, 1), padding)
        .expect("conv2d forward");
    let weighted = session
        .tensor_mul(output, loss_weights)
        .expect("conv2d mask");
    let loss = session.tensor_sum(weighted).expect("conv2d sum");
    let report = session.tensor_backward(loss).expect("conv2d backward");
    (
        started.elapsed().as_secs_f64() * 1_000.0,
        gradient_l1(&report, input),
    )
}

fn timed_attention(
    query: &[f64],
    key: &[f64],
    value: &[f64],
    mask: &[f64],
    batch: usize,
    heads: usize,
    sequence: usize,
    dimension: usize,
) -> (f64, f64) {
    let shape = vec![batch, heads, sequence, dimension];
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = session
        .tensor_variable(query.to_vec(), shape.clone(), true)
        .expect("attention query");
    let k = session
        .tensor_variable(key.to_vec(), shape.clone(), true)
        .expect("attention key");
    let v = session
        .tensor_variable(value.to_vec(), shape.clone(), true)
        .expect("attention value");
    let loss_weights = session
        .tensor_variable(mask.to_vec(), shape, false)
        .expect("attention loss weights");
    let started = Instant::now();
    let output = session
        .functional_scaled_dot_product_attention(q, k, v, None, false, None)
        .expect("attention forward");
    let weighted = session
        .tensor_mul(output, loss_weights)
        .expect("attention mask");
    let loss = session.tensor_sum(weighted).expect("attention sum");
    let report = session.tensor_backward(loss).expect("attention backward");
    let checksum = gradient_l1(&report, q) + gradient_l1(&report, k) + gradient_l1(&report, v);
    (started.elapsed().as_secs_f64() * 1_000.0, checksum)
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
            return Err(format!("PyTorch closed stdout while sampling {lane}").into());
        }
        if let Some(sample) = parse_sample_line(&line) {
            assert_eq!(sample.lane, lane, "PyTorch returned the wrong lane");
            return Ok((sample.milliseconds, sample.gradient_checksum));
        }
    }
}

fn warmup_count() -> usize {
    std::env::var("FT_H2H_WARMUP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32)
}

fn python_setup() -> String {
    format!(
        r#"
import time, torch
import torch.nn.functional as Fn
print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)
torch.set_num_threads(8)
def seq(n):
    return ((torch.arange(n,dtype=torch.int64)%251).double())*0.001-0.12
linx=seq({LIN_BATCH}*{LIN_IN}).reshape({LIN_BATCH},{LIN_IN})
linw_wide=seq({LIN_OUT_WIDE}*{LIN_IN}).reshape({LIN_OUT_WIDE},{LIN_IN})
linw_narrow=seq({LIN_OUT_NARROW}*{LIN_IN}).reshape({LIN_OUT_NARROW},{LIN_IN})
linm_wide=seq({LIN_BATCH}*{LIN_OUT_WIDE}).reshape({LIN_BATCH},{LIN_OUT_WIDE})
linm_narrow=seq({LIN_BATCH}*{LIN_OUT_NARROW}).reshape({LIN_BATCH},{LIN_OUT_NARROW})
c2x=seq({C2_BATCH}*{C2_CHANNELS}*{C2_HEIGHT}*{C2_WIDTH}).reshape({C2_BATCH},{C2_CHANNELS},{C2_HEIGHT},{C2_WIDTH})
c2w=seq({C2_CHANNELS}*{C2_CHANNELS}*{C2_KERNEL}*{C2_KERNEL}).reshape({C2_CHANNELS},{C2_CHANNELS},{C2_KERNEL},{C2_KERNEL})
c2m=seq({C2_BATCH}*{C2_CHANNELS}*{C2_HEIGHT}*{C2_WIDTH}).reshape({C2_BATCH},{C2_CHANNELS},{C2_HEIGHT},{C2_WIDTH})
attnq=seq({ATTN_BATCH}*{ATTN_HEADS}*{ATTN_SEQUENCE}*{ATTN_DIM}).reshape({ATTN_BATCH},{ATTN_HEADS},{ATTN_SEQUENCE},{ATTN_DIM})
attnk=seq({ATTN_BATCH}*{ATTN_HEADS}*{ATTN_SEQUENCE}*{ATTN_DIM}).reshape({ATTN_BATCH},{ATTN_HEADS},{ATTN_SEQUENCE},{ATTN_DIM})
attnv=seq({ATTN_BATCH}*{ATTN_HEADS}*{ATTN_SEQUENCE}*{ATTN_DIM}).reshape({ATTN_BATCH},{ATTN_HEADS},{ATTN_SEQUENCE},{ATTN_DIM})
attnm=seq({ATTN_BATCH}*{ATTN_HEADS}*{ATTN_SEQUENCE}*{ATTN_DIM}).reshape({ATTN_BATCH},{ATTN_HEADS},{ATTN_SEQUENCE},{ATTN_DIM})
def timed(build):
    # All leaf and mask construction is outside this timestamp, just as on the Rust arm.
    x, w, mask, op = build()
    started=time.perf_counter()
    (op(x,w)*mask).sum().backward()
    return (time.perf_counter()-started)*1e3, x.grad.abs().sum().item()
def linear_wide():
    return timed(lambda: (linx.detach().clone().requires_grad_(True), linw_wide.detach().clone().requires_grad_(True), linm_wide, lambda x,w: Fn.linear(x,w)))
def linear_narrow():
    return timed(lambda: (linx.detach().clone().requires_grad_(True), linw_narrow.detach().clone().requires_grad_(True), linm_narrow, lambda x,w: Fn.linear(x,w)))
def conv2d_train():
    return timed(lambda: (c2x.detach().clone().requires_grad_(True), c2w.detach().clone().requires_grad_(True), c2m, lambda x,w: Fn.conv2d(x,w,None,(1,1),(1,1))))
def attention():
    q=attnq.detach().clone().requires_grad_(True)
    k=attnk.detach().clone().requires_grad_(True)
    v=attnv.detach().clone().requires_grad_(True)
    started=time.perf_counter()
    (Fn.scaled_dot_product_attention(q,k,v)*attnm).sum().backward()
    return (time.perf_counter()-started)*1e3, q.grad.abs().sum().item()+k.grad.abs().sum().item()+v.grad.abs().sum().item()
def run(build, _unused):
    return build()
LANES={{
  'linear_wide_masked': (linear_wide, None),
  'linear_narrow_masked': (linear_narrow, None),
  'conv2d_masked_train_ops': (conv2d_train, None),
  'attention_masked': (attention, None),
}}
print('PT_TIMED_STEPS forward,loss_sum,backward', flush=True)
"#,
    )
}

#[derive(Default)]
struct Samples {
    ft: Vec<f64>,
    pt: Vec<f64>,
    ft_first: Vec<f64>,
    ft_second: Vec<f64>,
    pt_first: Vec<f64>,
    pt_second: Vec<f64>,
    ft_first_min: Vec<f64>,
    ft_second_min: Vec<f64>,
    pt_first_min: Vec<f64>,
    pt_second_min: Vec<f64>,
    checksum: f64,
    pt_checksum: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = ft_kernel_cpu::pool::configure_global_pool();
    let linear_input = seq(LIN_BATCH * LIN_IN);
    let linear_wide_weight = seq(LIN_OUT_WIDE * LIN_IN);
    let linear_narrow_weight = seq(LIN_OUT_NARROW * LIN_IN);
    let linear_wide_mask = seq(LIN_BATCH * LIN_OUT_WIDE);
    let linear_narrow_mask = seq(LIN_BATCH * LIN_OUT_NARROW);
    let conv_input = seq(C2_BATCH * C2_CHANNELS * C2_HEIGHT * C2_WIDTH);
    let conv_weight = seq(C2_CHANNELS * C2_CHANNELS * C2_KERNEL * C2_KERNEL);
    let conv_mask = seq(C2_BATCH * C2_CHANNELS * C2_HEIGHT * C2_WIDTH);
    let attention_query = seq(ATTN_BATCH * ATTN_HEADS * ATTN_SEQUENCE * ATTN_DIM);
    let attention_key = seq(ATTN_BATCH * ATTN_HEADS * ATTN_SEQUENCE * ATTN_DIM);
    let attention_value = seq(ATTN_BATCH * ATTN_HEADS * ATTN_SEQUENCE * ATTN_DIM);
    let attention_mask = seq(ATTN_BATCH * ATTN_HEADS * ATTN_SEQUENCE * ATTN_DIM);

    let lanes: Vec<(&str, LaneRun<'_>)> = vec![
        (
            "linear_wide_masked",
            Box::new(|| {
                timed_linear(
                    &linear_input,
                    &linear_wide_weight,
                    &linear_wide_mask,
                    LIN_BATCH,
                    LIN_IN,
                    LIN_OUT_WIDE,
                )
            }),
        ),
        (
            "linear_narrow_masked",
            Box::new(|| {
                timed_linear(
                    &linear_input,
                    &linear_narrow_weight,
                    &linear_narrow_mask,
                    LIN_BATCH,
                    LIN_IN,
                    LIN_OUT_NARROW,
                )
            }),
        ),
        (
            "conv2d_masked_train_ops",
            Box::new(|| {
                timed_conv2d(
                    &conv_input,
                    &conv_weight,
                    &conv_mask,
                    C2_BATCH,
                    C2_CHANNELS,
                    C2_CHANNELS,
                    C2_HEIGHT,
                    C2_WIDTH,
                    C2_KERNEL,
                    (1, 1),
                )
            }),
        ),
        (
            "attention_masked",
            Box::new(|| {
                timed_attention(
                    &attention_query,
                    &attention_key,
                    &attention_value,
                    &attention_mask,
                    ATTN_BATCH,
                    ATTN_HEADS,
                    ATTN_SEQUENCE,
                    ATTN_DIM,
                )
            }),
        ),
    ];

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let py = format!(
        "{}{}",
        python_setup(),
        ft_api::harness_interleave::SAMPLE_LOOP_PY
    );
    let mut child = Command::new(&python)
        .args(ft_api::harness_interleave::interpreter_args(&py))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
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
    let mut preamble = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!("PyTorch exited before {READY_MARKER}: {preamble}").into());
        }
        if line.trim() == READY_MARKER {
            break;
        }
        preamble.push_str(&line);
    }
    let torch_version = ft_api::harness_provenance::require_reported_version(&preamble)?;
    let incumbent_steps =
        parse_timed_steps(&preamble).ok_or("PyTorch did not declare timed steps")?;
    if let Some(error) = timed_region_disagreement(TIMED_STEPS, &incumbent_steps) {
        return Err(error.into());
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
        "{}",
        ft_api::harness_provenance::measurement_host_block(rayon::current_num_threads())
    );
    println!(
        "sampling=interleaved balanced-square dual-A/A rounds={REPS} warmup={}",
        warmup_count()
    );

    for (name, run) in &lanes {
        for _ in 0..warmup_count() {
            let _ = run();
            let _ = incumbent_sample(&mut stdin, &mut reader, name)?;
        }
    }

    let mut samples: Vec<Samples> = (0..lanes.len()).map(|_| Samples::default()).collect();
    for _ in 0..REPS {
        for (index, (name, run)) in lanes.iter().enumerate() {
            let mut pt_slots = Vec::with_capacity(4);
            let mut ft_slots = Vec::with_capacity(4);
            for incumbent_slot in BALANCED_SQUARE {
                if incumbent_slot {
                    let (milliseconds, checksum) = incumbent_sample(&mut stdin, &mut reader, name)?;
                    pt_slots.push(milliseconds);
                    samples[index].pt_checksum = checksum;
                } else {
                    let (milliseconds, checksum) = run();
                    ft_slots.push(milliseconds);
                    samples[index].checksum = checksum;
                }
            }
            let lane = &mut samples[index];
            lane.pt.push(median(pt_slots.clone()));
            lane.ft.push(median(ft_slots.clone()));
            lane.pt_first
                .push(paired_slot_median([pt_slots[0], pt_slots[1]]));
            lane.pt_second
                .push(paired_slot_median([pt_slots[2], pt_slots[3]]));
            lane.ft_first
                .push(paired_slot_median([ft_slots[0], ft_slots[1]]));
            lane.ft_second
                .push(paired_slot_median([ft_slots[2], ft_slots[3]]));
            lane.pt_first_min.push(pt_slots[0].min(pt_slots[1]));
            lane.pt_second_min.push(pt_slots[2].min(pt_slots[3]));
            lane.ft_first_min.push(ft_slots[0].min(ft_slots[1]));
            lane.ft_second_min.push(ft_slots[2].min(ft_slots[3]));
        }
    }
    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    child.wait()?;

    println!(
        "lane                         FT(ms)    PT(ms)   PT/FT median [CI]        median A/A PT FT     min A/A PT FT     parity"
    );
    for ((name, _), lane) in lanes.iter().zip(&samples) {
        let (ratio, low, high) = median_ratio_ci(&lane.pt, &lane.ft);
        let (pt_null, pt_low, pt_high) = median_ratio_ci(&lane.pt_first, &lane.pt_second);
        let (ft_null, ft_low, ft_high) = median_ratio_ci(&lane.ft_first, &lane.ft_second);
        let (pt_min, pt_min_low, pt_min_high) =
            median_ratio_ci(&lane.pt_first_min, &lane.pt_second_min);
        let (ft_min, ft_min_low, ft_min_high) =
            median_ratio_ci(&lane.ft_first_min, &lane.ft_second_min);
        let parity =
            (lane.checksum - lane.pt_checksum).abs() <= 1e-6 * lane.pt_checksum.abs().max(1.0);
        println!(
            "{name:<28} {:8.3} {:8.3} {ratio:6.3} [{low:.3},{high:.3}]  {:.3}[{:.3},{:.3}] {} {:.3}[{:.3},{:.3}] {}  {:.3}[{:.3},{:.3}] {} {:.3}[{:.3},{:.3}] {}  {}",
            median(lane.ft.clone()),
            median(lane.pt.clone()),
            pt_null,
            pt_low,
            pt_high,
            if null_passes(pt_null, pt_low, pt_high) {
                "PASS"
            } else {
                "FAIL"
            },
            ft_null,
            ft_low,
            ft_high,
            if null_passes(ft_null, ft_low, ft_high) {
                "PASS"
            } else {
                "FAIL"
            },
            pt_min,
            pt_min_low,
            pt_min_high,
            if null_passes(pt_min, pt_min_low, pt_min_high) {
                "PASS"
            } else {
                "FAIL"
            },
            ft_min,
            ft_min_low,
            ft_min_high,
            if null_passes(ft_min, ft_min_low, ft_min_high) {
                "PASS"
            } else {
                "FAIL"
            },
            if parity { "match" } else { "MISMATCH" },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{null_passes, timed_attention, timed_conv2d, timed_linear};

    #[test]
    fn linear_non_uniform_loss_reaches_backward() {
        let (_, gradient) = timed_linear(&[1.5], &[2.0], &[0.25], 1, 1, 1);
        assert_eq!(gradient.to_bits(), 0.5_f64.to_bits());
    }

    #[test]
    fn conv2d_non_uniform_loss_reaches_backward() {
        let (_, gradient) = timed_conv2d(&[2.0], &[3.0], &[0.25], 1, 1, 1, 1, 1, 1, (0, 0));
        assert_eq!(gradient.to_bits(), 0.75_f64.to_bits());
    }

    #[test]
    fn attention_non_uniform_loss_reaches_value_gradient() {
        let (_, gradient) = timed_attention(&[1.0], &[2.0], &[3.0], &[0.25], 1, 1, 1, 1);
        assert_eq!(gradient.to_bits(), 0.25_f64.to_bits());
    }

    #[test]
    fn null_requires_centered_tight_interval() {
        assert!(null_passes(1.0, 0.99, 1.01));
        assert!(!null_passes(1.03, 1.02, 1.04));
    }
}
