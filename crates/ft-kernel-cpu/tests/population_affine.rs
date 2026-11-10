//! `population_affine_f32_into`: one dense layer per population member, each with its own
//! weights, in a declared order of operations and on the caller's thread only.
//!
//! The oracle is the plain scalar loop, written here independently of the kernel: for each
//! output, the bias, then each input's product in ascending index, in f32. The kernel's own
//! test of the rayon pool lives here too, in its own test binary, because it saturates the
//! global pool for as long as it runs.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;

use ft_kernel_cpu::{KernelError, population_affine_f32_into};
use rayon::prelude::*;

/// SplitMix64, so the data is fixed and needs no RNG crate.
struct Words(u64);

impl Words {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in (-1, 1) times a scale from 1e-3 to 1e3, so sums cancel and round.
    fn value(&mut self) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1_u64 << 24) as f32;
        let scale = [1e-3_f32, 1.0, 1e3][(self.next() % 3) as usize];
        (2.0 * unit - 1.0) * scale
    }

    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.value()).collect()
    }
}

struct Layer {
    batch: usize,
    inputs: usize,
    outputs: usize,
    x: Vec<f32>,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Layer {
    fn random(words: &mut Words, batch: usize, inputs: usize, outputs: usize) -> Self {
        Self {
            batch,
            inputs,
            outputs,
            x: words.values(batch * inputs),
            w: words.values(batch * inputs * outputs),
            b: words.values(batch * outputs),
        }
    }

    fn weight(&self, member: usize, input: usize, output: usize) -> f32 {
        self.w[(member * self.inputs + input) * self.outputs + output]
    }

    /// The declared order, output by output: `step(z, x, w)` adds one product.
    fn oracle_with(&self, step: impl Fn(f32, f32, f32) -> f32, bias_last: bool) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.batch * self.outputs);
        for member in 0..self.batch {
            for output in 0..self.outputs {
                let bias = self.b[member * self.outputs + output];
                let mut z = if bias_last { 0.0 } else { bias };
                for input in 0..self.inputs {
                    let x = self.x[member * self.inputs + input];
                    z = step(z, x, self.weight(member, input, output));
                }
                out.push(if bias_last { z + bias } else { z });
            }
        }
        out
    }

    fn oracle(&self) -> Vec<f32> {
        self.oracle_with(|z, x, w| z + x * w, false)
    }

    fn kernel(&self) -> Result<Vec<f32>, KernelError> {
        let mut out = vec![f32::NAN; self.batch * self.outputs];
        population_affine_f32_into(
            &self.x,
            &self.w,
            &self.b,
            self.batch,
            self.inputs,
            self.outputs,
            &mut out,
        )?;
        Ok(out)
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// FrankenLife's controller layers (104 sensor inputs, hidden widths 8 to 128, 36 outputs) and
/// edge shapes.
const SHAPES: [(usize, usize, usize); 8] = [
    (37, 104, 64),
    (37, 64, 32),
    (37, 32, 36),
    (9, 104, 128),
    (9, 128, 96),
    (5, 96, 36),
    (1, 1, 1),
    (3, 7, 1),
];

#[test]
fn every_output_is_the_declared_scalar_order_bit_for_bit() -> Result<(), KernelError> {
    let mut words = Words(0x5eed);
    let mut fused_differs = false;
    let mut bias_last_differs = false;
    for (batch, inputs, outputs) in SHAPES {
        let layer = Layer::random(&mut words, batch, inputs, outputs);
        let expected = layer.oracle();
        assert_eq!(
            bits(&layer.kernel()?),
            bits(&expected),
            "[{batch}, {inputs}, {outputs}]"
        );
        // The comparison tells the orders apart: a fused multiply-add, or the bias added last,
        // gives other bits on this data.
        fused_differs |=
            bits(&layer.oracle_with(|z, x, w| x.mul_add(w, z), false)) != bits(&expected);
        bias_last_differs |= bits(&layer.oracle_with(|z, x, w| z + x * w, true)) != bits(&expected);
    }
    assert!(fused_differs, "no shape distinguishes a fused multiply-add");
    assert!(
        bias_last_differs,
        "no shape distinguishes the bias added last"
    );
    Ok(())
}

#[test]
fn each_member_uses_only_its_own_weights_and_any_split_agrees() -> Result<(), KernelError> {
    let mut words = Words(0xfeed);
    let (batch, inputs, outputs) = (12, 104, 64);
    let layer = Layer::random(&mut words, batch, inputs, outputs);
    let whole = layer.kernel()?;

    // Member by member, and in two uneven halves, the outputs are the same bits.
    let mut pieces = Vec::new();
    for (start, end) in [(0, 5), (5, batch)] {
        let mut out = vec![0.0; (end - start) * outputs];
        population_affine_f32_into(
            &layer.x[start * inputs..end * inputs],
            &layer.w[start * inputs * outputs..end * inputs * outputs],
            &layer.b[start * outputs..end * outputs],
            end - start,
            inputs,
            outputs,
            &mut out,
        )?;
        pieces.extend(out);
    }
    assert_eq!(bits(&pieces), bits(&whole));

    // Changing member 7's weights changes member 7's outputs and no other member's. The same
    // seed rebuilds the same layer first.
    let mut changed = Layer::random(&mut Words(0xfeed), batch, inputs, outputs);
    assert_eq!(bits(&changed.kernel()?), bits(&whole));
    for weight in &mut changed.w[7 * inputs * outputs..8 * inputs * outputs] {
        *weight = -*weight;
    }
    let after = changed.kernel()?;
    for member in 0..batch {
        let span = member * outputs..(member + 1) * outputs;
        let same = bits(&after[span.clone()]) == bits(&whole[span]);
        assert_eq!(same, member != 7, "member {member}");
    }
    Ok(())
}

#[test]
fn shapes_are_checked_and_empty_shapes_give_the_bias() -> Result<(), KernelError> {
    let (batch, inputs, outputs) = (3, 4, 2);
    let x = vec![1.0; batch * inputs];
    let w = vec![1.0; batch * inputs * outputs];
    let b = vec![0.5; batch * outputs];
    let mut out = vec![0.0; batch * outputs];
    population_affine_f32_into(&x, &w, &b, batch, inputs, outputs, &mut out)?;
    assert_eq!(out, vec![4.5; batch * outputs]);

    let short = |shape: &[usize], len: usize| KernelError::ShapeMismatch {
        lhs: shape.to_vec(),
        rhs: vec![len],
    };
    assert_eq!(
        population_affine_f32_into(&x[1..], &w, &b, batch, inputs, outputs, &mut out),
        Err(short(&[3, 4], 11))
    );
    assert_eq!(
        population_affine_f32_into(&x, &w[1..], &b, batch, inputs, outputs, &mut out),
        Err(short(&[3, 4, 2], 23))
    );
    assert_eq!(
        population_affine_f32_into(&x, &w, &b[1..], batch, inputs, outputs, &mut out),
        Err(short(&[3, 2], 5))
    );
    let mut long = vec![0.0; batch * outputs + 1];
    assert_eq!(
        population_affine_f32_into(&x, &w, &b, batch, inputs, outputs, &mut long),
        Err(short(&[3, 2], 7))
    );
    assert_eq!(
        population_affine_f32_into(&[], &[], &[], usize::MAX, 2, 1, &mut []),
        Err(KernelError::ShapeOverflow {
            context: "population_affine_f32_into"
        })
    );

    // No inputs: each output is its bias. No outputs, or no members: nothing to write.
    let mut biased = vec![0.0; 2];
    population_affine_f32_into(&[], &[], &[0.25, -3.0], 1, 0, 2, &mut biased)?;
    assert_eq!(bits(&biased), bits(&[0.25, -3.0]));
    population_affine_f32_into(&[1.0], &[], &[], 1, 1, 0, &mut [])?;
    population_affine_f32_into(&[], &[], &[], 0, 104, 64, &mut [])?;
    Ok(())
}

/// Every worker of rayon's global pool is held while the kernel runs a large batch on another
/// thread. A kernel that handed work to the pool would wait for a worker and never finish; a
/// rayon parallel loop started the same way is the control that shows it would.
#[test]
fn it_runs_on_the_caller_while_every_rayon_worker_is_busy() -> Result<(), KernelError> {
    let workers = rayon::current_num_threads();
    let release = Arc::new(Barrier::new(workers + 1));
    let (started, held) = mpsc::channel();
    for _ in 0..workers {
        let release = Arc::clone(&release);
        let started = started.clone();
        rayon::spawn(move || {
            let _ = started.send(());
            release.wait();
        });
    }
    for _ in 0..workers {
        held.recv_timeout(Duration::from_secs(30))
            .expect("every rayon worker takes a holding job");
    }

    // The control: a parallel loop from a plain thread cannot finish while the pool is held.
    let (control_done, control) = mpsc::channel();
    let control_thread = std::thread::spawn(move || {
        let sum: u64 = (0..4096_u64).into_par_iter().map(|value| value * 2).sum();
        let _ = control_done.send(sum);
    });
    let control_blocked = control.recv_timeout(Duration::from_millis(500)).is_err();

    // The kernel on 512 members of FrankenLife's widest first layer.
    let layer = Layer::random(&mut Words(0xb10c), 512, 104, 128);
    let (kernel_done, kernel) = mpsc::channel();
    let kernel_thread = std::thread::spawn(move || {
        let result = layer
            .kernel()
            .map(|out| (bits(&out) == bits(&layer.oracle()), out.len()));
        let _ = kernel_done.send(result);
    });
    let finished = kernel.recv_timeout(Duration::from_secs(60));

    release.wait();
    let control_sum = control.recv_timeout(Duration::from_secs(60));
    let _ = control_thread.join();
    let _ = kernel_thread.join();

    assert!(
        control_blocked,
        "the control finished while every rayon worker was held: the pool was not saturated"
    );
    assert_eq!(
        control_sum.ok(),
        Some(4096 * 4095),
        "the control finishes once released"
    );
    let (exact, len) = finished.expect("the kernel finished while every rayon worker was held")?;
    assert!(exact, "the kernel's outputs are the oracle's");
    assert_eq!(len, 512 * 128);
    Ok(())
}
