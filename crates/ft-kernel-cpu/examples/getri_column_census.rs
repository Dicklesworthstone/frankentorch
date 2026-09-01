//! `frankentorch-stale-tuning-constants-lzku6` — the getri column census.
//!
//! # The hypothesis this exists to test, not to confirm
//!
//! Ledger 292g put getri's BACKWARD solve at 34-37% of `inv`, roughly twice its own forward
//! (16-21%), and offered a structural reason: the forward restricts itself to the columns the
//! identity can make nonzero (`cols = pe`) while the backward sweeps all `n`. The tempting
//! conclusion is that the backward has simply never been given the same restriction.
//!
//! **That is a hypothesis about whether the backward COULD be restricted, and it is exactly the
//! kind of plausible claim this campaign has falsified three times by counting** (ledger 291: only
//! 5 of 31 calls qualified; 292: cholesky never calls `dgemm_sub_into` at all; 292g: my own
//! "residual" was the LU factorisation). So this counts columns instead of asserting structure.
//!
//! # What is counted
//!
//! Per half, the COLUMN-UPDATES actually performed — the inner sweep width summed over every
//! (row, k) pair. And for the backward, how many of those updates read a source element that was
//! still ZERO at the moment it was read.
//!
//! That last number is the whole question. A restriction can only save work that is provably zero
//! WHEN TOUCHED. `y` is unit-lower-triangular, so row `kk` starts nonzero only in `[0, kk]` — but
//! the backward runs bottom-up and overwrites rows with a DENSE `z` as it goes, so by the time a
//! row is read as a source it may already be dense. Counting at the moment of use is the only
//! honest way to ask; the structural argument alone cannot distinguish the two cases.
//!
//! **Registered prediction:** if `z = U^-1 y` is mathematically dense — U^-1 upper times y lower
//! is a full product — the zero-touch count should be near 0 and there is NO lever, only an
//! explanation for the 2x. If it is a large fraction, the restriction was genuinely missed.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example getri_column_census -- [n]

use ft_core::{DType, Device, TensorMeta};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(256);

    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    println!("PROV host={} n={n}  (census is a COUNT, not a timing — load-insensitive)", host.trim());
    println!(
        "PREDICTION REGISTERED: if z = U^-1 y is mathematically dense, backward zero-touch is ~0 \
         and there is NO restriction to win — only an explanation for the 2x."
    );

    let a: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let v = (((i * 7 + j * 13) % 101) as f64 - 50.0) / 25.0;
            if i == j { v + n as f64 } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    let previous = ft_kernel_cpu::set_lu_inverse_census_enabled(true);
    let _ = ft_kernel_cpu::lu_inverse_census_take();
    let out = ft_kernel_cpu::inv_tensor_contiguous_f64(&a, &meta).expect("inv");
    let (fwd, back, zero) = ft_kernel_cpu::lu_inverse_census_take();
    ft_kernel_cpu::set_lu_inverse_census_enabled(previous);

    // Non-vacuity: the census must have observed a real inverse, not an early return.
    let finite = out.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, n * n, "inverse is not finite everywhere");

    println!("\nCOLUMN-UPDATES PERFORMED");
    println!("  forward   {fwd:>14}");
    println!("  backward  {back:>14}   ({:.2}x the forward)", back as f64 / fwd as f64);
    println!(
        "  backward updates whose SOURCE element was still zero when read: {zero} \
         ({:.3}% of the backward's work)",
        100.0 * zero as f64 / back as f64
    );

    println!("\nVERDICT");
    let pct = 100.0 * zero as f64 / back as f64;
    if pct < 1.0 {
        println!(
            "  The backward's sweep is DENSE BY NECESSITY: {pct:.3}% of its column-updates read a \
             zero. `z = U^-1 y` is upper-triangular times lower-triangular, which is a full \
             product, so there is NO restriction to add — the 2x is explained, not fixable this \
             way. The structural hypothesis is REFUTED."
        );
    } else {
        println!(
            "  {pct:.3}% of the backward's column-updates read a structurally zero source, so a \
             restriction could skip them. That is a lever, and it needs its own sweep and \
             certification before anyone believes the size of it."
        );
    }
    println!(
        "  Forward/backward column-update ratio {:.2}x against a measured TIME ratio of ~2x \
         (ledger 292g) — if those agree, the phase split is explained by work done and not by a \
         rate difference.",
        back as f64 / fwd as f64
    );
}
