//! Golden-fixture generator for the fused no-grad `tensor_pdist` p != 2 path.
//!
//!   rch exec -- cargo run -p ft-api --example pdist_golden

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::fmt::Write as _;

type GoldenResult<T> = Result<T, Box<dyn std::error::Error>>;

fn p_label(p: f64) -> &'static str {
    if p == f64::INFINITY {
        "inf"
    } else if p == 0.5 {
        "0.5"
    } else if p == 1.0 {
        "1"
    } else if p == 3.0 {
        "3"
    } else {
        "custom"
    }
}

fn dump_case(out: &mut String, n: usize, m: usize, p: f64) -> GoldenResult<()> {
    let input: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.011).sin() - 0.2).collect();

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let reference_input = session
        .tensor_variable(input.clone(), vec![n, m], true)?;
    let reference = session.tensor_pdist(reference_input, p)?;
    let reference_values = session.tensor_values(reference)?;

    let fused_input = session.tensor_variable(input, vec![n, m], false)?;
    let fused = session.tensor_pdist(fused_input, p)?;
    let fused_values = session.tensor_values(fused)?;

    if reference_values.len() != fused_values.len() {
        return Err(format!(
            "n={n} m={m} p={} length mismatch: {} vs {}",
            p_label(p),
            reference_values.len(),
            fused_values.len()
        )
        .into());
    }
    let _ = writeln!(
        out,
        "case n={n} m={m} p={} len={}",
        p_label(p),
        fused_values.len()
    );
    for (idx, (reference, fused)) in reference_values.iter().zip(fused_values.iter()).enumerate() {
        match reference.to_bits().cmp(&fused.to_bits()) {
            std::cmp::Ordering::Equal => {}
            _ => return Err("pdist golden bit mismatch".into()),
        }
        let _ = writeln!(out, "{idx}: {:#018x}", fused.to_bits());
    }
    Ok(())
}

fn main() -> GoldenResult<()> {
    let mut output = String::new();
    let _ = writeln!(output, "frankentorch-0ijz pdist_fused_golden");
    for &(n, m) in &[(8usize, 5usize), (13, 7)] {
        for &p in &[1.0f64, 3.0, 0.5, f64::INFINITY] {
            dump_case(&mut output, n, m, p)?;
        }
    }
    if let Ok(path) = std::env::var("FT_PDIST_GOLDEN_OUT") {
        std::fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}
