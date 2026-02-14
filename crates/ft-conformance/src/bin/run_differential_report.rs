#![forbid(unsafe_code)]

use std::path::PathBuf;

use ft_conformance::{HarnessConfig, emit_differential_report};
use ft_core::ExecutionMode;
use serde_json::json;

fn main() -> Result<(), String> {
    let mut mode = String::from("both");
    let mut output: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mode requires one of: strict|hardened|both".to_string())?;
                mode = value;
            }
            "--output" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output requires a file path".to_string())?;
                output = Some(PathBuf::from(value));
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: run_differential_report [--mode strict|hardened|both] [--output path]"
                ));
            }
        }
    }

    let modes = parse_modes(mode.as_str())?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = output.unwrap_or_else(|| {
        repo_root.join("artifacts/phase2c/conformance/differential_report_v1.json")
    });

    let report = emit_differential_report(
        &HarnessConfig::default_paths(),
        output_path.as_path(),
        modes.as_slice(),
    )?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": if report.blocking_drifts == 0 && report.failed_checks == 0 { "ok" } else { "needs_attention" },
            "output_path": output_path.display().to_string(),
            "oracle_available": report.oracle.available,
            "oracle_message": report.oracle.message,
            "modes": report.modes,
            "total_checks": report.total_checks,
            "failed_checks": report.failed_checks,
            "allowlisted_drifts": report.allowlisted_drifts,
            "blocking_drifts": report.blocking_drifts,
        }))
        .map_err(|error| format!("failed to serialize summary: {error}"))?
    );

    Ok(())
}

fn parse_modes(raw: &str) -> Result<Vec<ExecutionMode>, String> {
    match raw {
        "strict" => Ok(vec![ExecutionMode::Strict]),
        "hardened" => Ok(vec![ExecutionMode::Hardened]),
        "both" => Ok(vec![ExecutionMode::Strict, ExecutionMode::Hardened]),
        _ => Err(format!(
            "unsupported mode '{raw}'; expected strict|hardened|both"
        )),
    }
}
