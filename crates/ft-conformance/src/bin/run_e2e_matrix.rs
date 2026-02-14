#![forbid(unsafe_code)]

use std::path::PathBuf;

use ft_conformance::{HarnessConfig, emit_e2e_forensics_matrix_filtered};
use ft_core::ExecutionMode;
use serde_json::json;

fn main() -> Result<(), String> {
    let mut mode = String::from("both");
    let mut output: Option<PathBuf> = None;
    let mut packet: Option<String> = None;

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
            "--packet" => {
                let value = args.next().ok_or_else(|| {
                    "--packet requires a packet id (e.g., FT-P2C-004)".to_string()
                })?;
                packet = Some(value);
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: run_e2e_matrix [--mode strict|hardened|both] [--packet FT-P2C-00X] [--output path]"
                ));
            }
        }
    }

    let modes = parse_modes(mode.as_str())?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = output
        .unwrap_or_else(|| repo_root.join("artifacts/phase2c/e2e_forensics/e2e_matrix.jsonl"));

    let summary = emit_e2e_forensics_matrix_filtered(
        &HarnessConfig::default_paths(),
        output_path.as_path(),
        modes.as_slice(),
        packet.as_deref(),
    )?;

    let mode_labels: Vec<&str> = summary
        .modes
        .iter()
        .map(|m| match m {
            ExecutionMode::Strict => "strict",
            ExecutionMode::Hardened => "hardened",
        })
        .collect();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "output_path": summary.output_path.display().to_string(),
            "log_entries": summary.log_entries,
            "failed_entries": summary.failed_entries,
            "modes": mode_labels,
            "packet_filter": packet,
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
