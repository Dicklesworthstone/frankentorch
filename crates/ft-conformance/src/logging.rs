#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use ft_core::ExecutionMode;
use serde::Serialize;
use serde_json::Value;

pub const STRUCTURED_LOG_SCHEMA_VERSION: &str = "ft-conformance-log-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StructuredCaseLog {
    pub schema_version: &'static str,
    pub ts_unix_ms: u128,
    pub suite_id: &'static str,
    pub scenario_id: String,
    pub fixture_id: String,
    pub packet_id: &'static str,
    pub mode: &'static str,
    pub seed: u64,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
    pub outcome: &'static str,
    pub reason_code: String,
    #[serde(flatten)]
    pub extra_fields: BTreeMap<String, Value>,
}

impl StructuredCaseLog {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        suite_id: &'static str,
        fixture_id: &'static str,
        packet_id: &'static str,
        case_name: &str,
        mode: ExecutionMode,
        artifact_refs: Vec<String>,
        replay_command: String,
        outcome: &'static str,
        reason_code: impl Into<String>,
    ) -> Self {
        let scenario_id = format!(
            "{suite_id}/{}:{}",
            mode_label(mode),
            canonicalize(case_name)
        );
        let seed = deterministic_seed(
            [
                suite_id,
                fixture_id,
                packet_id,
                scenario_id.as_str(),
                mode_label(mode),
            ]
            .as_slice(),
        );
        Self {
            schema_version: STRUCTURED_LOG_SCHEMA_VERSION,
            ts_unix_ms: now_unix_ms(),
            suite_id,
            scenario_id,
            fixture_id: fixture_id.to_string(),
            packet_id,
            mode: mode_label(mode),
            seed,
            env_fingerprint: env_fingerprint(),
            artifact_refs,
            replay_command,
            outcome,
            reason_code: reason_code.into(),
            extra_fields: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn with_extra_fields(mut self, extra_fields: BTreeMap<String, Value>) -> Self {
        self.extra_fields = extra_fields;
        self
    }
}

#[must_use]
pub fn mode_label(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Strict => "strict",
        ExecutionMode::Hardened => "hardened",
    }
}

#[must_use]
pub fn env_fingerprint() -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    env!("CARGO_PKG_NAME").hash(&mut hasher);
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    std::env::consts::OS.hash(&mut hasher);
    std::env::consts::ARCH.hash(&mut hasher);
    if cfg!(debug_assertions) {
        "debug".hash(&mut hasher);
    } else {
        "release".hash(&mut hasher);
    }
    format!("det64:{:016x}", hasher.finish())
}

fn deterministic_seed(parts: &[&str]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for part in parts {
        part.hash(&mut hasher);
    }
    hasher.finish()
}

fn canonicalize(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else if ch.is_whitespace() || ch == '/' || ch == ':' {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unnamed_case".to_string()
    } else {
        out
    }
}

fn now_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}
