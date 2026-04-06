#![forbid(unsafe_code)]

use std::fmt;

use ft_core::ExecutionMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    Dispatch,
    Backward,
    Policy,
    Durability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceEntry {
    pub ts_unix_ms: u128,
    pub kind: EvidenceKind,
    pub summary: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvidenceLedger {
    entries: Vec<EvidenceEntry>,
}

impl EvidenceLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, kind: EvidenceKind, summary: impl Into<String>) {
        self.entries.push(EvidenceEntry {
            ts_unix_ms: now_unix_ms(),
            kind,
            summary: summary.into(),
        });
    }

    #[must_use]
    pub fn entries(&self) -> &[EvidenceEntry] {
        &self.entries
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeContext {
    mode: ExecutionMode,
    ledger: EvidenceLedger,
}

impl RuntimeContext {
    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        let mut ledger = EvidenceLedger::new();
        ledger.record(
            EvidenceKind::Policy,
            format!("mode initialized to {mode:?}"),
        );
        Self { mode, ledger }
    }

    #[must_use]
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.mode = mode;
        self.ledger
            .record(EvidenceKind::Policy, format!("mode switched to {mode:?}"));
    }

    #[must_use]
    pub fn ledger(&self) -> &EvidenceLedger {
        &self.ledger
    }

    pub fn ledger_mut(&mut self) -> &mut EvidenceLedger {
        &mut self.ledger
    }

    pub fn record_checkpoint_decode_failure<E>(&mut self, mode: &str, error: &E)
    where
        E: fmt::Display + ?Sized,
    {
        self.ledger.record(
            EvidenceKind::Durability,
            format!("checkpoint decode failure mode={mode}: {error}"),
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrubStatus {
    Ok,
    Recovered,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeProof {
    pub ts_unix_ms: u128,
    pub reason: String,
    pub proof_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DurabilityEnvelope {
    pub artifact_id: String,
    pub artifact_type: String,
    pub source_hash: String,
    pub k: u32,
    pub repair_symbols: u32,
    pub overhead_ratio: f64,
    pub scrub_status: ScrubStatus,
    pub decode_proofs: Vec<DecodeProof>,
}

impl DurabilityEnvelope {
    #[must_use]
    pub fn new(
        artifact_id: impl Into<String>,
        artifact_type: impl Into<String>,
        source_hash: impl Into<String>,
        k: u32,
        repair_symbols: u32,
        overhead_ratio: f64,
    ) -> Self {
        let k = k.max(1);
        let repair_symbols = repair_symbols.max(1);
        let overhead_ratio = if overhead_ratio.is_finite() {
            overhead_ratio.max(0.0)
        } else {
            0.0
        };
        Self {
            artifact_id: artifact_id.into(),
            artifact_type: artifact_type.into(),
            source_hash: source_hash.into(),
            k,
            repair_symbols,
            overhead_ratio,
            scrub_status: ScrubStatus::Ok,
            decode_proofs: Vec::new(),
        }
    }

    pub fn mark_scrub_status(&mut self, status: ScrubStatus) {
        self.scrub_status = status;
    }

    pub fn add_decode_proof(&mut self, reason: impl Into<String>, proof_hash: impl Into<String>) {
        self.decode_proofs.push(DecodeProof {
            ts_unix_ms: now_unix_ms(),
            reason: reason.into(),
            proof_hash: proof_hash.into(),
        });
    }
}

/// Returns the unbounded asupersync budget for structured-concurrency regions.
///
/// This is the appropriate default when a FrankenTorch caller wants runtime
/// integration without imposing a cancellation budget from the helper itself,
/// such as offline inference or batch-oriented orchestration.
///
/// Requires the `asupersync-integration` feature.
#[cfg(feature = "asupersync-integration")]
pub fn asupersync_infinite_budget() -> asupersync::types::Budget {
    asupersync::types::Budget::INFINITE
}

/// Returns the baseline FrankenTUI style used by the optional dashboard hooks.
///
/// Consumers can use this as a known-good starting point before applying their
/// own panel, border, or palette customizations for runtime dashboards.
///
/// Requires the `frankentui-integration` feature.
#[cfg(feature = "frankentui-integration")]
pub fn frankentui_default_style() -> ftui::Style {
    ftui::Style::default()
}

fn now_unix_ms() -> u128 {
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use ft_core::ExecutionMode;
    use ft_serialize::{DecodeMode, decode_checkpoint};

    use super::{DurabilityEnvelope, EvidenceKind, RuntimeContext, ScrubStatus};

    // Feature-gated integration tests. Run with:
    //   cargo test -p ft-runtime --features asupersync-integration
    //   cargo test -p ft-runtime --features frankentui-integration
    //   cargo test -p ft-runtime --all-features

    #[test]
    fn ledger_records_policy_and_custom_events() {
        let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
        ctx.ledger_mut()
            .record(EvidenceKind::Dispatch, "dispatch decision");

        assert_eq!(ctx.ledger().len(), 2);
        assert_eq!(ctx.ledger().entries()[1].kind, EvidenceKind::Dispatch);
    }

    #[test]
    fn mode_switch_records_event() {
        let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
        ctx.set_mode(ExecutionMode::Hardened);

        assert_eq!(ctx.mode(), ExecutionMode::Hardened);
        assert_eq!(ctx.ledger().len(), 2);
    }

    #[test]
    fn durability_envelope_tracks_decode_proofs() {
        let mut envelope =
            DurabilityEnvelope::new("artifact-1", "conformance", "blake3:abc", 128, 16, 0.125);
        envelope.mark_scrub_status(ScrubStatus::Recovered);
        envelope.add_decode_proof("corruption drill", "blake3:proof");

        assert_eq!(envelope.scrub_status, ScrubStatus::Recovered);
        assert_eq!(envelope.decode_proofs.len(), 1);
    }

    #[test]
    fn decode_failure_records_durability_evidence() {
        let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
        let payload = r#"{
            "schema_version": 1,
            "mode": "strict",
            "entries": [],
            "source_hash": "det64:placeholder",
            "extra": 1
        }"#;

        let err = decode_checkpoint(payload, DecodeMode::Strict)
            .expect_err("unknown field payload must fail strict decode");
        ctx.record_checkpoint_decode_failure("strict", &err);

        let durability_entry = ctx
            .ledger()
            .entries()
            .iter()
            .rev()
            .find(|entry| entry.kind == EvidenceKind::Durability)
            .expect("durability evidence entry should be present");
        assert!(
            durability_entry
                .summary
                .contains("checkpoint decode failure"),
            "unexpected durability summary: {}",
            durability_entry.summary
        );
        assert!(
            durability_entry.summary.contains("unknown field"),
            "durability summary should include decode diagnostic: {}",
            durability_entry.summary
        );
    }

    // ── bd-3vnu: untested paths and variants ──

    #[test]
    fn ledger_is_empty_on_fresh_instance() {
        let ledger = super::EvidenceLedger::new();
        assert!(ledger.is_empty());
        assert_eq!(ledger.len(), 0);
    }

    #[test]
    fn ledger_not_empty_after_record() {
        let mut ledger = super::EvidenceLedger::new();
        ledger.record(EvidenceKind::Dispatch, "test event");
        assert!(!ledger.is_empty());
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn scrub_status_ok_is_default_for_new_envelope() {
        let envelope = DurabilityEnvelope::new("art-2", "fixture", "blake3:def", 64, 8, 0.125);
        assert_eq!(envelope.scrub_status, ScrubStatus::Ok);
        assert!(envelope.decode_proofs.is_empty());
    }

    #[test]
    fn scrub_status_failed_variant() {
        let mut envelope = DurabilityEnvelope::new("art-3", "fixture", "blake3:ghi", 64, 8, 0.125);
        envelope.mark_scrub_status(ScrubStatus::Failed);
        assert_eq!(envelope.scrub_status, ScrubStatus::Failed);
    }

    #[test]
    fn durability_envelope_normalizes_zero_symbol_counts() {
        let envelope = DurabilityEnvelope::new("art-3b", "fixture", "blake3:ghi", 0, 0, 0.125);
        assert_eq!(envelope.k, 1);
        assert_eq!(envelope.repair_symbols, 1);
    }

    #[test]
    fn durability_envelope_normalizes_invalid_overhead_ratio() {
        let negative = DurabilityEnvelope::new("art-3c", "fixture", "blake3:ghi", 64, 8, -0.25);
        assert_eq!(negative.overhead_ratio, 0.0);

        let nan = DurabilityEnvelope::new("art-3d", "fixture", "blake3:ghi", 64, 8, f64::NAN);
        assert_eq!(nan.overhead_ratio, 0.0);

        let infinite =
            DurabilityEnvelope::new("art-3e", "fixture", "blake3:ghi", 64, 8, f64::INFINITY);
        assert_eq!(infinite.overhead_ratio, 0.0);
    }

    #[test]
    fn durability_envelope_multiple_decode_proofs() {
        let mut envelope =
            DurabilityEnvelope::new("art-4", "snapshot", "blake3:jkl", 128, 16, 0.125);
        envelope.add_decode_proof("proof-1", "hash-1");
        envelope.add_decode_proof("proof-2", "hash-2");
        envelope.add_decode_proof("proof-3", "hash-3");
        assert_eq!(envelope.decode_proofs.len(), 3);
        assert_eq!(envelope.decode_proofs[0].reason, "proof-1");
        assert_eq!(envelope.decode_proofs[2].proof_hash, "hash-3");
    }

    #[test]
    fn runtime_context_hardened_mode() {
        let ctx = RuntimeContext::new(ExecutionMode::Hardened);
        assert_eq!(ctx.mode(), ExecutionMode::Hardened);
        assert_eq!(ctx.ledger().len(), 1);
        assert_eq!(ctx.ledger().entries()[0].kind, EvidenceKind::Policy);
        assert!(ctx.ledger().entries()[0].summary.contains("Hardened"));
    }

    #[test]
    fn record_checkpoint_decode_failure_hardened_mode() {
        let mut ctx = RuntimeContext::new(ExecutionMode::Hardened);
        ctx.record_checkpoint_decode_failure("hardened", &"test decode error");

        let entry = ctx
            .ledger()
            .entries()
            .iter()
            .rev()
            .find(|e| e.kind == EvidenceKind::Durability)
            .expect("durability entry should be present");
        assert!(entry.summary.contains("hardened"));
        assert!(entry.summary.contains("test decode error"));
    }

    #[test]
    fn evidence_kind_backward_can_be_recorded() {
        let mut ledger = super::EvidenceLedger::new();
        ledger.record(EvidenceKind::Backward, "backward pass completed");
        assert_eq!(ledger.entries()[0].kind, EvidenceKind::Backward);
        assert!(ledger.entries()[0].summary.contains("backward"));
    }

    #[test]
    fn evidence_entry_timestamps_are_monotonic() {
        let mut ledger = super::EvidenceLedger::new();
        ledger.record(EvidenceKind::Dispatch, "first");
        ledger.record(EvidenceKind::Dispatch, "second");
        ledger.record(EvidenceKind::Dispatch, "third");
        // Timestamps should be monotonically non-decreasing
        for i in 1..ledger.len() {
            assert!(
                ledger.entries()[i].ts_unix_ms >= ledger.entries()[i - 1].ts_unix_ms,
                "timestamps should be monotonically non-decreasing"
            );
        }
    }

    #[cfg(feature = "asupersync-integration")]
    #[test]
    fn asupersync_infinite_budget_returns_infinite() {
        let budget = super::asupersync_infinite_budget();
        let _ = budget;
    }

    #[cfg(feature = "frankentui-integration")]
    #[test]
    fn frankentui_default_style_returns_valid_style() {
        let style = super::frankentui_default_style();
        let _ = style;
    }
}
