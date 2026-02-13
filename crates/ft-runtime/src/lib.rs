#![forbid(unsafe_code)]

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

#[cfg(feature = "asupersync-integration")]
pub fn asupersync_infinite_budget() -> asupersync::types::Budget {
    asupersync::types::Budget::INFINITE
}

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

    use super::{DurabilityEnvelope, EvidenceKind, RuntimeContext, ScrubStatus};

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
}
