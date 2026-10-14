#![forbid(unsafe_code)]

use std::borrow::Cow;
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
    pub summary: Cow<'static, str>,
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

    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Upper bound on retained evidence entries. The ledger records one entry per
    /// dispatched op (plus policy/backward/durability events); left unbounded this
    /// grew without limit for the lifetime of a session — a real memory leak for
    /// long-running inference, since each entry owns a heap-allocated summary
    /// `String` and the only readers want either the first (policy anchor) or the
    /// most recent entries (e.g. the durability search scans from the tail).
    const SOFT_CAP: usize = 1 << 15; // 32768 entries

    pub fn record(&mut self, kind: EvidenceKind, summary: impl Into<Cow<'static, str>>) {
        if self.entries.len() >= Self::SOFT_CAP {
            // Bound memory by dropping the oldest *middle* entries: keep
            // `entries[0]` (the policy-init anchor that consumers assert on) and
            // retain the most recent half. Amortised O(1) per record.
            let drop = self.entries.len() / 2;
            self.entries.drain(1..=drop);
        }
        self.entries.push(EvidenceEntry {
            ts_unix_ms: now_unix_ms(),
            kind,
            summary: summary.into(),
        });
    }

    /// Drop all retained evidence, freeing the backing allocation's contents.
    /// Lets long-running consumers reclaim ledger memory between batches without
    /// tearing down the whole `RuntimeContext`.
    pub fn clear(&mut self) {
        self.entries = Vec::new();
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

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeContext {
    mode: ExecutionMode,
    ledger: EvidenceLedger,
    controller: DecisionController,
}

impl RuntimeContext {
    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        let mut ledger = EvidenceLedger::with_capacity(2);
        ledger.record(EvidenceKind::Policy, policy_initialized_summary(mode));
        let controller = DecisionController::new(mode);
        Self {
            mode,
            ledger,
            controller,
        }
    }

    #[must_use]
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.mode = mode;
        self.controller.set_mode(mode);
        self.ledger
            .record(EvidenceKind::Policy, policy_switched_summary(mode));
    }

    #[must_use]
    pub fn ledger(&self) -> &EvidenceLedger {
        &self.ledger
    }

    pub fn ledger_mut(&mut self) -> &mut EvidenceLedger {
        &mut self.ledger
    }

    #[must_use]
    pub fn controller(&self) -> &DecisionController {
        &self.controller
    }

    pub fn controller_mut(&mut self) -> &mut DecisionController {
        &mut self.controller
    }

    /// Observe an operational evidence signal, update the Spec §6 Bayesian
    /// posterior and e-process drift monitor, select the minimum-expected-loss
    /// control action, and record consequential decisions to the evidence ledger.
    pub fn observe_signal(&mut self, signal: EvidenceSignal) -> DecisionOutcome {
        self.controller
            .evaluate_and_record(signal, &mut self.ledger)
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

// ── Spec Section 6: Alien-Artifact Decision Layer ────────────────────────

/// Runtime regime state space $S$ for the Spec §6 Bayesian controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeRegime {
    /// Normal deterministic execution within numerical and memory envelopes.
    Nominal,
    /// NaN/Inf activations or severe gradient-norm spike observed.
    NumericalInstability,
    /// Pool/heap allocation pressure approaching configured ceiling.
    MemoryPressure,
    /// Checkpoint/artifact checksum mismatch or RaptorQ scrub anomaly.
    CorruptionDetected,
    /// Reentrant autograd/dispatch hazard or invariant violation.
    ReentrantHazard,
}

impl RuntimeRegime {
    pub const ALL: [Self; 5] = [
        Self::Nominal,
        Self::NumericalInstability,
        Self::MemoryPressure,
        Self::CorruptionDetected,
        Self::ReentrantHazard,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Nominal => 0,
            Self::NumericalInstability => 1,
            Self::MemoryPressure => 2,
            Self::CorruptionDetected => 3,
            Self::ReentrantHazard => 4,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nominal => "Nominal",
            Self::NumericalInstability => "NumericalInstability",
            Self::MemoryPressure => "MemoryPressure",
            Self::CorruptionDetected => "CorruptionDetected",
            Self::ReentrantHazard => "ReentrantHazard",
        }
    }
}

impl fmt::Display for RuntimeRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Evidence signals observed by the runtime decision controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvidenceSignal {
    /// Clean kernel dispatch and backward step within nominal tolerances.
    DispatchOk,
    /// Non-finite (NaN or Inf) tensor elements detected.
    NanOrInfDetected { count: usize },
    /// Gradient L2 norm spike ratio relative to moving baseline.
    GradNormSpike { ratio: f64 },
    /// Memory allocation footprint relative to soft budget.
    AllocationPressure {
        used_bytes: usize,
        limit_bytes: usize,
    },
    /// Serialization or RaptorQ scrub anomaly (`recovered = true` if repaired via symbols).
    ChecksumOrDecodeAnomaly { recovered: bool },
    /// Illegal reentrant backward/hook/dispatch call detected.
    ReentrantCallDetected,
    /// Explicit calibration check comparing empirical error rate against target.
    CalibrationCheck {
        empirical_error_rate: f64,
        target_rate: f64,
    },
}

/// Control actions $A$ available to the Spec §6 decision controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlAction {
    /// Continue standard execution on the fast path.
    ContinueNominal,
    /// Enable finite-value anomaly sanitization / gradient clipping guards.
    EnableAnomalySanitizer,
    /// Reclaim cached buffers or advise batch-size reduction.
    ShrinkBatchOrReclaimCache,
    /// Trigger RaptorQ symbol scrub and quarantine suspect state.
    TriggerRaptorQScrubAndQuarantine,
    /// Fail closed immediately to prevent silent corruption or UB-equivalent drift.
    FailClosedAbort,
}

impl ControlAction {
    pub const ALL: [Self; 5] = [
        Self::ContinueNominal,
        Self::EnableAnomalySanitizer,
        Self::ShrinkBatchOrReclaimCache,
        Self::TriggerRaptorQScrubAndQuarantine,
        Self::FailClosedAbort,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::ContinueNominal => 0,
            Self::EnableAnomalySanitizer => 1,
            Self::ShrinkBatchOrReclaimCache => 2,
            Self::TriggerRaptorQScrubAndQuarantine => 3,
            Self::FailClosedAbort => 4,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContinueNominal => "ContinueNominal",
            Self::EnableAnomalySanitizer => "EnableAnomalySanitizer",
            Self::ShrinkBatchOrReclaimCache => "ShrinkBatchOrReclaimCache",
            Self::TriggerRaptorQScrubAndQuarantine => "TriggerRaptorQScrubAndQuarantine",
            Self::FailClosedAbort => "FailClosedAbort",
        }
    }
}

impl fmt::Display for ControlAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Asymmetric 5×5 loss matrix $L(a, s)$ indexed by `[action.index()][regime.index()]`.
///
/// Under-reacting to `CorruptionDetected` or `ReentrantHazard` carries severe
/// asymmetric cost compared to defensive mitigation on `Nominal`.
#[derive(Debug, Clone, PartialEq)]
pub struct LossMatrix {
    costs: [[f64; 5]; 5],
}

impl LossMatrix {
    #[must_use]
    pub const fn new(costs: [[f64; 5]; 5]) -> Self {
        Self { costs }
    }

    /// Canonical asymmetric loss matrix for `ExecutionMode::Strict`.
    #[must_use]
    pub const fn strict_default() -> Self {
        Self {
            // Columns: [Nominal, NumericalInstability, MemoryPressure, CorruptionDetected, ReentrantHazard]
            costs: [
                // ContinueNominal
                [0.0, 95.0, 75.0, 260.0, 500.0],
                // EnableAnomalySanitizer
                [14.0, 0.0, 55.0, 180.0, 340.0],
                // ShrinkBatchOrReclaimCache
                [12.0, 70.0, 0.0, 190.0, 350.0],
                // TriggerRaptorQScrubAndQuarantine
                [28.0, 65.0, 60.0, 0.0, 160.0],
                // FailClosedAbort
                [65.0, 40.0, 55.0, 22.0, 0.0],
            ],
        }
    }

    /// Canonical asymmetric loss matrix for `ExecutionMode::Hardened`.
    ///
    /// Hardened mode lowers the cost of defensive mitigations (`EnableAnomalySanitizer`,
    /// `ShrinkBatchOrReclaimCache`, `TriggerRaptorQScrubAndQuarantine`) so the controller
    /// transitions into protective postures earlier under uncertainty.
    #[must_use]
    pub const fn hardened_default() -> Self {
        Self {
            costs: [
                // ContinueNominal
                [0.0, 110.0, 90.0, 300.0, 550.0],
                // EnableAnomalySanitizer
                [12.0, 0.0, 45.0, 160.0, 310.0],
                // ShrinkBatchOrReclaimCache
                [11.0, 60.0, 0.0, 175.0, 320.0],
                // TriggerRaptorQScrubAndQuarantine
                [20.0, 50.0, 45.0, 0.0, 130.0],
                // FailClosedAbort
                [50.0, 35.0, 45.0, 18.0, 0.0],
            ],
        }
    }

    #[must_use]
    pub const fn for_mode(mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::Strict => Self::strict_default(),
            ExecutionMode::Hardened => Self::hardened_default(),
        }
    }

    #[must_use]
    pub const fn cost(&self, action: ControlAction, regime: RuntimeRegime) -> f64 {
        self.costs[action.index()][regime.index()]
    }

    /// Compute expected loss $E[L(a)] = \sum_{s \in S} L(a, s) P(s)$ for all 5 actions.
    #[must_use]
    pub fn expected_losses(&self, posterior: &[f64; 5]) -> [f64; 5] {
        let mut out = [0.0; 5];
        for (a_idx, row) in self.costs.iter().enumerate() {
            let mut sum = 0.0;
            for (s_idx, &cost) in row.iter().enumerate() {
                sum += cost * posterior[s_idx];
            }
            out[a_idx] = sum;
        }
        out
    }
}

/// Snapshot of a single Bayesian decision evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOutcome {
    pub regime: RuntimeRegime,
    pub action: ControlAction,
    pub expected_loss: f64,
    pub expected_losses: [f64; 5],
    pub posterior: [f64; 5],
    pub confidence: f64,
    pub entropy_bits: f64,
    pub e_value: f64,
    pub fallback_triggered: bool,
}

/// Spec §6 Bayesian Decision Controller with Dirichlet pseudo-count posterior,
/// asymmetric loss minimization, and anytime-valid e-process drift alarm.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionController {
    mode: ExecutionMode,
    loss_matrix: LossMatrix,
    alpha: [f64; 5],
    decay: f64,
    e_value: f64,
    e_threshold: f64,
    min_confidence: f64,
    observations: u64,
    last_action: ControlAction,
}

impl DecisionController {
    const INITIAL_ALPHA: [f64; 5] = [10.0, 0.15, 0.15, 0.05, 0.02];
    const DEFAULT_DECAY: f64 = 0.96;
    const DEFAULT_E_THRESHOLD: f64 = 20.0; // Ville's inequality alpha = 0.05
    const DEFAULT_MIN_CONFIDENCE: f64 = 0.42;

    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            loss_matrix: LossMatrix::for_mode(mode),
            alpha: Self::INITIAL_ALPHA,
            decay: Self::DEFAULT_DECAY,
            e_value: 1.0,
            e_threshold: Self::DEFAULT_E_THRESHOLD,
            min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
            observations: 0,
            last_action: ControlAction::ContinueNominal,
        }
    }

    #[must_use]
    pub const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.mode = mode;
        self.loss_matrix = LossMatrix::for_mode(mode);
    }

    #[must_use]
    pub const fn loss_matrix(&self) -> &LossMatrix {
        &self.loss_matrix
    }

    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    #[must_use]
    pub const fn e_value(&self) -> f64 {
        self.e_value
    }

    #[must_use]
    pub const fn last_action(&self) -> ControlAction {
        self.last_action
    }

    /// Normalized Dirichlet posterior mean $P(s \mid \text{evidence})$.
    #[must_use]
    pub fn posterior(&self) -> [f64; 5] {
        let total: f64 = self.alpha.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return [0.2; 5];
        }
        [
            self.alpha[0] / total,
            self.alpha[1] / total,
            self.alpha[2] / total,
            self.alpha[3] / total,
            self.alpha[4] / total,
        ]
    }

    /// Maximum-a-posteriori (MAP) runtime regime.
    #[must_use]
    pub fn map_regime(&self) -> RuntimeRegime {
        let post = self.posterior();
        let mut best_idx = 0;
        let mut best_prob = post[0];
        for (idx, &prob) in post.iter().enumerate().skip(1) {
            if prob > best_prob {
                best_prob = prob;
                best_idx = idx;
            }
        }
        RuntimeRegime::ALL[best_idx]
    }

    /// Posterior probability of the MAP regime.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        let post = self.posterior();
        post.into_iter().fold(0.0_f64, f64::max)
    }

    /// Shannon entropy $H(P) = -\sum_s P(s)\log_2 P(s)$ in bits (range $[0, \log_2 5 \approx 2.322]$).
    #[must_use]
    pub fn shannon_entropy_bits(&self) -> f64 {
        let post = self.posterior();
        let mut entropy = 0.0;
        for &p in &post {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Whether the anytime-valid e-process or minimum-confidence guard has triggered fallback.
    #[must_use]
    pub fn calibration_fallback_triggered(&self) -> bool {
        self.e_value >= self.e_threshold || self.confidence() < self.min_confidence
    }

    /// Evaluate the current state without mutating the controller.
    #[must_use]
    pub fn current_outcome(&self) -> DecisionOutcome {
        let posterior = self.posterior();
        let expected_losses = self.loss_matrix.expected_losses(&posterior);
        let mut best_idx = 0;
        let mut best_loss = expected_losses[0];
        for (idx, &loss) in expected_losses.iter().enumerate().skip(1) {
            if loss < best_loss {
                best_loss = loss;
                best_idx = idx;
            }
        }
        let mut action = ControlAction::ALL[best_idx];
        let fallback_triggered = self.calibration_fallback_triggered();
        if fallback_triggered && action == ControlAction::ContinueNominal {
            action = match self.mode {
                ExecutionMode::Hardened => {
                    if posterior[RuntimeRegime::CorruptionDetected.index()] >= 0.15 {
                        ControlAction::TriggerRaptorQScrubAndQuarantine
                    } else {
                        ControlAction::EnableAnomalySanitizer
                    }
                }
                ExecutionMode::Strict => {
                    if posterior[RuntimeRegime::ReentrantHazard.index()] >= 0.20
                        || posterior[RuntimeRegime::CorruptionDetected.index()] >= 0.25
                    {
                        ControlAction::FailClosedAbort
                    } else {
                        ControlAction::EnableAnomalySanitizer
                    }
                }
            };
            best_loss = expected_losses[action.index()];
        }

        DecisionOutcome {
            regime: self.map_regime(),
            action,
            expected_loss: best_loss,
            expected_losses,
            posterior,
            confidence: self.confidence(),
            entropy_bits: self.shannon_entropy_bits(),
            e_value: self.e_value,
            fallback_triggered,
        }
    }

    /// Observe an `EvidenceSignal`, update posterior and e-process state, and return the outcome.
    pub fn observe(&mut self, signal: EvidenceSignal) -> DecisionOutcome {
        self.observations = self.observations.saturating_add(1);
        for (idx, a) in self.alpha.iter_mut().enumerate() {
            let floor = Self::INITIAL_ALPHA[idx] * 0.25;
            *a = (*a * self.decay).max(floor);
        }

        let likelihood_ratio = match signal {
            EvidenceSignal::DispatchOk => {
                self.alpha[RuntimeRegime::Nominal.index()] += 2.5;
                0.75
            }
            EvidenceSignal::NanOrInfDetected { count } => {
                let weight = 12.0 + (count.min(64) as f64) * 0.5;
                self.alpha[RuntimeRegime::NumericalInstability.index()] += weight;
                6.5
            }
            EvidenceSignal::GradNormSpike { ratio } => {
                let clean_ratio = if ratio.is_finite() {
                    ratio.max(0.0)
                } else {
                    100.0
                };
                if clean_ratio > 10.0 {
                    let weight = (clean_ratio / 2.5).min(24.0);
                    self.alpha[RuntimeRegime::NumericalInstability.index()] += weight;
                    (clean_ratio / 4.0).clamp(1.5, 12.0)
                } else {
                    self.alpha[RuntimeRegime::Nominal.index()] += 1.0;
                    0.9
                }
            }
            EvidenceSignal::AllocationPressure {
                used_bytes,
                limit_bytes,
            } => {
                let ratio = if limit_bytes == 0 {
                    1.0
                } else {
                    (used_bytes as f64) / (limit_bytes as f64)
                };
                if ratio >= 0.85 {
                    let weight = ((ratio - 0.75) * 40.0).clamp(6.0, 28.0);
                    self.alpha[RuntimeRegime::MemoryPressure.index()] += weight;
                    (ratio * 4.0).clamp(2.0, 10.0)
                } else {
                    self.alpha[RuntimeRegime::Nominal.index()] += 1.2;
                    0.85
                }
            }
            EvidenceSignal::ChecksumOrDecodeAnomaly { recovered } => {
                let weight = if recovered { 10.0 } else { 20.0 };
                self.alpha[RuntimeRegime::CorruptionDetected.index()] += weight;
                if recovered { 4.5 } else { 14.0 }
            }
            EvidenceSignal::ReentrantCallDetected => {
                self.alpha[RuntimeRegime::ReentrantHazard.index()] += 25.0;
                25.0
            }
            EvidenceSignal::CalibrationCheck {
                empirical_error_rate,
                target_rate,
            } => {
                let emp = empirical_error_rate.clamp(0.0, 1.0);
                let tgt = target_rate.clamp(1e-6, 1.0);
                let ratio = emp / tgt;
                if ratio > 1.5 {
                    self.alpha[RuntimeRegime::NumericalInstability.index()] +=
                        (ratio * 1.5).min(12.0);
                } else {
                    self.alpha[RuntimeRegime::Nominal.index()] += 1.0;
                }
                ratio.clamp(0.5, 25.0)
            }
        };

        // Anytime-valid betting martingale / e-process update:
        // E_t = E_{t-1} * ((1 - lambda) + lambda * LR_t), clamped to [0.1, 1e6].
        const LAMBDA: f64 = 0.35;
        let factor = (1.0 - LAMBDA) + LAMBDA * likelihood_ratio;
        self.e_value = (self.e_value * factor).clamp(0.1, 1_000_000.0);

        let outcome = self.current_outcome();
        self.last_action = outcome.action;
        outcome
    }

    /// Observe an `EvidenceSignal`, update the controller, and record consequential
    /// decisions or drift alarms to the `EvidenceLedger`.
    pub fn evaluate_and_record(
        &mut self,
        signal: EvidenceSignal,
        ledger: &mut EvidenceLedger,
    ) -> DecisionOutcome {
        let prev_action = self.last_action;
        let outcome = self.observe(signal);
        if outcome.action != ControlAction::ContinueNominal
            || outcome.action != prev_action
            || outcome.fallback_triggered
        {
            ledger.record(
                EvidenceKind::Policy,
                format!(
                    "decision_controller regime={} action={} expected_loss={:.2} conf={:.3} e_val={:.2} fallback={}",
                    outcome.regime,
                    outcome.action,
                    outcome.expected_loss,
                    outcome.confidence,
                    outcome.e_value,
                    outcome.fallback_triggered
                ),
            );
        }
        outcome
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

// ── 4-Panel Operational Cockpit Snapshot ─────────────────────────────────

/// Structured 4-panel telemetry snapshot for FrankenTorch runtime dashboards.
#[derive(Debug, Clone, PartialEq)]
pub struct CockpitSnapshot {
    pub mode: ExecutionMode,
    pub decision: DecisionOutcome,
    pub total_entries: usize,
    pub dispatch_count: usize,
    pub backward_count: usize,
    pub policy_count: usize,
    pub durability_count: usize,
    pub envelopes_total: usize,
    pub scrub_ok: usize,
    pub scrub_recovered: usize,
    pub scrub_failed: usize,
    pub total_decode_proofs: usize,
    pub recent_tail: Vec<String>,
}

impl CockpitSnapshot {
    #[must_use]
    pub fn from_runtime(ctx: &RuntimeContext, envelopes: &[DurabilityEnvelope]) -> Self {
        let mut dispatch_count = 0usize;
        let mut backward_count = 0usize;
        let mut policy_count = 0usize;
        let mut durability_count = 0usize;
        for entry in ctx.ledger().entries() {
            match entry.kind {
                EvidenceKind::Dispatch => dispatch_count += 1,
                EvidenceKind::Backward => backward_count += 1,
                EvidenceKind::Policy => policy_count += 1,
                EvidenceKind::Durability => durability_count += 1,
            }
        }

        let mut scrub_ok = 0usize;
        let mut scrub_recovered = 0usize;
        let mut scrub_failed = 0usize;
        let mut total_decode_proofs = 0usize;
        for env in envelopes {
            match env.scrub_status {
                ScrubStatus::Ok => scrub_ok += 1,
                ScrubStatus::Recovered => scrub_recovered += 1,
                ScrubStatus::Failed => scrub_failed += 1,
            }
            total_decode_proofs += env.decode_proofs.len();
        }

        let recent_tail = ctx
            .ledger()
            .entries()
            .iter()
            .rev()
            .take(4)
            .map(|e| format!("{:?}: {}", e.kind, e.summary))
            .collect();

        Self {
            mode: ctx.mode(),
            decision: ctx.controller().current_outcome(),
            total_entries: ctx.ledger().len(),
            dispatch_count,
            backward_count,
            policy_count,
            durability_count,
            envelopes_total: envelopes.len(),
            scrub_ok,
            scrub_recovered,
            scrub_failed,
            total_decode_proofs,
            recent_tail,
        }
    }

    #[must_use]
    pub fn panel_decision_text(&self) -> String {
        format!(
            "mode: {:?}\nregime: {}\naction: {}\nconf: {:.3} | E[L]: {:.2}\ne-val: {:.2} | fallback: {}",
            self.mode,
            self.decision.regime,
            self.decision.action,
            self.decision.confidence,
            self.decision.expected_loss,
            self.decision.e_value,
            self.decision.fallback_triggered
        )
    }

    #[must_use]
    pub fn panel_dac_text(&self) -> String {
        format!(
            "ledger entries: {}\ndispatch: {}\nbackward: {}\npolicy: {}\ndurability: {}",
            self.total_entries,
            self.dispatch_count,
            self.backward_count,
            self.policy_count,
            self.durability_count
        )
    }

    #[must_use]
    pub fn panel_durability_text(&self) -> String {
        format!(
            "envelopes: {}\nscrub ok: {}\nrecovered: {}\nfailed: {}\ndecode proofs: {}",
            self.envelopes_total,
            self.scrub_ok,
            self.scrub_recovered,
            self.scrub_failed,
            self.total_decode_proofs
        )
    }

    #[must_use]
    pub fn panel_tail_text(&self) -> String {
        if self.recent_tail.is_empty() {
            "no evidence recorded".to_string()
        } else {
            self.recent_tail.join("\n")
        }
    }

    #[must_use]
    pub fn render_plain_text(&self) -> String {
        format!(
            "=== [1] Decision Layer ===\n{}\n=== [2] DAC Ledger ===\n{}\n=== [3] RaptorQ Durability ===\n{}\n=== [4] Evidence Tail ===\n{}",
            self.panel_decision_text(),
            self.panel_dac_text(),
            self.panel_durability_text(),
            self.panel_tail_text()
        )
    }
}

// ── Asupersync Structured-Concurrency Integration ────────────────────────

/// Execution error surfaced when an `asupersync::Cx` capability context is
/// cancelled or exhausts its poll/cost quota during a runtime-guarded step.
#[cfg(feature = "asupersync-integration")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CxExecutionError {
    Cancelled { stage: String, reason: String },
    BudgetExhausted { stage: String, detail: String },
}

#[cfg(feature = "asupersync-integration")]
impl fmt::Display for CxExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { stage, reason } => {
                write!(f, "asupersync Cx cancelled at stage '{stage}': {reason}")
            }
            Self::BudgetExhausted { stage, detail } => {
                write!(
                    f,
                    "asupersync Cx budget exhausted at stage '{stage}': {detail}"
                )
            }
        }
    }
}

#[cfg(feature = "asupersync-integration")]
impl std::error::Error for CxExecutionError {}

/// Returns the unbounded asupersync budget for structured-concurrency regions.
///
/// Requires the `asupersync-integration` feature.
#[cfg(feature = "asupersync-integration")]
#[must_use]
pub fn asupersync_infinite_budget() -> asupersync::types::Budget {
    asupersync::types::Budget::INFINITE
}

/// Constructs a detached capability-restricted `asupersync::Cx` for synchronous
/// adapters and deterministic cancellation tests without ambient authority.
#[cfg(feature = "asupersync-integration")]
#[must_use]
pub fn detached_runtime_cx() -> asupersync::Cx<asupersync::cx::cap::None> {
    asupersync::Cx::<asupersync::cx::cap::None>::detached_cancel_context()
}

/// Verifies that `cx` has not been cancelled and still has non-zero poll/cost budget,
/// recording a `Policy` audit entry in `ctx.ledger_mut()` on failure.
#[cfg(feature = "asupersync-integration")]
pub fn check_cx_checkpoint<CAPS>(
    cx: &asupersync::Cx<CAPS>,
    ctx: &mut RuntimeContext,
    stage: &str,
) -> Result<(), CxExecutionError> {
    let budget = cx.budget();
    if budget.poll_quota == 0 {
        let detail = "poll_quota == 0".to_string();
        ctx.ledger_mut().record(
            EvidenceKind::Policy,
            format!("cx budget exhausted stage={stage}: {detail}"),
        );
        return Err(CxExecutionError::BudgetExhausted {
            stage: stage.to_string(),
            detail,
        });
    }
    if budget.cost_quota == Some(0) {
        let detail = "cost_quota == 0".to_string();
        ctx.ledger_mut().record(
            EvidenceKind::Policy,
            format!("cx budget exhausted stage={stage}: {detail}"),
        );
        return Err(CxExecutionError::BudgetExhausted {
            stage: stage.to_string(),
            detail,
        });
    }
    if let Err(err) = cx.checkpoint() {
        let reason = err.to_string();
        ctx.ledger_mut().record(
            EvidenceKind::Policy,
            format!("cx cancelled stage={stage}: {reason}"),
        );
        return Err(CxExecutionError::Cancelled {
            stage: stage.to_string(),
            reason,
        });
    }
    Ok(())
}

/// Runs a synchronous runtime-guarded operation under an explicit `&asupersync::Cx`,
/// enforcing pre- and post-execution cancellation checkpoints and recording DAC
/// provenance into `ctx`.
#[cfg(feature = "asupersync-integration")]
pub fn run_with_cx<CAPS, T, F>(
    cx: &asupersync::Cx<CAPS>,
    ctx: &mut RuntimeContext,
    op_name: &str,
    f: F,
) -> Result<T, CxExecutionError>
where
    F: FnOnce(&asupersync::Cx<CAPS>, &mut RuntimeContext) -> T,
{
    check_cx_checkpoint(cx, ctx, op_name)?;
    let output = f(cx, ctx);
    check_cx_checkpoint(cx, ctx, op_name)?;
    ctx.ledger_mut().record(
        EvidenceKind::Dispatch,
        format!("cx_exec op={op_name} status=completed"),
    );
    Ok(output)
}

// ── FrankenTUI 4-Panel Cockpit Integration ───────────────────────────────

/// Returns the baseline FrankenTUI style used by the optional dashboard hooks.
///
/// Requires the `frankentui-integration` feature.
#[cfg(feature = "frankentui-integration")]
#[must_use]
pub fn frankentui_default_style() -> ftui::Style {
    ftui::Style::default()
}

/// Returns a regime-appropriate `ftui::Style` (green for nominal, yellow for
/// numerical/memory warnings, red for corruption/reentrant hazards).
#[cfg(feature = "frankentui-integration")]
#[must_use]
pub fn frankentui_regime_style(regime: RuntimeRegime) -> ftui::Style {
    use ftui::{PackedRgba, Style};
    match regime {
        RuntimeRegime::Nominal => Style::default().fg(PackedRgba::rgb(80, 220, 120)),
        RuntimeRegime::NumericalInstability | RuntimeRegime::MemoryPressure => {
            Style::default().fg(PackedRgba::rgb(240, 200, 60))
        }
        RuntimeRegime::CorruptionDetected | RuntimeRegime::ReentrantHazard => {
            Style::default().fg(PackedRgba::rgb(240, 80, 80)).bold()
        }
    }
}

/// Four-panel FrankenTUI widget rendering the runtime's Decision Layer, DAC
/// Evidence Ledger, RaptorQ Durability Envelope summary, and Recent Evidence Tail.
#[cfg(feature = "frankentui-integration")]
#[derive(Debug, Clone)]
pub struct FrankenTuiCockpit<'a> {
    pub snapshot: &'a CockpitSnapshot,
}

#[cfg(feature = "frankentui-integration")]
impl<'a> FrankenTuiCockpit<'a> {
    #[must_use]
    pub const fn new(snapshot: &'a CockpitSnapshot) -> Self {
        Self { snapshot }
    }
}

#[cfg(feature = "frankentui-integration")]
impl ftui::widgets::Widget for FrankenTuiCockpit<'_> {
    fn render(&self, area: ftui::core::geometry::Rect, frame: &mut ftui::Frame) {
        use ftui::core::geometry::Rect;
        use ftui::widgets::block::Block;
        use ftui::widgets::paragraph::Paragraph;

        if area.width < 4 || area.height < 4 {
            return;
        }
        let half_w = area.width / 2;
        let rem_w = area.width - half_w;
        let half_h = area.height / 2;
        let rem_h = area.height - half_h;

        let top_left = Rect::new(area.x, area.y, half_w, half_h);
        let top_right = Rect::new(area.x + half_w, area.y, rem_w, half_h);
        let bottom_left = Rect::new(area.x, area.y + half_h, half_w, rem_h);
        let bottom_right = Rect::new(area.x + half_w, area.y + half_h, rem_w, rem_h);

        let regime_style = frankentui_regime_style(self.snapshot.decision.regime);

        let p1 = Paragraph::new(self.snapshot.panel_decision_text())
            .style(regime_style)
            .block(
                Block::bordered()
                    .title("Decision Layer (Spec §6)")
                    .border_style(regime_style),
            );
        p1.render(top_left, frame);

        let p2 = Paragraph::new(self.snapshot.panel_dac_text())
            .style(frankentui_default_style())
            .block(Block::bordered().title("DAC Provenance Ledger"));
        p2.render(top_right, frame);

        let p3 = Paragraph::new(self.snapshot.panel_durability_text())
            .style(frankentui_default_style())
            .block(Block::bordered().title("RaptorQ Durability"));
        p3.render(bottom_left, frame);

        let p4 = Paragraph::new(self.snapshot.panel_tail_text())
            .style(frankentui_default_style())
            .block(Block::bordered().title("Recent Evidence Tail"));
        p4.render(bottom_right, frame);
    }
}

/// Renders the 4-panel `FrankenTuiCockpit` into an offscreen `ftui::Buffer` of
/// dimensions `width × height` and returns both the buffer and its extracted
/// multi-line text representation.
#[cfg(feature = "frankentui-integration")]
#[must_use]
pub fn render_cockpit_buffer(
    ctx: &RuntimeContext,
    envelopes: &[DurabilityEnvelope],
    width: u16,
    height: u16,
) -> (ftui::Buffer, String) {
    use ftui::core::geometry::Rect;
    use ftui::widgets::Widget as _;

    let snapshot = CockpitSnapshot::from_runtime(ctx, envelopes);
    let cockpit = FrankenTuiCockpit::new(&snapshot);
    let mut pool = ftui::GraphemePool::new();
    let mut frame = ftui::Frame::new(width, height, &mut pool);
    cockpit.render(Rect::new(0, 0, width, height), &mut frame);

    let mut rendered = String::with_capacity(usize::from(width + 1) * usize::from(height));
    for y in 0..height {
        if y > 0 {
            rendered.push('\n');
        }
        let mut line = String::with_capacity(usize::from(width));
        for cell in frame.buffer.row_cells(y) {
            line.push(cell.content.as_char().unwrap_or(' '));
        }
        rendered.push_str(line.trim_end());
    }
    (frame.buffer, rendered)
}

fn now_unix_ms() -> u128 {
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn policy_initialized_summary(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Strict => "mode initialized to Strict",
        ExecutionMode::Hardened => "mode initialized to Hardened",
    }
}

fn policy_switched_summary(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Strict => "mode switched to Strict",
        ExecutionMode::Hardened => "mode switched to Hardened",
    }
}

#[cfg(test)]
mod tests {
    use ft_core::ExecutionMode;
    use ft_serialize::{DecodeMode, decode_checkpoint};

    use super::{
        CockpitSnapshot, ControlAction, DecisionController, DurabilityEnvelope, EvidenceKind,
        EvidenceSignal, LossMatrix, RuntimeContext, RuntimeRegime, ScrubStatus,
    };

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
    fn runtime_policy_evidence_golden_summary_matches_fixture() {
        use std::fmt::Write as _;

        let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
        ctx.set_mode(ExecutionMode::Hardened);
        let mut summary = String::new();
        summary.push_str("runtime_policy=ft_runtime_policy_pass20\n");
        let _ = writeln!(&mut summary, "mode={:?}", ctx.mode());
        let _ = writeln!(&mut summary, "entries={}", ctx.ledger().len());
        for (idx, entry) in ctx.ledger().entries().iter().enumerate() {
            let _ = writeln!(&mut summary, "{idx}:{:?}:{}", entry.kind, entry.summary);
        }

        assert_eq!(
            summary,
            include_str!(
                "../../../artifacts/optimization/golden_outputs/ft_runtime_policy_pass20.txt"
            )
        );
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
            "source_hash": "det64:0000000000000000",
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

    #[test]
    fn evidence_ledger_is_bounded_and_pins_policy_anchor() {
        let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
        let anchor = ctx.ledger().entries()[0].clone();
        assert_eq!(anchor.kind, EvidenceKind::Policy);

        // Far exceed the soft cap; a previously-unbounded ledger would retain
        // every entry (the inference memory leak this guards against).
        for i in 0..(super::EvidenceLedger::SOFT_CAP * 3) {
            ctx.ledger_mut()
                .record(EvidenceKind::Dispatch, format!("op {i}"));
        }

        assert!(
            ctx.ledger().len() <= super::EvidenceLedger::SOFT_CAP,
            "ledger must stay bounded by the soft cap"
        );
        // The policy-init anchor is preserved at index 0...
        assert_eq!(ctx.ledger().entries()[0], anchor);
        // ...and the most recent evidence is retained (tail readers depend on it).
        assert!(
            ctx.ledger()
                .entries()
                .last()
                .expect("non-empty")
                .summary
                .contains(&format!("op {}", super::EvidenceLedger::SOFT_CAP * 3 - 1))
        );

        ctx.ledger_mut().clear();
        assert_eq!(ctx.ledger().len(), 0);
    }

    #[test]
    fn evidence_ledger_clear_reclaims_retained_storage() {
        let mut ledger = super::EvidenceLedger::new();
        for i in 0..super::EvidenceLedger::SOFT_CAP {
            ledger.record(EvidenceKind::Dispatch, format!("op {i}"));
        }
        assert!(
            ledger.entries.capacity() >= super::EvidenceLedger::SOFT_CAP,
            "setup should grow the ledger allocation"
        );

        ledger.clear();

        assert!(ledger.is_empty());
        assert_eq!(
            ledger.entries.capacity(),
            0,
            "clear should release the retained allocation for long-running sessions"
        );
    }

    #[test]
    fn decision_controller_nominal_and_regime_transitions() {
        let mut controller = DecisionController::new(ExecutionMode::Strict);
        let initial = controller.current_outcome();
        assert_eq!(initial.regime, RuntimeRegime::Nominal);
        assert_eq!(initial.action, ControlAction::ContinueNominal);
        assert!(initial.confidence > 0.70);
        assert!(!initial.fallback_triggered);

        // NaN/Inf signal transitions MAP regime to NumericalInstability and selects EnableAnomalySanitizer
        let nan_outcome = controller.observe(EvidenceSignal::NanOrInfDetected { count: 16 });
        assert_eq!(nan_outcome.regime, RuntimeRegime::NumericalInstability);
        assert_eq!(nan_outcome.action, ControlAction::EnableAnomalySanitizer);
        assert!(nan_outcome.e_value > 1.0);

        // Memory pressure signal transitions MAP regime to MemoryPressure and selects ShrinkBatchOrReclaimCache
        let mut mem_controller = DecisionController::new(ExecutionMode::Hardened);
        let _ = mem_controller.observe(EvidenceSignal::AllocationPressure {
            used_bytes: 980,
            limit_bytes: 1000,
        });
        let mem_outcome = mem_controller.observe(EvidenceSignal::AllocationPressure {
            used_bytes: 990,
            limit_bytes: 1000,
        });
        assert_eq!(mem_outcome.regime, RuntimeRegime::MemoryPressure);
        assert_eq!(mem_outcome.action, ControlAction::ShrinkBatchOrReclaimCache);

        // Corruption signal selects TriggerRaptorQScrubAndQuarantine
        let mut corr_controller = DecisionController::new(ExecutionMode::Strict);
        let corr_outcome =
            corr_controller.observe(EvidenceSignal::ChecksumOrDecodeAnomaly { recovered: false });
        assert_eq!(corr_outcome.regime, RuntimeRegime::CorruptionDetected);
        assert_eq!(
            corr_outcome.action,
            ControlAction::TriggerRaptorQScrubAndQuarantine
        );

        // Reentrant hazard selects FailClosedAbort
        let mut reent_controller = DecisionController::new(ExecutionMode::Strict);
        let reent_outcome = reent_controller.observe(EvidenceSignal::ReentrantCallDetected);
        assert_eq!(reent_outcome.regime, RuntimeRegime::ReentrantHazard);
        assert_eq!(reent_outcome.action, ControlAction::FailClosedAbort);
    }

    #[test]
    fn decision_controller_e_process_calibration_fallback_and_ledger_recording() {
        let mut ctx = RuntimeContext::new(ExecutionMode::Hardened);
        assert_eq!(ctx.ledger().len(), 1);

        // Nominal dispatch does not spam policy ledger when action remains ContinueNominal
        let nom = ctx.observe_signal(EvidenceSignal::DispatchOk);
        assert_eq!(nom.action, ControlAction::ContinueNominal);
        assert_eq!(ctx.ledger().len(), 1);

        // Repeated calibration drift trips the e-process threshold (>= 20.0)
        for _ in 0..5 {
            let _ = ctx.observe_signal(EvidenceSignal::CalibrationCheck {
                empirical_error_rate: 0.25,
                target_rate: 0.01,
            });
        }
        assert!(ctx.controller().e_value() >= 20.0);
        assert!(ctx.controller().calibration_fallback_triggered());
        assert!(ctx.ledger().len() > 1);
        assert!(
            ctx.ledger()
                .entries()
                .iter()
                .any(|e| e.summary.contains("decision_controller"))
        );

        // Verify LossMatrix symmetry/asymmetry invariants
        let strict_matrix = LossMatrix::strict_default();
        assert!(
            strict_matrix.cost(
                ControlAction::ContinueNominal,
                RuntimeRegime::ReentrantHazard
            ) > strict_matrix.cost(ControlAction::FailClosedAbort, RuntimeRegime::Nominal)
        );
    }

    #[test]
    fn cockpit_snapshot_renders_four_panels() {
        let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
        ctx.ledger_mut().record(EvidenceKind::Dispatch, "matmul op");
        ctx.ledger_mut()
            .record(EvidenceKind::Backward, "backward step");
        let _ = ctx.observe_signal(EvidenceSignal::GradNormSpike { ratio: 30.0 });

        let mut env =
            DurabilityEnvelope::new("art-cockpit", "checkpoint", "blake3:123", 32, 4, 0.125);
        env.mark_scrub_status(ScrubStatus::Recovered);
        env.add_decode_proof("symbol repair", "blake3:proof1");

        let snap = CockpitSnapshot::from_runtime(&ctx, &[env]);
        assert_eq!(snap.dispatch_count, 1);
        assert_eq!(snap.backward_count, 1);
        assert_eq!(snap.envelopes_total, 1);
        assert_eq!(snap.scrub_recovered, 1);
        assert_eq!(snap.total_decode_proofs, 1);

        let text = snap.render_plain_text();
        assert!(text.contains("[1] Decision Layer"));
        assert!(text.contains("[2] DAC Ledger"));
        assert!(text.contains("[3] RaptorQ Durability"));
        assert!(text.contains("[4] Evidence Tail"));
        assert!(text.contains("EnableAnomalySanitizer"));
    }

    #[cfg(feature = "asupersync-integration")]
    #[test]
    fn asupersync_infinite_budget_and_cx_execution_work() {
        let budget = super::asupersync_infinite_budget();
        assert_eq!(budget.poll_quota, u32::MAX);

        let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
        let cx = super::detached_runtime_cx();

        let val = super::run_with_cx(&cx, &mut ctx, "linear_step", |_cx, runtime| {
            runtime
                .ledger_mut()
                .record(EvidenceKind::Backward, "grad_ok");
            42usize
        })
        .expect("active Cx should succeed");
        assert_eq!(val, 42);
        assert!(ctx.ledger().entries().iter().any(|e| {
            e.summary
                .contains("cx_exec op=linear_step status=completed")
        }));

        // Cancel Cx and verify fail-closed checkpoint error + audit ledger entry
        cx.cancel_with(asupersync::CancelKind::User, Some("test abort"));
        let err = super::run_with_cx(&cx, &mut ctx, "cancelled_step", |_cx, _runtime| 99usize)
            .expect_err("cancelled Cx must fail closed");
        assert!(matches!(err, super::CxExecutionError::Cancelled { .. }));
        assert!(
            ctx.ledger()
                .entries()
                .iter()
                .any(|e| e.summary.contains("cx cancelled stage=cancelled_step"))
        );
    }

    #[cfg(feature = "frankentui-integration")]
    #[test]
    fn frankentui_default_style_and_cockpit_buffer_render() {
        let style = super::frankentui_default_style();
        let _ = style;

        let mut ctx = RuntimeContext::new(ExecutionMode::Hardened);
        let _ = ctx.observe_signal(EvidenceSignal::NanOrInfDetected { count: 2 });
        let env = DurabilityEnvelope::new("art-tui", "conformance", "blake3:xyz", 16, 4, 0.25);

        let (buffer, text) = super::render_cockpit_buffer(&ctx, &[env], 96, 20);
        assert_eq!(buffer.width(), 96);
        assert_eq!(buffer.height(), 20);
        assert!(text.contains("Decision Layer"));
        assert!(text.contains("DAC Provenance Ledger"));
        assert!(text.contains("RaptorQ Durability"));
        assert!(text.contains("Recent Evidence Tail"));
    }
}
