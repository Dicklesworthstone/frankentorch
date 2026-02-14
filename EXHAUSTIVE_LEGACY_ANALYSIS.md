# EXHAUSTIVE_LEGACY_ANALYSIS.md — FrankenTorch

Date: 2026-02-13  
Method stack: `$porting-to-rust` Phase-2 Deep Extraction + `$alien-artifact-coding` + `$extreme-software-optimization` + RaptorQ durability + frankenlibc/frankenfs strict/hardened doctrine.

Doc-overhaul baseline matrix:
- `artifacts/phase2c/DOC_PASS00_BASELINE_GAP_MATRIX_V1.md`

## 0. Mission and Completion Criteria

This document defines exhaustive legacy extraction for FrankenTorch. Phase-2 is complete only when each subsystem required for full drop-in parity has:
1. explicit invariants,
2. explicit crate ownership,
3. explicit oracle families,
4. explicit strict/hardened policy behavior,
5. explicit performance and durability gates.

## 1. Source-of-Truth Crosswalk

Legacy corpus:
- `/data/projects/frankentorch/legacy_pytorch_code/pytorch`
- Upstream oracle: `pytorch/pytorch`

Project contracts:
- `/data/projects/frankentorch/COMPREHENSIVE_SPEC_FOR_FRANKENTORCH_V1.md` (sections 14-21)
- `/data/projects/frankentorch/EXISTING_PYTORCH_STRUCTURE.md`
- `/data/projects/frankentorch/PLAN_TO_PORT_PYTORCH_TO_RUST.md`
- `/data/projects/frankentorch/PROPOSED_ARCHITECTURE.md`
- `/data/projects/frankentorch/FEATURE_PARITY.md`

## 2. Quantitative Legacy Inventory (Measured)

- Total files: `21654`
- Python: `4266`
- Native: `c=194`, `cc=96`, `cpp=2058`, `cu=345`, `h=2266`, `hpp=66`
- Test-like files: `12226`

High-density zones:
- `torch/csrc/jit` (786 files)
- `torch/csrc/distributed` (247)
- `torch/csrc/api` (221)
- `torch/csrc/autograd` (100)
- `torch/_inductor/codegen` (92)
- `test/*` corpus (11792 files)

## 3. Subsystem Extraction Matrix (Legacy -> Rust)

| Legacy locus | Non-negotiable behavior to preserve | Target crates | Primary oracles | Phase-2 extraction deliverables |
|---|---|---|---|---|
| `c10/core/TensorImpl.h`, allocator/device headers | tensor metadata ownership, sizes/strides/device invariants | `ft-core` | tensor metadata tests + creation/view suites | TensorImpl contract ledger |
| `c10/core/DispatchKey*.h`, `ATen/Dispatch.h` | dispatch key ordering and kernel selection | `ft-dispatch` | `test/test_dispatch.py`, `test/test_python_dispatch.py` | dispatch precedence graph and proofs |
| `aten/src/ATen/native/cpu/*` | eager CPU kernel semantics | `ft-kernel-cpu` | op parity suites | op-by-op tolerance contract table |
| `torch/csrc/autograd/engine.*`, `function.h` | backward scheduling, reentrancy, graph task behavior | `ft-autograd` | `test/test_autograd.py` | autograd transition/state machine ledger |
| `torch/nn/*` + module plumbing | module forward/backward and state behavior | `ft-nn` | `test/test_nn.py`, `test/nn/*` | module state invariant catalog |
| `torch/csrc/serialization.cpp` | checkpoint round-trip and state-dict compatibility | `ft-serialize` | `test/test_serialization.py` | serialization compatibility matrix |
| `c10/core/DeviceGuard*`, `torch/csrc/{cuda,mps,xpu}` | device/stream guard semantics | `ft-device` | `test/test_cuda.py`, backend tests | device transition safety ledger |
| `torchgen/*`, `native_functions.yaml` | op schema -> dispatch registration consistency | `ft-dispatch`, `ft-kernel-cpu` | schema/parity tests | schema traceability map |

## 4. Alien-Artifact Invariant Ledger (Formal Obligations)

- `FT-I1` Tensor metadata integrity: `(storage, sizes, strides, dtype, device)` tuple remains valid under all implemented ops with explicit closure paths to full coverage.
- `FT-I2` Dispatch determinism: identical dispatch context chooses identical kernel.
- `FT-I3` Autograd consistency: backward graph scheduling preserves dependency and accumulation semantics.
- `FT-I4` In-place version safety: mutation paths uphold versioning and alias guarantees.
- `FT-I5` Checkpoint round-trip stability: model state survives save/load without semantic drift across implemented surfaces with explicit closure to full parity.

Required proof artifacts per implemented slice:
1. invariant statement,
2. executable witness fixtures,
3. counterexample archive (if violated),
4. remediation proof.

## 5. Native/C++/CUDA Boundary Register

| Boundary | Files | Risk | Mandatory mitigation |
|---|---|---|---|
| Python/C++ boundary | `torch/csrc/*` | critical | deterministic bridge fixtures + error-surface parity |
| CPU kernel boundary | `aten/native/cpu/*` | high | op-level differential fixtures |
| CUDA/device boundary | `aten/native/cuda/*`, `DeviceGuard*` | critical | stream/device guard conformance suites |
| autograd engine boundary | `autograd/engine.*` | critical | schedule and reentrancy stress corpus |
| serialization boundary | `serialization.cpp` | high | payload fuzz + round-trip fixtures |

## 6. Compatibility and Security Doctrine (Mode-Split)

Decision law (runtime):
`mode + graph_contract + risk_score + budget -> allow | full_validate | fail_closed`

| Threat | Strict mode | Hardened mode | Required ledger artifact |
|---|---|---|---|
| malformed checkpoint payload | fail-closed | fail-closed + bounded diagnostics | serialization incident ledger |
| invalid autograd graph state | fail invalid graph | quarantine+fail with trace | autograd validation report |
| dispatch confusion | fail ambiguous dispatch | fail ambiguous dispatch | dispatch audit report |
| in-place misuse | strict version checks | stricter guards + audit | version safety ledger |
| unknown checkpoint/version metadata | fail-closed | fail-closed | compatibility drift report |

## 7. Conformance Program (Exhaustive First Wave)

### 7.1 Fixture families

1. Tensor creation/view fixtures
2. Dispatch routing fixtures
3. Autograd backward/gradient fixtures
4. nn module forward/state fixtures
5. Optimizer-step fixtures
6. Checkpoint round-trip fixtures
7. Device/stream fixtures

### 7.2 Differential harness outputs (ft-conformance)

Each run emits:
- parity report,
- gradient-delta report,
- mismatch taxonomy,
- minimized repro fixture bundle,
- strict/hardened divergence report.

Release gate rule: critical-family drift => hard fail.

## 8. Extreme Optimization Program

Primary hotspots:
- dispatch fast path
- tensor iterator CPU kernels
- autograd scheduling path
- optimizer and checkpoint hot loops

Budgets (from spec section 17):
- elementwise forward p95 <= 200 ms
- backward p95 <= 1.35x forward baseline
- matmul p95 <= 240 ms
- optimizer step p95 <= 130 ms
- checkpoint throughput >= 350 MB/s
- p99 regression <= +7%, peak RSS regression <= +8%

Optimization governance:
1. baseline,
2. profile,
3. one lever,
4. conformance+gradient proof,
5. budget gate,
6. evidence commit.

## 9. RaptorQ-Everywhere Artifact Contract

Durable artifacts requiring RaptorQ sidecars:
- conformance fixture bundles,
- gradient and benchmark baselines,
- compatibility/risk ledgers.

Required envelope fields:
- source hash,
- symbol manifest,
- scrub status,
- decode proof chain.

## 10. Phase-2 Execution Backlog (Concrete)

1. Extract TensorImpl/storage invariants from `c10/core`.
2. Extract dispatch-key ordering and fallback rules.
3. Extract CPU op semantic contracts in staged waves with explicit closure to full operator coverage.
4. Extract autograd engine state transitions and queue invariants.
5. Extract module state/register semantics from nn core.
6. Extract checkpoint format semantics with closure path to full payload compatibility.
7. Extract device guard and stream contract behavior.
8. Build first differential fixture corpus for items 1-7.
9. Implement mismatch/gradient taxonomy in `ft-conformance`.
10. Add strict/hardened divergence report output.
11. Add adversarial payload fixtures (serialization + graph state).
12. Attach RaptorQ sidecar generation and decode-proof validation.

Definition of done for Phase-2:
- all section-3 rows have extraction artifacts,
- all seven fixture families runnable,
- comprehensive-spec G1-G6 gates trace to concrete outputs.

## 11. Residual Gaps and Risks

- `PROPOSED_ARCHITECTURE.md` crate map formatting contains literal `\n`; normalize before tooling automation.
- native boundary breadth is very large; avoid false confidence from tiny fixture subsets.
- autograd and device semantics are critical and must block release on unresolved drift.

## 12. Deep-Pass Hotspot Inventory (Measured)

Measured from `/data/projects/frankentorch/legacy_pytorch_code/pytorch`:
- file count: `21654`
- large concentration in `aten/src` (`2675` files), `torch/csrc` (`1850` files), and deep test surfaces

Top source hotspots by line count (first-wave extraction anchors):
1. `aten/src/ATen/native/SobolEngineOpsUtils.cpp` (`42459`)
2. `torch/testing/_internal/common_methods_invocations.py` (`27247`)
3. `test/test_export.py` (`18394`)
4. `test/inductor/test_torchinductor.py` (`17321`)
5. `test/test_jit.py` (`16279`)
6. `test/test_autograd.py` (`15622`)

Interpretation:
- semantics are spread across C++ core + Python test oracles,
- dispatch/autograd/serialization boundaries remain highest risk,
- conformance harness must sequence op coverage by waves while preserving explicit closure plans to full coverage.

## 13. Phase-2C Extraction Payload Contract (Per Ticket)

Each `FT-P2C-*` ticket MUST produce:
1. type and state inventory (tensor/storage/device/grad metadata),
2. dispatch route and fallback decision tables,
3. error and diagnostics contract map,
4. gradient/graph invariants for implemented ops plus closure criteria for remaining parity surface,
5. serialization format and version behavior notes,
6. strict/hardened split policy,
7. sequencing-boundary ledger with explicit parity-closure dependencies,
8. fixture mapping manifest,
9. optimization candidate and isomorphism risk note,
10. RaptorQ artifact declaration.

Artifact location (normative):
- `artifacts/phase2c/FT-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FT-P2C-00X/contract_table.md`
- `artifacts/phase2c/FT-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FT-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FT-P2C-00X/risk_note.md`

## 14. Strict/Hardened Compatibility Drift Budgets

Packet acceptance budgets:
- strict mode critical drift budget: `0`
- strict mode non-critical drift budget: `<= 0.10%`
- hardened divergence budget: `<= 1.00%` and only in allowlisted defensive classes
- unknown schema/device/serialization paths: fail-closed

Per-packet report requirements:
- `strict_parity`,
- `hardened_parity`,
- `gradient_drift_summary`,
- `dispatch_drift_summary`,
- `compatibility_drift_hash`.

## 15. Extreme-Software-Optimization Execution Law

Mandatory loop:
1. baseline,
2. profile,
3. one lever,
4. conformance + invariant replay,
5. re-baseline.

Primary sentinel workloads:
- dispatch-heavy operator mixes (`FT-P2C-002`, `FT-P2C-003`),
- autograd chain execution (`FT-P2C-004`),
- serialization checkpoint round-trip (`FT-P2C-006`),
- device transition workloads (`FT-P2C-007`).

Optimization scoring gate:
`score = (impact * confidence) / effort`, merge only if `score >= 2.0`.

## 16. RaptorQ Evidence Topology and Recovery Drills

Durable artifacts requiring sidecars:
- parity reports,
- gradient mismatch corpora,
- benchmark baselines,
- strict/hardened decision logs,
- serialization compatibility ledgers.

Naming convention:
- payload: `packet_<id>_<artifact>.json`
- sidecar: `packet_<id>_<artifact>.raptorq.json`
- proof: `packet_<id>_<artifact>.decode_proof.json`

Any decode-proof failure blocks packet promotion.

## 17. Phase-2C Exit Checklist (Operational)

Phase-2C is complete only when:
1. `FT-P2C-001..008` artifact packs exist and validate.
2. Every packet has strict and hardened fixture coverage.
3. Drift budgets from section 14 are met.
4. High-risk packets have at least one optimization proof artifact.
5. RaptorQ sidecars + decode proofs are scrub-clean.
6. Residual risks are explicitly assigned and tracked.
