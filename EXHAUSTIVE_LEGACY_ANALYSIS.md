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
- typical local mirror when present: `/data/projects/frankentorch/legacy_pytorch_code/pytorch`
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

## 18. Complexity, Performance, and Memory Characterization (`bd-3v0.23.6`)

### 18.1 Source-anchored operation complexity matrix

| Subsystem | Operation family | Time complexity class | Memory growth | Hotspot hypothesis | Anchors |
|---|---|---|---|---|---|
| Tensor metadata (`ft-core`) | shape/stride validation (`validate`), linear index projection (`storage_index_for`), contiguous stride synthesis | `O(rank)` per call | `O(1)` extra for validate/index; `O(rank)` for contiguous stride vector | high-frequency indexing and metadata checks can dominate small-op workloads before kernel dispatch starts | `crates/ft-core/src/lib.rs:85`, `crates/ft-core/src/lib.rs:158`, `crates/ft-core/src/lib.rs:393` |
| Dispatch (`ft-dispatch`) | keyset validation + type/backend selection + mode split routing | `O(k)` over priority lists (`k` = key classes, currently small/fixed) | `O(1)` | dispatch path is per-op and latency-sensitive; constant-factor regression here inflates all eager traces | `crates/ft-dispatch/src/lib.rs:123`, `crates/ft-dispatch/src/lib.rs:145`, `crates/ft-dispatch/src/lib.rs:264` |
| Autograd (`ft-autograd`) | backward scheduler (`compute_reachable` + `compute_dependencies` + ready-queue execution) | `O(V + E)` for graph traversal and dependency resolution | `O(V)` vectors (`reachable`, `pending`, `grads`) + queue/order telemetry | wide/deep DAGs amplify scheduler overhead and allocation churn, impacting backward p95/p99 tails | `crates/ft-autograd/src/lib.rs:273`, `crates/ft-autograd/src/lib.rs:393`, `crates/ft-autograd/src/lib.rs:418` |
| Serialization/durability (`ft-serialize`) | checkpoint normalize/hash/encode/decode and RaptorQ sidecar generation | normalize `O(n log n)` (entry sort), hash `O(n)`, parse/decode `O(payload_bytes)`, sidecar/decode-proof `O(symbol_count + decode_work)` | `O(n)` normalized entry copy + symbol buffers/proof artifacts | large checkpoints make normalization/symbolization dominant; decode-proof retries can inflate tail cost | `crates/ft-serialize/src/lib.rs:114`, `crates/ft-serialize/src/lib.rs:148`, `crates/ft-serialize/src/lib.rs:352` |
| Conformance harness (`ft-conformance`) | differential check synthesis/sort and e2e forensics emit | build checks `O(total_cases * comparators)`, canonical sort `O(C log C)`, e2e emit `O(L)` | `O(C)` check vector + `O(L)` forensic log vector | full-packet CI runs are sensitive to comparator volume and in-memory log accumulation before flush | `crates/ft-conformance/src/lib.rs:538`, `crates/ft-conformance/src/lib.rs:610`, `crates/ft-conformance/src/lib.rs:1375` |

### 18.2 Measurable hotspot probes and required observability

| Probe target | Minimal benchmark input envelope | Required observability fields | Expected artifact touchpoints |
|---|---|---|---|
| Dispatch fast path | packet `FT-P2C-002` differential/e2e suite with strict+hardened mode split cases | `suite_id`, `scenario_id`, `mode`, `reason_code`, `artifact_refs`, `replay_command` | `artifacts/phase2c/FT-P2C-002/parity_report.json`, e2e forensics matrix entries |
| Autograd scheduler tail | scheduler fixtures with branching DAG cases and reentrant-depth edges | `seed`, `scenario_id`, `mode`, `execution_order`, `queue_pushes`, `queue_pops` | `artifacts/phase2c/FT-P2C-004/parity_report.json`, scheduler telemetry logs |
| Serialization throughput + durability overhead | checkpoint encode/decode + sidecar/proof generation runs over representative entry counts | `scenario_id`, `mode`, `artifact_refs`, `reason_code`, decode proof hash fields | `artifacts/phase2c/FT-P2C-006/parity_report.json`, sidecar/proof artifacts |
| Harness scaling | full differential report and e2e matrix generation across enabled packets | `suite_id`, `scenario_id`, `packet_id`, `outcome`, `replay_command` | `artifacts/phase2c/conformance/differential_report_v1.json`, `artifacts/phase2c/e2e_forensics/e2e_matrix.jsonl` |

### 18.3 Optimization-risk notes (non-regression doctrine)

- Dispatch optimization risk: key-resolution caching that omits `mode` or full keyset bits can merge strict/hardened behavior and silently break policy semantics.
- Autograd optimization risk: parallel/reordered queue processing can alter deterministic execution order and gradient accumulation invariants (`FT-I3` outranks speed gains).
- Serialization optimization risk: changing entry-normalization or hash inputs breaks deterministic replay and sidecar proof stability (`FT-I5`/`FT-I6`).
- Conformance optimization risk: removing canonical sort or mutating comparator order can make drift reports non-deterministic and invalidate replay/audit workflows.

### 18.4 Evidence traceability linkage (docs-pass N/A rule)

This bead is docs/planning-only and does not itself change executable behavior.

Execution evidence remains owned by implementation beads:
- unit/property evidence: `bd-3v0.12.5`, `bd-3v0.13.5`, `bd-3v0.14.5`, `bd-3v0.15.5`, `bd-3v0.17.5`
- differential/metamorphic/adversarial evidence: `bd-3v0.12.6`, `bd-3v0.13.6`, `bd-3v0.14.6`, `bd-3v0.15.6`, `bd-3v0.17.6`
- e2e/logging evidence: `bd-3v0.12.7`, `bd-3v0.13.7`, `bd-3v0.14.7`, `bd-3v0.15.7`, `bd-3v0.17.7`

## 19. Pass-B Behavior and Invariant Expansion (`bd-3v0.23.12`)

### 19.1 Packet-level behavior obligations with strict/hardened semantics

| Packet | Behavioral core | Strict mode contract | Hardened mode contract | Invariant families | Source anchors | Evidence outputs |
|---|---|---|---|---|---|---|
| `FT-P2C-001` tensor metadata | shape/stride/storage offset validity, index projection, alias/version safety preconditions | invalid shape/index/offset fails closed with typed `TensorMetaError` | same fail-closed behavior (no automatic repair/coercion) | `FT-I1`, `FT-I4` | `crates/ft-core/src/lib.rs:61`, `crates/ft-core/src/lib.rs:85`, `crates/ft-core/src/lib.rs:158` | `artifacts/phase2c/FT-P2C-001/parity_report.json` + tensor_meta e2e logs |
| `FT-P2C-002` dispatch key model | deterministic key selection, mode split on composite/backend fallback, malformed-key rejection | composite/backend fallback rejected (`IncompatibleSet`), unknown/invalid keysets fail closed | bounded composite/backend fallback only when policy branch explicitly allows it | `FT-I2` | `crates/ft-dispatch/src/lib.rs:145`, `crates/ft-dispatch/src/lib.rs:264`, `crates/ft-dispatch/src/lib.rs:280` | `artifacts/phase2c/FT-P2C-002/parity_report.json`, hardened allowlist artifacts |
| `FT-P2C-004` autograd scheduler | reachable graph discovery, dependency counting, deterministic ready-queue execution | reentrant depth overflow fails closed; dependency underflow fails closed | bounded reentrant fallback path with explicit telemetry (`reentrant_guard_triggered`) | `FT-I3` | `crates/ft-autograd/src/lib.rs:273`, `crates/ft-autograd/src/lib.rs:393`, `crates/ft-autograd/src/lib.rs:418` | `artifacts/phase2c/FT-P2C-004/parity_report.json`, scheduler e2e logs |
| `FT-P2C-006` serialization + durability | deterministic checkpoint envelope/hash, strict decode gate, sidecar/proof generation | unknown fields/version/hash mismatch fail closed; decode proof failures block readiness | malformed JSON diagnostics allowed, but unknown/incompatible schema still fail closed | `FT-I5`, `FT-I6` | `crates/ft-serialize/src/lib.rs:114`, `crates/ft-serialize/src/lib.rs:128`, `crates/ft-serialize/src/lib.rs:148`, `crates/ft-serialize/src/lib.rs:352` | `artifacts/phase2c/FT-P2C-006/parity_report.json`, `*.raptorq.json`, `*.decode_proof.json` |

### 19.2 Evidence-carrying API/runtime behavior contract

Behavioral contract from API entrypoints through runtime evidence ledger:
- session operations (`add`, `mul`) must append dispatch evidence entries including key and fallback context.
- backward execution must append scheduler telemetry summaries with queue push/pop and reentrant guard fields.
- mode transitions must emit policy events so strict/hardened branch selection is reconstructible in forensic replay.

Anchors:
- `crates/ft-api/src/lib.rs:36`
- `crates/ft-api/src/lib.rs:57`
- `crates/ft-api/src/lib.rs:93`
- `crates/ft-runtime/src/lib.rs:31`
- `crates/ft-runtime/src/lib.rs:77`

### 19.3 High-risk edge-case semantics (expanded)

| Edge-case family | Trigger | Expected behavior | Failure class | Required regression/e2e hook |
|---|---|---|---|---|
| rank/stride mismatch and offset overflow | malformed tensor metadata fixture | fail closed before dispatch/autograd | `TensorMetaError::{RankStrideMismatch,StrideOverflow,StorageOffsetOverflow}` | tensor_meta invalid-case differential + packet `FT-P2C-001` e2e slice |
| unknown dispatch key bits | invalid key token in fixture or parse path | deterministic reject with explicit error reason | `DispatchKeyError::UnknownBits` or parse failure | dispatch adversarial comparator + replay command in logs |
| autograd depth overflow | reentrant depth exceeds configured max | strict rejects; hardened may continue only via bounded fallback branch | `AutogradError::ReentrantDepthExceeded` | scheduler mode-split differential and e2e reentrant scenario |
| checkpoint schema/hash drift | unknown field, version mismatch, checksum mismatch | decode rejected; no silent coercion | `SerializeError::{UnknownField,VersionMismatch,ChecksumMismatch}` | serialization differential + durability validator run |
| sidecar/decode proof mismatch | decode candidate cannot prove payload recovery | fail closed, mark artifact not ready | `SerializeError::RaptorQFailure` | packet sidecar/proof regeneration + validator replay |

## 20. Security/Compatibility and Failure-Semantics Integration

### 20.1 Threat-to-control matrix with artifact references

| Threat class | Control mechanism | Mode semantics | Artifact/evidence refs |
|---|---|---|---|
| gradient corruption via nondeterministic scheduling | dependency accounting + deterministic queue execution + telemetry | strict/hardened both require dependency closure; hardened only changes reentrant overflow handling | `artifacts/phase2c/FT-P2C-004/parity_report.json`, e2e scheduler logs |
| dispatch confusion/fallback misuse | keyset validation + explicit mode split + allowlisted hardened drift IDs | strict forbids fallback; hardened fallback is explicit, traceable, and scoped | `artifacts/phase2c/FT-P2C-002/parity_report.json`, `artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json` |
| serialization replay inconsistency | normalized entries + deterministic hash + strict schema/version checks | strict/hardened both fail closed on incompatible schema/hash | `artifacts/phase2c/FT-P2C-006/parity_report.json`, durability proofs |
| evidence loss/corruption | sidecar generation + scrub + decode proof verification | both modes block promotion on decode-proof failure | `packet_<id>_<artifact>.raptorq.json`, `packet_<id>_<artifact>.decode_proof.json` |

### 20.2 Failure lifecycle and recovery obligations

Recovery semantics are explicit and bounded:
1. detect failure with typed error/reason code,
2. emit structured forensic record (`scenario_id`, `mode`, `reason_code`, `artifact_refs`, `replay_command`),
3. fail closed unless hardened policy explicitly allows bounded fallback,
4. require deterministic replay command before allowlist or closure decisions.

Primary anchors:
- `crates/ft-conformance/src/lib.rs:610`
- `crates/ft-conformance/src/lib.rs:1375`
- `crates/ft-conformance/src/lib.rs:2191`
- `crates/ft-conformance/src/lib.rs:2352`
- `crates/ft-conformance/src/logging.rs:11`

## 21. Subsystem Test/E2E/Logging Expectations (Explicit)

| Subsystem | Unit/property minimum | Differential/metamorphic/adversarial minimum | E2E minimum | Mandatory forensic fields |
|---|---|---|---|---|
| tensor metadata | property coverage over shape/stride/index invalid and boundary cases | oracle + metamorphic offset-shift checks, fail-closed checks for invalid fixtures | packet-filtered `FT-P2C-001` matrix with valid/invalid scenario IDs | `scenario_id`, `mode`, `reason_code`, `artifact_refs`, `replay_command` |
| dispatch | key-priority/mode-split unit assertions, unknown-key and incompatible-set negatives | oracle output comparison + commutativity metamorphic check + adversarial unknown/autograd-without-cpu rejection checks | packet-filtered `FT-P2C-002` matrix including fallback branch evidence | `scenario_id`, `mode`, `fallback_used`, `reason_code`, `replay_command` |
| autograd scheduler | dependency ordering and reentrant policy unit/property coverage | oracle compare for output/grads + hardened policy comparator checks | packet-filtered `FT-P2C-004` matrix with telemetry replay | `seed`, `execution_order`, `queue_pushes`, `queue_pops`, `replay_command` |
| serialization/durability | decode strict/hardened negative tests + sidecar determinism checks | oracle/schema/hash parity checks + adversarial malformed payload/recovery checks | packet-filtered `FT-P2C-006` matrix + sidecar/decode-proof validation runs | `scenario_id`, `mode`, `artifact_refs`, `reason_code`, proof hash |

Execution entrypoints:
- `crates/ft-conformance/src/lib.rs:422`
- `crates/ft-conformance/src/lib.rs:449`
- `crates/ft-conformance/src/lib.rs:476`
- `crates/ft-conformance/src/lib.rs:503`
- `crates/ft-conformance/src/lib.rs:538`
- `crates/ft-conformance/src/lib.rs:1423`

## 22. Contradiction Check vs Pass-A Structure Doc

Pass-B consistency review against `EXISTING_PYTORCH_STRUCTURE.md` (sections 14-23):

| Check area | Pass-A anchor | Pass-B status | Outcome |
|---|---|---|---|
| dispatch mode-split semantics | workflow/state-machine and edge-case tables | preserved and expanded with complexity + risk coupling | aligned |
| autograd ordering and reentrant policy | ordering/lifecycle and error taxonomy sections | preserved and expanded with explicit recovery obligations | aligned |
| serialization fail-closed doctrine | edge-case and threat narrative sections | preserved and expanded with durability proof gating | aligned |
| logging/replay schema requirements | invariant and crosswalk sections | preserved with explicit subsystem minimums and failure lifecycle steps | aligned |
| docs-only N/A gate linkage | cross-cutting validation note | preserved and explicitly restated in this pass | aligned |

No direct contradictions detected in current source anchors. Any future behavior drift requires updating both documents in the same bead series to avoid split-brain planning artifacts.

## 23. Cross-Cutting Validation Gate Note (Pass B)

This pass (`bd-3v0.23.12`) is docs/planning only and does not introduce executable behavior changes.

Execution evidence remains delegated to implementation/conformance beads:
- unit/property evidence: `bd-3v0.12.5`, `bd-3v0.13.5`, `bd-3v0.14.5`, `bd-3v0.15.5`, `bd-3v0.17.5`
- differential/metamorphic/adversarial evidence: `bd-3v0.12.6`, `bd-3v0.13.6`, `bd-3v0.14.6`, `bd-3v0.15.6`, `bd-3v0.17.6`
- e2e/logging evidence: `bd-3v0.12.7`, `bd-3v0.13.7`, `bd-3v0.14.7`, `bd-3v0.15.7`, `bd-3v0.17.7`

## 24. Red-Team Review Attachment (`bd-3v0.23.13`)

Independent contradiction/completeness review results are attached in:
- `EXISTING_PYTORCH_STRUCTURE.md` section `24. Red-Team Contradiction and Completeness Review`

Review disposition summary:
- traceability drift in docs-pass listing was corrected,
- no doctrine-breaking contradictions were found in strict/hardened behavior statements,
- remaining completeness gaps are explicitly tracked via `GAP-UX-*` and packet follow-up beads (not silently accepted).

## 25. Behavior-Specialist Attachment (`bd-3v0.23.16`)

Specialist deep-dive output is attached in:
- `EXISTING_PYTORCH_STRUCTURE.md` section `25. Behavior-Specialist Deep Dive`

Handoff rule for integration passes:
- treat section-25 drift-sensitive semantics and rewrite prerequisites as mandatory merge-check inputs for `bd-3v0.23.14` and `bd-3v0.23.17`.

## 26. Risk/Perf/Test Specialist Attachment (`bd-3v0.23.17`)

Specialist pass output is attached in:
- `EXISTING_PYTORCH_STRUCTURE.md` section `26. Risk/Perf/Test Specialist Deep Dive`

Integration gate:
- section-26 ambiguity register and benchmark/e2e realism prerequisites are mandatory inputs for final integrated rewrite sign-off (`bd-3v0.23.14`).

## 27. Final Integration Sign-off Attachment (`bd-3v0.23.14`)

Integrated sweep/sign-off output is attached in:
- `EXISTING_PYTORCH_STRUCTURE.md` section `27. Final Integrated Consistency Sweep + Sign-off`

Sign-off interpretation for this document:
- Phase-2C docs overhaul is planning-complete for current scope,
- no unresolved critical doc contradictions remain,
- unresolved execution risks remain explicitly tracked (`bd-3v0.13.6`, `GAP-UX-*`, `RPT-*`) and are not considered closed.
