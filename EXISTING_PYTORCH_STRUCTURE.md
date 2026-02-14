# EXISTING_PYTORCH_STRUCTURE

Doc-overhaul baseline matrix:
- `artifacts/phase2c/DOC_PASS00_BASELINE_GAP_MATRIX_V1.md`

This pass is the topology/cartography closure for bead `bd-3v0.23.2`:
- full module/package map
- ownership boundaries
- dependency direction and layering constraints
- e2e-critical control paths
- ambiguity ledger with owners and follow-up beads

## 0. Scope and Method

Scope:
- active Rust workspace in this repo (`Cargo.toml:1`)
- legacy PyTorch oracle topology used for parity extraction (`legacy_pytorch_code/pytorch`)
- conformance/evidence/artifact plumbing that ties them together

Method:
- docs-first constraints from `AGENTS.md` and root `README.md`
- source-anchored mapping from crate code, tests, and conformance binaries
- explicit links from module boundaries to test/evidence responsibilities

Primary source anchors used in this pass:
- `Cargo.toml:1`
- `README.md:7`
- `README.md:46`
- `AGENTS.md:49`
- `AGENTS.md:89`
- `crates/ft-core/src/lib.rs:27`
- `crates/ft-dispatch/src/lib.rs:264`
- `crates/ft-autograd/src/lib.rs:183`
- `crates/ft-api/src/lib.rs:8`
- `crates/ft-runtime/src/lib.rs:56`
- `crates/ft-serialize/src/lib.rs:114`
- `crates/ft-conformance/src/lib.rs:28`
- `crates/ft-conformance/src/lib.rs:538`
- `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:75`

## 1. Project Purpose and Non-Negotiable Constraints

Project purpose:
- clean-room Rust reimplementation targeting full drop-in PyTorch compatibility (`README.md:7`, `README.md:11`)
- crown-jewel invariant is DAC: deterministic, replayable autograd with provenance-complete evidence (`README.md:21`, `AGENTS.md:76`)

Global constraints that directly shape architecture:
- strict/hardened mode split must preserve API contract (`AGENTS.md:107`)
- unknown incompatible features must fail closed (`AGENTS.md:131`)
- autograd correctness outranks raw kernel speed (`AGENTS.md:85`)
- optimization is profile-first with behavior-isomorphism proof requirements (`AGENTS.md:155`)
- long-lived artifacts require RaptorQ sidecars and decode-proof evidence (`AGENTS.md:137`)

## 2. Active Rust Workspace Cartography

### 2.1 Crate inventory and responsibility boundaries

Workspace members:
- `ft-core`
- `ft-dispatch`
- `ft-kernel-cpu`
- `ft-autograd`
- `ft-device`
- `ft-serialize`
- `ft-api`
- `ft-conformance`
- `ft-runtime`

Source anchor:
- `Cargo.toml:3`

| Crate | Layer role | Owns | Must not own | Anchor(s) |
|---|---|---|---|---|
| `ft-core` | Data model substrate | tensor metadata/value/version primitives (`TensorMeta`, `ScalarTensor`, compatibility checks) | dispatch policy, scheduler orchestration, fixture IO | `crates/ft-core/src/lib.rs:27`, `crates/ft-core/src/lib.rs:287`, `crates/ft-core/src/lib.rs:372` |
| `ft-kernel-cpu` | CPU kernel impl | scalar add/mul kernel behavior, compatibility error wrapping | dispatch key policy, gradient graph logic | `crates/ft-kernel-cpu/src/lib.rs:28`, `crates/ft-kernel-cpu/src/lib.rs:33` |
| `ft-dispatch` | Dispatch policy/router | dispatch keys, keyset algebra, priority rules, strict/hardened fallback gates, dispatch decision metadata | tensor metadata storage model, checkpoint codec | `crates/ft-dispatch/src/lib.rs:16`, `crates/ft-dispatch/src/lib.rs:43`, `crates/ft-dispatch/src/lib.rs:264` |
| `ft-autograd` | Backward graph engine | node tape, deterministic dependency scheduling, gradient accumulation, reentrant guard policy | fixture orchestration, artifact file writing | `crates/ft-autograd/src/lib.rs:183`, `crates/ft-autograd/src/lib.rs:273`, `crates/ft-autograd/src/lib.rs:441` |
| `ft-runtime` | Policy/evidence context | mode state, evidence ledger, durability envelope and decode-proof metadata | op routing, gradient math, fixture parsing | `crates/ft-runtime/src/lib.rs:21`, `crates/ft-runtime/src/lib.rs:56`, `crates/ft-runtime/src/lib.rs:108` |
| `ft-api` | User-facing composition layer | session facade binding `Tape` + runtime ledger, operation/backward API | low-level keyset math, checkpoint codecs | `crates/ft-api/src/lib.rs:8`, `crates/ft-api/src/lib.rs:57`, `crates/ft-api/src/lib.rs:93` |
| `ft-device` | Device boundary guard | device consistency checks and guard contract | dispatch fallback decisions, artifact/fixture handling | `crates/ft-device/src/lib.rs:25`, `crates/ft-device/src/lib.rs:52` |
| `ft-serialize` | Snapshot/durability codec | checkpoint envelope encode/decode, strict/hardened decode policy, RaptorQ sidecar/proof generation | autograd scheduler logic, dispatch key policy | `crates/ft-serialize/src/lib.rs:114`, `crates/ft-serialize/src/lib.rs:128`, `crates/ft-serialize/src/lib.rs:148` |
| `ft-conformance` | Differential and evidence harness | fixture loading, local-vs-legacy differential checks, e2e forensics JSONL, packet artifact validation | core tensor semantics themselves, kernel execution internals | `crates/ft-conformance/src/lib.rs:28`, `crates/ft-conformance/src/lib.rs:610`, `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:75` |

### 2.2 Dependency direction (implemented, not aspirational)

Observed internal dependency graph (from `cargo metadata`):
- `ft-core` has no internal crate deps (base layer)
- `ft-kernel-cpu -> ft-core`
- `ft-dispatch -> ft-core, ft-kernel-cpu`
- `ft-autograd -> ft-core, ft-dispatch`
- `ft-runtime -> ft-core` (+ optional `asupersync`, `ftui`)
- `ft-api -> ft-autograd, ft-core, ft-runtime`
- `ft-device -> ft-core`
- `ft-serialize -> asupersync, serde, serde_json`
- `ft-conformance -> ft-api, ft-core, ft-autograd, ft-dispatch, ft-serialize`

Source anchors:
- `Cargo.toml:20`
- `crates/ft-runtime/Cargo.toml:6`
- `crates/ft-conformance/Cargo.toml:6`

### 2.3 Layering constraints (current enforcement intent)

L0: data primitives
- `ft-core`

L1: execution primitives
- `ft-kernel-cpu`, `ft-dispatch`, `ft-device`

L2: graph/runtime composition
- `ft-autograd`, `ft-runtime`

L3: public API
- `ft-api`

L4: verification/evidence tooling
- `ft-conformance` + `ft-conformance` bins

Allowed direction:
- `L0 -> L1 -> L2 -> L3 -> L4` (plus `L4` reading legacy oracle files/scripts)

Prohibited direction (policy intent):
- no lower layer imports from higher layer test/harness crates
- no `ft-core` dependency on `ft-dispatch`/`ft-autograd`
- no `ft-autograd` dependency on `ft-conformance`
- no runtime policy decisions embedded in kernel crate

## 3. Core Module-Level Topology and Data Flow

### 3.1 Primary control path: local execution

1. session starts with mode and fresh tape/ledger:
   - `FrankenTorchSession::new` (`crates/ft-api/src/lib.rs:13`)
2. variable creation allocates leaf node:
   - `FrankenTorchSession::variable` -> `Tape::leaf` (`crates/ft-api/src/lib.rs:31`, `crates/ft-autograd/src/lib.rs:197`)
3. op execution routes through dispatch:
   - `FrankenTorchSession::add/mul` -> `Tape::add/mul` (`crates/ft-api/src/lib.rs:36`, `crates/ft-autograd/src/lib.rs:212`)
   - `dispatch_scalar_binary_with_keyset` resolves key and fallback policy (`crates/ft-dispatch/src/lib.rs:264`)
   - `ft-kernel-cpu` computes scalar output (`crates/ft-kernel-cpu/src/lib.rs:28`)
4. evidence entry emitted on every operation:
   - `FrankenTorchSession::record_operation` (`crates/ft-api/src/lib.rs:93`)
5. backward pass uses deterministic ready queue with dependency accounting:
   - `Tape::backward_with_options` (`crates/ft-autograd/src/lib.rs:273`)
   - `compute_reachable` + `compute_dependencies` + `complete_dependency` (`crates/ft-autograd/src/lib.rs:393`, `crates/ft-autograd/src/lib.rs:418`, `crates/ft-autograd/src/lib.rs:441`)
6. backward evidence is recorded in runtime ledger:
   - `FrankenTorchSession::backward_with_options` (`crates/ft-api/src/lib.rs:57`)

### 3.2 Policy split points (strict vs hardened)

Dispatch split:
- strict rejects composite/backend fallback route (`crates/ft-dispatch/src/lib.rs:277`)
- hardened permits bounded fallback and marks `fallback_used` (`crates/ft-dispatch/src/lib.rs:287`)

Autograd reentrant split:
- strict can fail on reentrant depth excess
- hardened uses bounded fallback policy
- anchor: `BackwardOptions::for_mode` and guard handling (`crates/ft-autograd/src/lib.rs:54`, `crates/ft-autograd/src/lib.rs:284`)

Serialization split:
- strict mode uses strict schema decode
- hardened mode provides bounded diagnostics but still fail-closed for unknown fields/version/checksum drift
- anchors: `decode_checkpoint`, `decode_checkpoint_strict`, `decode_checkpoint_hardened`, `validate_checkpoint` (`crates/ft-serialize/src/lib.rs:128`, `crates/ft-serialize/src/lib.rs:279`, `crates/ft-serialize/src/lib.rs:293`, `crates/ft-serialize/src/lib.rs:328`)

### 3.3 Evidence and durability topology

Runtime evidence:
- `EvidenceLedger` stores policy/dispatch/backward/durability summaries (`crates/ft-runtime/src/lib.rs:21`)
- `RuntimeContext` centralizes mode + ledger (`crates/ft-runtime/src/lib.rs:56`)

Durability evidence:
- checkpoint sidecars + decode proof from `generate_raptorq_sidecar` (`crates/ft-serialize/src/lib.rs:148`)
- packet-sidecar emitter writes `parity_report.raptorq.json` and decode proof files (`crates/ft-conformance/src/bin/emit_packet_sidecar.rs:11`)
- packet artifact validator enforces required files/keys (`crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:75`)

## 4. Legacy Oracle Cartography (PyTorch -> FrankenTorch ownership)

Legacy root:
- `/data/projects/frankentorch/legacy_pytorch_code/pytorch`
- upstream: `pytorch/pytorch`

| Legacy package zone | Observable contract class | FrankenTorch owner crate(s) | Current evidence harness linkage | Lifecycle note |
|---|---|---|---|---|
| `c10/core/TensorImpl.h`, tensor metadata internals | shape/stride/storage/version invariants | `ft-core` | tensor-meta fixture and packet artifacts (`crates/ft-conformance/fixtures/tensor_meta_cases.json`, `artifacts/phase2c/FT-P2C-001/*`) | active first-wave extraction |
| `c10/core/DispatchKey*.h`, `ATen/Dispatch.h` | key precedence and backend selection | `ft-dispatch` | dispatch fixture + differential checks (`crates/ft-conformance/fixtures/dispatch_key_cases.json`) | active first-wave extraction |
| `aten/src/ATen/native/*` (CPU subset first) | op kernel semantics | `ft-kernel-cpu` + `ft-dispatch` | scalar fixture path (`crates/ft-conformance/fixtures/scalar_autograd_cases.json`) | currently scalar-only slice |
| `torch/csrc/autograd/*` | backward scheduler, reentrant behavior, gradient accumulation | `ft-autograd` | scheduler fixture + differential checks (`crates/ft-conformance/fixtures/autograd_scheduler_cases.json`) | active first-wave extraction |
| `torch/csrc/serialization.cpp` + related state surfaces | checkpoint compatibility and recoverability | `ft-serialize` | serialization fixture + sidecar proof checks (`crates/ft-conformance/fixtures/serialization_cases.json`) | active first-wave extraction |
| device guard/stream boundaries (`c10`, `torch/csrc/{cuda,mps,xpu}`) | device safety and transition semantics | `ft-device` (+ future backend crates) | currently unit-only for device guard (`crates/ft-device/src/lib.rs:64`) | partial; broader backend parity remains open |
| Python bridge and full frontend API (`torch/csrc`, Python `torch/*`) | user-visible API and bridge semantics | future expansion beyond current scalar session | differential harness invokes legacy python scripts (`crates/ft-conformance/src/lib.rs:2224`) | major closure wave, not completed |

## 5. Boundary-to-Test and Evidence Responsibility Matrix

| Boundary owner | Unit/property anchor | Differential/adversarial anchor | E2E/logging anchor | Artifact/gate anchor |
|---|---|---|---|---|
| `ft-core` tensor metadata and versioning | `crates/ft-core/src/lib.rs:405` | `run_tensor_meta_case` and tensor-meta checks (`crates/ft-conformance/src/lib.rs:1477`) | structured case logs in matrix output (`crates/ft-conformance/src/lib.rs:538`) | `artifacts/phase2c/FT-P2C-001/parity_report.json` |
| `ft-dispatch` key model and fallback | `crates/ft-dispatch/src/lib.rs:329` | `run_dispatch_case` + packet `FT-P2C-002` checks (`crates/ft-conformance/src/lib.rs:1635`) | e2e matrix includes dispatch suite logs (`crates/ft-conformance/src/lib.rs:558`) | `artifacts/phase2c/FT-P2C-002/parity_report.json` |
| `ft-autograd` scheduler/reentrant policy | `crates/ft-autograd/src/lib.rs:461` | `run_scheduler_case` + legacy scheduler oracle compare (`crates/ft-conformance/src/lib.rs:1755`, `crates/ft-conformance/src/lib.rs:2194`) | e2e matrix includes scheduler logs (`crates/ft-conformance/src/lib.rs:561`) | `artifacts/phase2c/FT-P2C-004/parity_report.json` |
| `ft-serialize` checkpoint + sidecar proof | `crates/ft-serialize/src/lib.rs:425` | `run_serialization_case` (`crates/ft-conformance/src/lib.rs:1845`) | e2e matrix includes serialization logs (`crates/ft-conformance/src/lib.rs:564`) | `parity_report.raptorq.json` + `parity_report.decode_proof.json` |
| `ft-runtime` evidence ledger/policy mode | `crates/ft-runtime/src/lib.rs:170` | consumed indirectly by `ft-api` operation/backward evidence | log contract schema (`artifacts/phase2c/TEST_LOG_CONTRACT_V1.md`) | method-stack/durability docs under `artifacts/phase2c` |
| `ft-conformance` harness/reporting | `crates/ft-conformance/src/lib.rs:2525`, `crates/ft-conformance/tests/smoke.rs:1` | `run_differential_conformance` and allowlist loading (`crates/ft-conformance/src/lib.rs:610`, `crates/ft-conformance/src/lib.rs:2311`) | `emit_e2e_forensics_matrix_filtered` (`crates/ft-conformance/src/lib.rs:538`) | `run_differential_report` and packet validator bins (`crates/ft-conformance/src/bin/run_differential_report.rs:9`, `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:75`) |

## 6. E2E-Critical Control Paths

### Path A: API execution to deterministic backward evidence

Route:
- `ft-api` session init and ops (`crates/ft-api/src/lib.rs:13`)
- dispatch decision/kernels (`crates/ft-dispatch/src/lib.rs:264`, `crates/ft-kernel-cpu/src/lib.rs:28`)
- autograd scheduler/backward report (`crates/ft-autograd/src/lib.rs:273`)
- runtime evidence ledger append (`crates/ft-runtime/src/lib.rs:21`)

Failure sensitivity:
- any change to key priority (`TYPE_PRIORITY`) or dependency scheduler order can silently alter gradient behavior (`crates/ft-dispatch/src/lib.rs:43`, `crates/ft-autograd/src/lib.rs:69`)

### Path B: fixture execution to differential parity verdict

Route:
- load fixture families via `HarnessConfig::default_paths` (`crates/ft-conformance/src/lib.rs:38`)
- execute local behavior (`run_scalar_case`, `run_tensor_meta_case`, `run_dispatch_case`, `run_scheduler_case`, `run_serialization_case`)
- query legacy oracle script path (`crates/ft-conformance/src/lib.rs:2224`)
- classify drifts and apply allowlist policy (`crates/ft-conformance/src/lib.rs:1999`, `crates/ft-conformance/src/lib.rs:2311`)
- emit machine-readable report (`crates/ft-conformance/src/lib.rs:1350`)

Failure sensitivity:
- path assumptions for python executable and repo layout can flip parity checks to `oracle_unavailable` status (`crates/ft-conformance/src/lib.rs:2230`)

### Path C: conformance suites to e2e forensics JSONL

Route:
- run all suites for selected modes and gather `StructuredCaseLog` entries (`crates/ft-conformance/src/lib.rs:550`)
- optional packet filter to isolate one packet (`crates/ft-conformance/src/lib.rs:572`)
- emit jsonl for replay/forensics (`crates/ft-conformance/src/lib.rs:576`)
- CLI entry: `run_e2e_matrix` (`crates/ft-conformance/src/bin/run_e2e_matrix.rs:9`)

Failure sensitivity:
- schema drift in log object breaks downstream forensic tooling; schema anchored in `logging.rs` (`crates/ft-conformance/src/logging.rs:8`)

### Path D: parity report to RaptorQ durability gate

Route:
- generate sidecar/proof from parity payload (`crates/ft-serialize/src/lib.rs:148`)
- persist sidecar and proof JSON via `emit_packet_sidecar` (`crates/ft-conformance/src/bin/emit_packet_sidecar.rs:11`)
- validate packet artifact completeness and schema keys (`crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:75`)

Failure sensitivity:
- missing sidecar/proof or malformed keys blocks packet readiness

## 7. Hidden Couplings and Compatibility Pockets

1. Repo-layout coupling in harness defaults:
   - `HarnessConfig::default_paths` hardcodes repo-relative locations for fixtures, allowlist, and legacy oracle root (`crates/ft-conformance/src/lib.rs:38`)
   - impact: running from non-standard checkout layout can degrade checks

2. Packet-id coupling embedded in differential checks:
   - packet strings such as `FT-P2C-001`, `FT-P2C-002`, `FT-P2C-004`, `FT-P2C-006` are embedded in report records/evidence refs
   - impact: packet renames require coordinated refactors across fixtures and validators

3. Scalar-only execution surface:
   - `DType::F64` and `Device::Cpu` are the only concrete variants in current core model (`crates/ft-core/src/lib.rs:11`, `crates/ft-core/src/lib.rs:16`)
   - impact: current crate boundaries are sound, but parity breadth is intentionally incomplete and must close via packet waves

4. Dispatch fallback policy tightly coupled to mode semantics:
   - strict mode forbids composite fallback, hardened mode allows bounded fallback (`crates/ft-dispatch/src/lib.rs:277`)
   - impact: mode changes can alter behavior even when kernels are unchanged; differential gates must remain mandatory

5. Legacy-oracle dependency on external python runtime:
   - oracle script invocation uses subprocess execution and serialized I/O (`crates/ft-conformance/src/lib.rs:2224`)
   - impact: differential confidence depends on reproducible python environment

6. Artifact validator uses mixed schema checks:
   - JSON key checks are strict; YAML checks rely on key-line presence for selected sections (`crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:493`)
   - impact: possible false negatives/positives if parity gate YAML formatting shifts significantly

## 8. Boundary Ambiguity Ledger (with owners)

| Ambiguity | Why it matters | Proposed owner | Follow-up bead(s) |
|---|---|---|---|
| Target crate list in `AGENTS.md` does not match currently implemented workspace crate set | confuses long-term layering and ownership expectations | architecture/docs owner | `bd-3v0.23.3`, `bd-3v0.23.5` |
| No explicit crate-level ownership file for each packet beyond artifacts | increases handoff friction across agents and waves | packet leads + docs pass owners | `bd-3v0.23.3`, `bd-3v0.23.4` |
| Device backend expansion boundaries (CPU-only today) are not yet codified in compatibility matrix granularity | risk of implicit drift when CUDA/MPS/XPU land | `ft-device` + compatibility/security track | `bd-3v0.23.9`, `bd-3v0.7` |
| Legacy oracle dependency contract (python version/env) is implicit | can yield non-deterministic oracle availability outcomes | conformance owner | `bd-3v0.23.10`, `bd-3v0.20` |
| Artifact validator checks some YAML semantics structurally but not fully schema-validated | potential silent acceptance of semantically invalid gate docs | conformance tooling owner | `bd-3v0.23.8` |

## 9. Sequencing Boundary (No Permanent Exclusions)

Immediate implemented slice (validated by code and fixtures):
- tensor metadata/storage invariants (`ft-core`)
- scalar CPU dispatch + kernels (`ft-dispatch`, `ft-kernel-cpu`)
- deterministic scalar autograd scheduling (`ft-autograd`)
- checkpoint + sidecar proof primitives (`ft-serialize`)
- conformance harness with strict+hardened mode runs (`ft-conformance`)

Mandatory closure waves still open:
- tensorized operators beyond scalar
- broad dtype/backend matrix
- expanded autograd surfaces and bridge behavior
- full nn/optimizer and broader serialization/API parity

Evidence anchor examples:
- packet artifacts under `artifacts/phase2c/FT-P2C-00X/*`
- differential report output via `run_differential_report`
- e2e forensic matrix output via `run_e2e_matrix`

## 10. Next Actions for Downstream Doc Passes

Required for `bd-3v0.23.4` (state/invariant mapping):
- formal state transition tables for tensor metadata, dispatch decision states, autograd queue/dependency states, and serialization decode/recovery states

Required for `bd-3v0.23.5` (execution-path narratives):
- deepen each e2e-critical path above with failure branches and replay commands from structured logs

Required for `bd-3v0.23.10` (unit/e2e crosswalk):
- attach every high-regression symbol below to explicit scenario IDs, fixture IDs, and replay commands

## 11. Symbol/API Census and Surface Classification (`bd-3v0.23.3`)

### 11.1 Surface classes used in this census

- `S0-user-contract`: intended user-facing API surface
- `S1-cross-crate-contract`: public for workspace composition; not yet stable as external API
- `S2-harness-tooling-contract`: public for conformance/reporting toolchain
- `I0-internal-only`: non-`pub` implementation details; excluded from this public symbol census

Compatibility obligations:
- `S0-user-contract`: strict backward compatibility once V1 lock is declared
- `S1-cross-crate-contract`: compatibility required for packet waves and conformance harness expectations
- `S2-harness-tooling-contract`: compatibility required for artifact/replay pipelines and CI gates

### 11.2 Complete exported symbol census (top-level public symbols)

`ft-api` (`S0-user-contract` plus alias bridge):
- `FrankenTorchSession` (`crates/ft-api/src/lib.rs:8`)
- `DacBackwardOptions` (alias) (`crates/ft-api/src/lib.rs:113`)
- `DacBackwardReport` (alias) (`crates/ft-api/src/lib.rs:113`)
- `DacNodeId` (alias) (`crates/ft-api/src/lib.rs:113`)
- `DacReentrantPolicy` (alias) (`crates/ft-api/src/lib.rs:113`)

`ft-core` (`S1-cross-crate-contract`):
- `DType` (`crates/ft-core/src/lib.rs:11`)
- `Device` (`crates/ft-core/src/lib.rs:16`)
- `ExecutionMode` (`crates/ft-core/src/lib.rs:21`)
- `TensorMeta` (`crates/ft-core/src/lib.rs:27`)
- `TensorMetaError` (`crates/ft-core/src/lib.rs:207`)
- `TensorCompatError` (`crates/ft-core/src/lib.rs:266`)
- `ScalarTensor` (`crates/ft-core/src/lib.rs:287`)
- `ensure_compatible` (`crates/ft-core/src/lib.rs:372`)
- `contiguous_strides` (`crates/ft-core/src/lib.rs:391`)

`ft-dispatch` (`S1-cross-crate-contract`):
- `BinaryOp` (`crates/ft-dispatch/src/lib.rs:9`)
- `DispatchKey` (`crates/ft-dispatch/src/lib.rs:16`)
- `DispatchKeySet` (`crates/ft-dispatch/src/lib.rs:54`)
- `DispatchKeyError` (`crates/ft-dispatch/src/lib.rs:161`)
- `DispatchDecision` (`crates/ft-dispatch/src/lib.rs:191`)
- `DispatchOutcome` (`crates/ft-dispatch/src/lib.rs:202`)
- `DispatchError` (`crates/ft-dispatch/src/lib.rs:208`)
- `dispatch_keyset_for_tensors` (`crates/ft-dispatch/src/lib.rs:237`)
- `dispatch_scalar_binary` (`crates/ft-dispatch/src/lib.rs:253`)
- `dispatch_scalar_binary_with_keyset` (`crates/ft-dispatch/src/lib.rs:264`)

`ft-kernel-cpu` (`S1-cross-crate-contract`):
- `KernelError` (`crates/ft-kernel-cpu/src/lib.rs:8`)
- `add_scalar` (`crates/ft-kernel-cpu/src/lib.rs:28`)
- `mul_scalar` (`crates/ft-kernel-cpu/src/lib.rs:33`)

`ft-autograd` (`S1-cross-crate-contract`):
- `NodeId` (`crates/ft-autograd/src/lib.rs:11`)
- `ReentrantPolicy` (`crates/ft-autograd/src/lib.rs:28`)
- `BackwardOptions` (`crates/ft-autograd/src/lib.rs:34`)
- `SchedulerTelemetry` (`crates/ft-autograd/src/lib.rs:69`)
- `OperationEvent` (`crates/ft-autograd/src/lib.rs:122`)
- `BackwardStep` (`crates/ft-autograd/src/lib.rs:131`)
- `BackwardReport` (`crates/ft-autograd/src/lib.rs:138`)
- `AutogradError` (`crates/ft-autograd/src/lib.rs:157`)
- `Tape` (`crates/ft-autograd/src/lib.rs:183`)

`ft-runtime` (`S1-cross-crate-contract`):
- `EvidenceKind` (`crates/ft-runtime/src/lib.rs:6`)
- `EvidenceEntry` (`crates/ft-runtime/src/lib.rs:14`)
- `EvidenceLedger` (`crates/ft-runtime/src/lib.rs:21`)
- `RuntimeContext` (`crates/ft-runtime/src/lib.rs:56`)
- `ScrubStatus` (`crates/ft-runtime/src/lib.rs:94`)
- `DecodeProof` (`crates/ft-runtime/src/lib.rs:101`)
- `DurabilityEnvelope` (`crates/ft-runtime/src/lib.rs:108`)
- `asupersync_infinite_budget` (feature-gated) (`crates/ft-runtime/src/lib.rs:155`)
- `frankentui_default_style` (feature-gated) (`crates/ft-runtime/src/lib.rs:160`)

`ft-device` (`S1-cross-crate-contract`):
- `DeviceError` (`crates/ft-device/src/lib.rs:8`)
- `DeviceGuard` (`crates/ft-device/src/lib.rs:25`)
- `ensure_same_device` (`crates/ft-device/src/lib.rs:52`)

`ft-serialize` (`S1-cross-crate-contract`):
- `CHECKPOINT_SCHEMA_VERSION` (`crates/ft-serialize/src/lib.rs:14`)
- `RAPTORQ_SIDECAR_SCHEMA_VERSION` (`crates/ft-serialize/src/lib.rs:15`)
- `SnapshotEntry` (`crates/ft-serialize/src/lib.rs:19`)
- `CheckpointMode` (`crates/ft-serialize/src/lib.rs:27`)
- `DecodeMode` (`crates/ft-serialize/src/lib.rs:33`)
- `CheckpointEnvelope` (`crates/ft-serialize/src/lib.rs:40`)
- `RepairSymbolRecord` (`crates/ft-serialize/src/lib.rs:49`)
- `RaptorQSidecar` (`crates/ft-serialize/src/lib.rs:57`)
- `DecodeProofArtifact` (`crates/ft-serialize/src/lib.rs:72`)
- `SerializeError` (`crates/ft-serialize/src/lib.rs:82`)
- `encode_checkpoint` (`crates/ft-serialize/src/lib.rs:114`)
- `decode_checkpoint` (`crates/ft-serialize/src/lib.rs:128`)
- `encode_snapshot` (`crates/ft-serialize/src/lib.rs:139`)
- `decode_snapshot` (`crates/ft-serialize/src/lib.rs:143`)
- `generate_raptorq_sidecar` (`crates/ft-serialize/src/lib.rs:148`)

`ft-conformance` (`S2-harness-tooling-contract`):
- data/config/report types: `HarnessConfig`, `CaseReport`, `DispatchCaseReport`, `SchedulerCaseReport`, `SerializationCaseReport`, `TensorMetaCaseReport`, `HarnessReport`, `BenchReport`, `E2EForensicsSummary`, `DifferentialHarnessReport`, `LegacyOracleStatus`, `DifferentialCheck` (`crates/ft-conformance/src/lib.rs:28`)
- suite runners: `run_smoke`, `run_scalar_conformance`, `run_tensor_meta_conformance`, `run_dispatch_conformance`, `run_autograd_scheduler_conformance`, `run_serialization_conformance` (`crates/ft-conformance/src/lib.rs:336`)
- evidence/report emitters: `emit_e2e_forensics_matrix`, `emit_e2e_forensics_matrix_filtered`, `run_differential_conformance`, `emit_differential_report`, `run_scalar_microbench` (`crates/ft-conformance/src/lib.rs:530`)

`ft-conformance::logging` (`S2-harness-tooling-contract`):
- `STRUCTURED_LOG_SCHEMA_VERSION` (`crates/ft-conformance/src/logging.rs:8`)
- `StructuredCaseLog` (`crates/ft-conformance/src/logging.rs:11`)
- `mode_label` (`crates/ft-conformance/src/logging.rs:75`)
- `env_fingerprint` (`crates/ft-conformance/src/logging.rs:83`)

### 11.3 Public vs internal boundary statement

User-facing contract (current):
- only `ft-api::FrankenTorchSession` plus DAC alias exports are treated as user-facing (`S0`)

Engineering contract (cross-crate):
- `ft-core`, `ft-dispatch`, `ft-kernel-cpu`, `ft-autograd`, `ft-runtime`, `ft-device`, `ft-serialize` exports are public inside the workspace but not yet declared stable external API (`S1`)

Harness/tooling contract:
- `ft-conformance` exports are CI/evidence interfaces (`S2`) and should be versioned with log/artifact schema expectations

Internal-only excluded from census:
- non-public helper functions such as `run_legacy_oracle_script`, `query_legacy_*`, `classify_drift_status` remain `I0` and can change without external compatibility commitments (`crates/ft-conformance/src/lib.rs:1999`, `crates/ft-conformance/src/lib.rs:2224`)

## 12. High-Regression Symbol Registry and Evidence Expectations

| Symbol ID | Class | Risk tag | Compatibility obligation | Unit/property anchor | Differential/adversarial anchor | E2E/logging + replay expectation |
|---|---|---|---|---|---|---|
| `ft-api::FrankenTorchSession::backward_with_options` | `S0` | critical | backward semantics + evidence stability | `crates/ft-api/src/lib.rs:118` | scheduler + scalar suites (`crates/ft-conformance/src/lib.rs:392`, `crates/ft-conformance/src/lib.rs:476`) | must produce `StructuredCaseLog` with deterministic `scenario_id/seed` (`crates/ft-conformance/src/logging.rs:30`) |
| `ft-core::TensorMeta::from_shape_and_strides` | `S1` | high | fail-closed metadata validation | `crates/ft-core/src/lib.rs:405` | tensor-meta differential checks (`crates/ft-conformance/src/lib.rs:1477`) | e2e tensor-meta entries in jsonl matrix (`crates/ft-conformance/src/lib.rs:555`) |
| `ft-core::TensorMeta::storage_index_for` | `S1` | high | index/rank/oob invariants | `crates/ft-core/src/lib.rs:405` | tensor-meta comparator paths (`crates/ft-conformance/src/lib.rs:1535`) | replay via matrix command `run_e2e_matrix --packet FT-P2C-001` |
| `ft-core::ScalarTensor::alias_view` | `S1` | high | alias/storage/version semantics | `crates/ft-core/src/lib.rs:405` | tensor-meta + scalar differential surfaces | log reason codes must distinguish alias failures |
| `ft-dispatch::dispatch_scalar_binary_with_keyset` | `S1` | critical | strict/hardened fallback contract | `crates/ft-dispatch/src/lib.rs:329` | dispatch differential checks (`crates/ft-conformance/src/lib.rs:1635`) | dispatch case logs include packet `FT-P2C-002` and replay cmd |
| `ft-dispatch::DispatchKeySet::validate_for_scalar_binary` | `S1` | high | fail-closed unknown/incompatible keysets | `crates/ft-dispatch/src/lib.rs:329` | dispatch invalid-case comparisons | reason codes must separate keyset policy vs kernel failure |
| `ft-autograd::Tape::backward_with_options` | `S1` | critical | deterministic order + reentrant policy | `crates/ft-autograd/src/lib.rs:461` | scheduler differential path (`crates/ft-conformance/src/lib.rs:1755`) | scheduler logs must carry strict/hardened mode + replay |
| `ft-autograd::BackwardOptions::for_mode` | `S1` | high | mode split correctness | `crates/ft-autograd/src/lib.rs:461` | hardened allowlist drift checks (`crates/ft-conformance/src/lib.rs:2311`) | logs must expose reason codes for reentrant policy branches |
| `ft-runtime::RuntimeContext::set_mode` | `S1` | medium | policy-event ledger continuity | `crates/ft-runtime/src/lib.rs:170` | indirectly validated via session and suite mode runs | evidence ledger entries should remain deterministic-format |
| `ft-serialize::decode_checkpoint` | `S1` | critical | strict/hardened fail-closed decode behavior | `crates/ft-serialize/src/lib.rs:425` | serialization differential suite (`crates/ft-conformance/src/lib.rs:1845`) | serialization logs must include fixture + packet refs |
| `ft-serialize::generate_raptorq_sidecar` | `S1` | critical | deterministic repair manifest + decode proof | `crates/ft-serialize/src/lib.rs:425` | parity artifact validator and sidecar checks (`crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:75`) | proof hash/recovered bytes must be replay-verifiable |
| `ft-conformance::run_differential_conformance` | `S2` | critical | report schema + drift accounting correctness | `crates/ft-conformance/src/lib.rs:2525` | itself is differential core | emitted report must be reproducible via `run_differential_report` |
| `ft-conformance::emit_e2e_forensics_matrix_filtered` | `S2` | high | packet-filtered forensic completeness | `crates/ft-conformance/src/lib.rs:2525` | cross-suite differential link via shared scenario IDs | output JSONL must preserve `StructuredCaseLog` schema |
| `ft-conformance::logging::StructuredCaseLog::new` | `S2` | high | deterministic scenario/seed/env fingerprinting | `crates/ft-conformance/src/lib.rs:2525` | all suites consume this log contract | replay command and artifact refs are mandatory fields |
| `validate_phase2c_artifacts` binary gate | `S2` | critical | packet artifact readiness policy | `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:636` | validates differential/sidecar outputs | CI replay must include validator output JSON |

## 13. Data Model and Mutability Boundary Map (`bd-3v0.23.4`)

| Model | State vector | Mutable operations | Mutation boundary | Invalid-state handling | Recovery behavior | Anchor |
|---|---|---|---|---|---|---|
| `TensorMeta` | `shape`, `strides`, `storage_offset`, `dtype`, `device` | constructor + `with_storage_offset` | immutable-after-construction style; new value returned | `validate` rejects rank/stride mismatch, overflow, index rank/oob | fail closed with typed errors (`TensorMetaError`) | `crates/ft-core/src/lib.rs:27` |
| `ScalarTensor` | `id`, `storage_id`, `meta`, `value`, `version` | `with_value`, `alias_view`, `set_in_place` | explicit out-of-place vs in-place split | compatibility + metadata checks guard invalid transitions | in-place bump version; out-of-place allocates new ids/storage | `crates/ft-core/src/lib.rs:287` |
| `DispatchKeySet` | bitset state + selected/backend key derivation | `add`, `remove`, set algebra | key validation before dispatch resolution | unknown bits/incompatible keysets fail closed | strict rejects fallback; hardened bounded fallback to backend key | `crates/ft-dispatch/src/lib.rs:54` |
| `Tape` + autograd nodes | node list, op graph edges, pending dependencies, gradient vector | `leaf`, `add`, `mul`, `backward_with_options` | graph mutation only through tape APIs | unknown node/dependency underflow/reentrant overflow errors | strict fail on depth violation; hardened bounded fallback marks telemetry | `crates/ft-autograd/src/lib.rs:183` |
| `RuntimeContext` + `EvidenceLedger` | execution mode + ordered evidence entries | `set_mode`, `record` | mode changes and event appends only | no silent mode mutation path | ledger records policy/dispatch/backward events deterministically | `crates/ft-runtime/src/lib.rs:21` |
| `CheckpointEnvelope` + `RaptorQSidecar` | schema version, mode, entries, hash, sidecar/proof metadata | encode/decode + sidecar generation | decode path split by strict/hardened policy | unknown fields/version/checksum mismatch fail closed | hardened diagnostics bounded; decode proof required for durability evidence | `crates/ft-serialize/src/lib.rs:40` |
| `StructuredCaseLog` | scenario/replay/env/provenance fields | `StructuredCaseLog::new` | generated per case by harness | schema drift treated as contract break | replay command + artifact refs provide forensic recovery path | `crates/ft-conformance/src/logging.rs:11` |

## 14. Critical State Machines, Violation Semantics, and Recovery

### 14.1 Tensor metadata validation state machine

| State | Trigger | Next state | Violation semantics | Recovery |
|---|---|---|---|---|
| `MetaDraft` | build from shape/strides | `MetaValidated` if `validate` passes | n/a | n/a |
| `MetaDraft` | `validate` detects rank/stride mismatch | `MetaRejected` | fail closed with `RankStrideMismatch` | caller must rebuild metadata |
| `MetaDraft` | overflow in span/offset | `MetaRejected` | fail closed with overflow error variant | caller must narrow shape/stride/offset |
| `MetaValidated` | `storage_index_for` with bad rank/oob | `IndexRejected` | fail closed with `IndexRankMismatch`/`IndexOutOfBounds` | caller fixes index and retries |

Anchors:
- `crates/ft-core/src/lib.rs:83`
- `crates/ft-core/src/lib.rs:156`

### 14.2 Dispatch resolution state machine

| State | Trigger | Next state | Violation semantics | Recovery |
|---|---|---|---|---|
| `KeysetBuilt` | `validate_for_scalar_binary` passes | `KeysetValidated` | n/a | n/a |
| `KeysetBuilt` | empty/unknown/incompatible bits | `DispatchRejected` | fail closed with `DispatchKeyError` | caller repairs keyset construction |
| `KeysetValidated` | selected key is composite/backendselect in strict mode | `DispatchRejected` | strict policy violation | rerun in hardened only if explicitly policy-allowed |
| `KeysetValidated` | selected key is composite/backendselect in hardened mode | `FallbackRouted` | non-fatal, `fallback_used=true` | continue with backend key |
| `FallbackRouted` or `KeysetValidated` | kernel resolved and executed | `DispatchSucceeded` | unsupported pair fails closed | adjust operator/keyset coverage |

Anchors:
- `crates/ft-dispatch/src/lib.rs:145`
- `crates/ft-dispatch/src/lib.rs:264`

### 14.3 Autograd scheduler and reentrant guard state machine

| State | Trigger | Next state | Violation semantics | Recovery |
|---|---|---|---|---|
| `BackwardInit` | root node exists | `QueuePrimed` | unknown root -> fail closed | caller passes valid root id |
| `QueuePrimed` | pop ready node | `NodeProcessing` | dependency underflow -> fail closed | graph/dependency accounting bug; stop run |
| `NodeProcessing` | propagate gradients and dependencies | `QueuePrimed` or `BackwardDone` | n/a | n/a |
| `BackwardInit` | reentrant depth exceeds limit in strict | `BackwardRejected` | `ReentrantDepthExceeded` error | caller lowers depth or policy |
| `BackwardInit` | depth exceeds limit in hardened | `FallbackGuarded` then normal flow | guard flag set in telemetry | continue with bounded depth and audit flag |

Anchors:
- `crates/ft-autograd/src/lib.rs:273`
- `crates/ft-autograd/src/lib.rs:441`

### 14.4 Serialization decode and durability proof state machine

| State | Trigger | Next state | Violation semantics | Recovery |
|---|---|---|---|---|
| `PayloadRead` | strict decode path parse success | `SchemaCheck` | malformed json -> fail closed | producer fixes payload |
| `PayloadRead` | hardened decode parse success | `SchemaCheck` | malformed json -> bounded diagnostic error | producer fixes payload; diagnostics retained |
| `SchemaCheck` | version/hash/field checks pass | `DecodeSucceeded` | mismatch/unknown field -> fail closed | payload regeneration required |
| `DecodeSucceeded` | sidecar generation + decode proof success | `DurabilityVerified` | proof mismatch/decode failure -> fail closed | regenerate sidecar/proof and revalidate |

Anchors:
- `crates/ft-serialize/src/lib.rs:128`
- `crates/ft-serialize/src/lib.rs:148`
- `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:317`

## 15. Invariant Families with Test/E2E Observability Requirements

| Invariant ID | Invariant statement | Property test candidate | Differential/adversarial candidate | E2E observability point | Required log/evidence fields |
|---|---|---|---|---|---|
| `FT-I1` | tensor metadata validity is preserved across construction/indexing | fuzz shape/stride/offset/index tuples around overflow and rank mismatch | tensor-meta fixture invalid-case + metamorphic offset-shift checks | e2e matrix entries for `tensor_meta` packet `FT-P2C-001` | `scenario_id`, `mode`, `reason_code`, `artifact_refs`, `replay_command` |
| `FT-I2` | dispatch selection is deterministic for equal keysets | property checks over keyset permutations preserving set semantics | dispatch fixture comparisons with strict/hardened branches | e2e matrix entries for `dispatch_key` packet `FT-P2C-002` | `suite_id`, `scenario_id`, `mode`, `outcome`, `reason_code` |
| `FT-I3` | backward execution order respects dependency completion | DAG generator with deterministic expected topological order | scheduler oracle compare including reentrant fallback branch | e2e matrix entries for `autograd_scheduler` packet `FT-P2C-004` | `seed`, `scenario_id`, `mode`, `artifact_refs`, `replay_command` |
| `FT-I4` | in-place mutation increments version while alias semantics remain coherent | property checks for alias/non-alias mutation paths | tensor-meta + scalar differential checks on version/fingerprint drift | targeted packet e2e run with alias-oriented cases | `fixture_id`, `scenario_id`, `reason_code`, runtime ledger evidence |
| `FT-I5` | checkpoint decode is fail-closed on unknown/incompatible payloads | parser mutation corpus for unknown fields/version/hash mismatch | serialization suite strict+hardened mismatch taxonomies | e2e matrix entries for `serialization` packet `FT-P2C-006` | `mode`, `outcome`, `reason_code`, `artifact_refs` |
| `FT-I6` | sidecar/decode proof remains deterministic for same payload | deterministic repeat-run proof hash assertion | corruption drill and recovery event checks | packet validator + sidecar emitter command replay | `artifact_refs`, decode proof hash, recovered bytes |
| `FT-I7` | mode policy transitions are auditable and ordered | mode-switch sequence property tests | compare strict vs hardened ledger footprints | replay via session + e2e forensics logs | runtime evidence ledger entries + `mode` in structured logs |

Required structured logging schema fields for invariant proof:
- `schema_version`
- `suite_id`
- `scenario_id`
- `fixture_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `outcome`
- `reason_code`

Anchor:
- `crates/ft-conformance/src/logging.rs:11`

## 16. Execution-Path Tracing and Control-Flow Narratives (`bd-3v0.23.5`)

### 16.1 Workflow A: API op/backward execution (nominal + branch paths)

Entry:
- `FrankenTorchSession::add/mul` and `FrankenTorchSession::backward_with_options` (`crates/ft-api/src/lib.rs:36`, `crates/ft-api/src/lib.rs:57`)

| Branch | Trigger | Branch behavior | Source anchors | Verification beads | Logging checkpoint and replay |
|---|---|---|---|---|---|
| `A1 nominal_strict_success` | strict mode + resolvable keyset | dispatch resolves key, kernel executes, tape records node, backward completes | `crates/ft-dispatch/src/lib.rs:264`, `crates/ft-autograd/src/lib.rs:273` | `bd-3v0.12.5`, `bd-3v0.12.6`, `bd-3v0.12.7` | log `suite_id=scalar_dac` and replay with `cargo run -p ft-conformance --bin run_e2e_matrix -- --mode strict --packet FT-P2C-001 --output artifacts/phase2c/e2e_forensics/ft-p2c-001.strict.jsonl` |
| `A2 strict_dispatch_reject` | strict mode + composite/backend fallback key path | fail closed with dispatch key incompatibility error | `crates/ft-dispatch/src/lib.rs:277` | `bd-3v0.13.5`, `bd-3v0.13.6`, `bd-3v0.13.7` | log `reason_code` must capture strict fallback denial; replay via dispatch fixture run |
| `A3 hardened_dispatch_fallback` | hardened mode + composite/backendselect selected key | fallback to backend key with `fallback_used=true` | `crates/ft-dispatch/src/lib.rs:287` | `bd-3v0.13.6`, `bd-3v0.13.7` | verify `fallback_used` branch in differential/e2e logs with hardened mode replay |
| `A4 strict_reentrant_reject` | reentrant depth exceeds max under strict policy | fail closed with `ReentrantDepthExceeded` | `crates/ft-autograd/src/lib.rs:284` | `bd-3v0.12.6` | scheduler log must capture strict rejection reason; replay with scheduler fixture in strict mode |
| `A5 hardened_reentrant_guarded` | reentrant depth exceeds max under hardened policy | continue bounded execution with guard telemetry flags | `crates/ft-autograd/src/lib.rs:294` | `bd-3v0.12.6`, `bd-3v0.12.7` | assert `reentrant_guard_triggered=true` in case logs and packet e2e slice |

### 16.2 Workflow B: differential conformance run (nominal + failure paths)

Entry:
- `run_differential_conformance` and report emit (`crates/ft-conformance/src/lib.rs:610`, `crates/ft-conformance/src/lib.rs:1350`)

| Branch | Trigger | Branch behavior | Source anchors | Verification beads | Logging checkpoint and replay |
|---|---|---|---|---|---|
| `B1 oracle_available` | legacy oracle script returns valid response | local-vs-oracle comparisons emitted as pass/fail drift checks | `crates/ft-conformance/src/lib.rs:2118`, `crates/ft-conformance/src/lib.rs:2224` | `bd-3v0.12.6`, `bd-3v0.13.6`, `bd-3v0.14.6`, `bd-3v0.15.6`, `bd-3v0.17.6` | replay with `cargo run -p ft-conformance --bin run_differential_report -- --mode both --output artifacts/phase2c/conformance/differential_report_v1.json` |
| `B2 oracle_unavailable` | python not available/script failure | mark checks with `oracle_unavailable`, keep deterministic evidence refs | `crates/ft-conformance/src/lib.rs:681` | `bd-3v0.23.10` (crosswalk), packet differential beads above | logs must include unavailable reason and fixed replay command to re-run once oracle is restored |
| `B3 allowlisted_drift` | drift id present in allowlist | drift recorded as non-blocking allowlisted event | `crates/ft-conformance/src/lib.rs:2311` | `bd-3v0.9` + packet-specific differential beads | report must expose `allowlisted_drifts` count and drift ids |
| `B4 blocking_drift` | drift not allowlisted and comparator fails | report status escalates to blocking drift | `crates/ft-conformance/src/lib.rs:1350` | packet-specific differential beads | CI replay uses report output path and packet artifact refs |

### 16.3 Workflow C: e2e forensics matrix generation (nominal + branch paths)

Entry:
- `emit_e2e_forensics_matrix_filtered` (`crates/ft-conformance/src/lib.rs:538`)

| Branch | Trigger | Branch behavior | Source anchors | Verification beads | Logging checkpoint and replay |
|---|---|---|---|---|---|
| `C1 both_modes_full_matrix` | modes omitted/`both` | runs all suites for strict+hardened and writes full JSONL | `crates/ft-conformance/src/lib.rs:544` | `bd-3v0.12.7`, `bd-3v0.13.7`, `bd-3v0.14.7`, `bd-3v0.15.7`, `bd-3v0.17.7` | replay with `cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --output artifacts/phase2c/e2e_forensics/e2e_matrix.jsonl` |
| `C2 packet_filtered_slice` | `--packet FT-P2C-00X` passed | only packet-matching log entries retained | `crates/ft-conformance/src/lib.rs:572` | packet-specific e2e beads above | replay with packet filter command and verify `packet_filter` in CLI summary |
| `C3 failed_entries_detected` | any case log outcome != pass | summary `failed_entries > 0`, run not silently green | `crates/ft-conformance/src/lib.rs:600` | packet e2e beads above + follow-on remediation bead | capture failing scenario ids and replay each from structured logs |

### 16.4 Workflow D: packet durability/readiness gate (nominal + branch paths)

Entry:
- sidecar emitter + artifact validator binaries (`crates/ft-conformance/src/bin/emit_packet_sidecar.rs:11`, `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:75`)

| Branch | Trigger | Branch behavior | Source anchors | Verification beads | Logging checkpoint and replay |
|---|---|---|---|---|---|
| `D1 sidecar_and_proof_generated` | parity payload exists and RaptorQ decode proof succeeds | writes `.raptorq.json` and `.decode_proof.json` artifacts | `crates/ft-conformance/src/bin/emit_packet_sidecar.rs:23` | `bd-3v0.9`, packet evidence beads | replay with `cargo run -p ft-conformance --bin emit_packet_sidecar -- FT-P2C-00X` |
| `D2 missing_required_packet_file` | any required packet artifact absent | validator emits `NOT_READY` with file-specific error | `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:325` | `bd-3v0.9` + packet closure beads | replay validator and inspect JSON checks map |
| `D3 schema_or_key_mismatch` | required JSON keys absent/malformed | validator emits hard error on key checks | `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:505` | packet-specific evidence beads | replay validator after regenerating report/sidecar |
| `D4 global_security_allowlist_missing` | global threat matrix or allowlist absent | validator global status `NOT_READY` | `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:135` | `bd-3v0.23.9`, `bd-3v0.9` | replay validator from repo root and inspect `global.errors` |

### 16.5 Trace ambiguity and missing-branch ledger

| Missing or ambiguous branch | Impact | Proposed owner | Follow-up bead |
|---|---|---|---|
| explicit timeout/retry branch for legacy oracle subprocess calls is not yet formalized | intermittent oracle instability may look like deterministic drift | conformance owner | `bd-3v0.23.8` |
| dispatch path has no non-CPU backend branch today | backend expansion may introduce unmodeled fallback interactions | dispatch/device owners | `bd-3v0.23.9`, `bd-3v0.7` |
| e2e matrix pipeline does not yet classify flaky vs deterministic failures | can inflate blocking alerts without flake taxonomy | conformance + scenario-corpus owners | `bd-3v0.20`, `bd-3v0.23.10` |
| packet validator parity-gate YAML checking is section-presence based, not semantic schema validation | branch-level quality gates may be under-specified | conformance tooling owner | `bd-3v0.23.8` |

## 17. Concurrency/Lifecycle Semantics and Ordering Guarantees (`bd-3v0.23.7`)

### 17.1 Concurrency and lifecycle contract map

| Domain | Concurrency model | Ordering guarantee | Hazard class | Current mitigation | Source anchors |
|---|---|---|---|---|---|
| Tensor/storage ID allocation (`ft-core`) | global atomics (`AtomicU64`, relaxed fetch-add) | monotonic unique IDs, but cross-thread interleaving order is not semantically stable | ID ordering assumptions in tests/tools | treat IDs as opaque identity, never semantic order | `crates/ft-core/src/lib.rs:5`, `crates/ft-core/src/lib.rs:299` |
| Autograd scheduler (`ft-autograd`) | single-threaded ready queue lifecycle | deterministic execution order for identical graph + options via queue and dependency gating | dependency underflow, reentrant overflow, accidental nondeterminism | explicit telemetry (`execution_order`, queue stats) + deterministic tests | `crates/ft-autograd/src/lib.rs:98`, `crates/ft-autograd/src/lib.rs:314`, `crates/ft-autograd/src/lib.rs:375` |
| Runtime evidence ledger (`ft-runtime`) | single-writer mutable context in current API shape | append order equals call order within context | timestamp nondeterminism across runs | compare semantic fields, not wall-clock timestamp | `crates/ft-runtime/src/lib.rs:21`, `crates/ft-runtime/src/lib.rs:31`, `crates/ft-runtime/src/lib.rs:164` |
| Differential report assembly (`ft-conformance`) | sequential suite/oracle iteration | canonical sorted check order before emit | comparator order drift if sort removed | explicit sort by packet/suite/mode/case/comparator/drift_id | `crates/ft-conformance/src/lib.rs:610`, `crates/ft-conformance/src/lib.rs:1302` |
| E2E forensics matrix emit | sequential suite aggregation then single file write | deterministic case ordering per selected mode/suite traversal | concurrent writers to same output path | run-specific output paths; serialized writes in single process | `crates/ft-conformance/src/lib.rs:550`, `crates/ft-conformance/src/lib.rs:593` |
| Packet artifact validator | bounded parallel workers with join and final deterministic sort | stable packet ordering in summary after parallel validation | worker panic, shared-output race assumptions | `thread::scope` join + final packet sort + opt-out env flag | `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:277`, `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:295`, `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:106` |
| Legacy oracle subprocess lifecycle | child process spawn, stdin payload, wait-for-output | one request/response transaction per invocation | no timeout -> potential hang/stall surface | explicit error mapping; no timeout yet | `crates/ft-conformance/src/lib.rs:2224`, `crates/ft-conformance/src/lib.rs:2256` |

### 17.2 Ordering-critical lifecycle transitions

| Lifecycle | Required order | Violation symptom | Debug signal |
|---|---|---|---|
| Autograd backward | `compute_reachable` -> dependency snapshot -> queue pop/push -> finalize telemetry | gradients right but execution order drift, or dependency underflow | `AutogradError::DependencyUnderflow`, telemetry `execution_order`, `queue_pushes/pops` |
| Reentrant handling | depth check before queue scheduling | strict unexpectedly proceeds or hardened unexpectedly fails | `ReentrantDepthExceeded` error or missing `reentrant_guard_triggered` flag |
| Differential report emit | gather checks -> classify -> canonical sort -> write json | nondeterministic diff ordering across runs | sorted tuple keys in report generator path |
| Validator summary | parallel packet checks -> join -> packet sort -> global validation | unstable packet ordering, flaky summary diffs | `packets.sort_by(packet_id)` after worker joins |

Anchors:
- `crates/ft-autograd/src/lib.rs:302`
- `crates/ft-autograd/src/lib.rs:336`
- `crates/ft-autograd/src/lib.rs:555`
- `crates/ft-autograd/src/lib.rs:582`
- `crates/ft-conformance/src/lib.rs:1302`
- `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:106`

### 17.3 Race-sensitive surfaces and targeted verification mapping

| Race-sensitive surface | Existing targeted checks | Required stress/e2e scenario hook | Owner bead |
|---|---|---|---|
| scheduler dependency readiness ordering | `dependency_scheduler_waits_for_all_children`, composite determinism tests | e2e stress corpus with multi-branch graphs and replayed seeds | `bd-3v0.12.5`, `bd-3v0.12.6`, `bd-3v0.12.7` |
| strict vs hardened reentrant branch split | strict overflow fail test + hardened fallback telemetry test | adversarial reentrant-depth matrix in differential suite | `bd-3v0.12.6` |
| validator parallelism reproducibility | validator worker join + packet sort logic | CI stress run toggling `FT_DISABLE_PACKET_PARALLELISM` for equivalence | `bd-3v0.9`, `bd-3v0.23.8` |
| concurrent artifact output path collisions | current single-process write calls | e2e scenario corpus should enforce unique output paths per run id | `bd-3v0.20`, `bd-3v0.23.10` |
| oracle subprocess lifecycle stalls | spawn/wait error handling path | timeout/kill stress scenario and hang forensics logging | `bd-3v0.23.8` |

Anchors:
- `crates/ft-autograd/src/lib.rs:497`
- `crates/ft-autograd/src/lib.rs:532`
- `crates/ft-autograd/src/lib.rs:555`
- `crates/ft-autograd/src/lib.rs:582`
- `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:277`
- `crates/ft-conformance/src/lib.rs:2234`

### 17.4 Debugging and logging requirements for ordering violations

Minimum forensic fields (must be present in ordering/race incident evidence):
- `suite_id`
- `scenario_id`
- `mode`
- `seed`
- `packet_id`
- `reason_code`
- `artifact_refs`
- `replay_command`

Additional ordering-specific payload requirements:
- autograd: `execution_order`, `queue_pushes`, `queue_pops`, `max_queue_len`, `dependency_snapshot`
- differential: sorted comparator key tuple (`packet_id/suite/mode/case/comparator/drift_id`)
- validator: worker-mode context (`FT_DISABLE_PACKET_PARALLELISM` on/off)

Anchors:
- `crates/ft-conformance/src/logging.rs:11`
- `crates/ft-autograd/src/lib.rs:69`
- `crates/ft-autograd/src/lib.rs:375`
- `crates/ft-conformance/src/lib.rs:1302`

### 17.5 Known undefined ordering zones and mitigation strategy

| Undefined/weakly-defined zone | Why undefined | Mitigation strategy | Follow-up bead |
|---|---|---|---|
| cross-thread relative ordering of atomically assigned tensor/storage IDs | relaxed atomics guarantee uniqueness, not semantic chronology | never assert on relative ID ordering across threads; assert only uniqueness/invariants | `bd-3v0.12.5` |
| wall-clock timestamps in evidence/logs (`ts_unix_ms`) | clock source varies across runs/environments | treat timestamps as diagnostic only; key comparisons use deterministic fields (`scenario_id`, `seed`) | `bd-3v0.23.10` |
| subprocess completion timing for legacy oracle | external python runtime scheduling is uncontrolled | add timeout + cancellation envelope and emit explicit timeout reason codes | `bd-3v0.23.8` |
| simultaneous writes to shared output file path by independent runs | file writes are not coordinated across separate processes | enforce run-scoped output paths in CI/e2e orchestration | `bd-3v0.20` |

## 18. Error Taxonomy, Failure Modes, and Recovery Semantics (`bd-3v0.23.8`)

### 18.1 Canonical error-class taxonomy across active crates

| Subsystem | Error type | Trigger class | Strict mode semantics | Hardened mode semantics | Owner |
|---|---|---|---|---|---|
| Tensor metadata/indexing | `TensorMetaError` (`RankStrideMismatch`, `StrideOverflow`, `StorageOffsetOverflow`, `IndexRankMismatch`, `IndexOutOfBounds`) | invalid shape/stride layout, offset overflow, rank/oob indexing | fail closed with typed metadata/index errors | same as strict (no alternate repair path in `ft-core`) | tensor core owner |
| Tensor compatibility gate | `TensorCompatError` (`DTypeMismatch`, `DeviceMismatch`) | scalar tensor dtype/device incompatibility | fail closed before dispatch/autograd | same as strict | tensor/device boundary owners |
| Dispatch key validation | `DispatchKeyError` (`EmptySet`, `NoTypeKey`, `NoBackendKey`, `UnknownBits`, `IncompatibleSet`) | malformed keyset, unknown bits, illegal backend/type combinations | fail closed on invalid keysets and composite fallback attempts | bounded fallback only for composite routing branch; otherwise fail closed | dispatch owner |
| Dispatch execution | `DispatchError::{Key, Kernel}` | key-validation failure or kernel execution failure | fail with wrapped typed error | same as strict except composite key fallback path in key-resolution branch | dispatch + kernel owners |
| Autograd scheduler | `AutogradError` (`UnknownNode`, `ReentrantDepthExceeded`, `DependencyUnderflow`, `Dispatch`) | missing node IDs, reentrant overflow, dependency accounting inconsistency, forwarded dispatch failure | fail closed for unknown node/dependency/reentrant overflow | reentrant overflow may proceed through bounded fallback when policy allows; other failures remain fail closed | autograd owner |
| Device guard | `DeviceError::Mismatch` | device mismatch across tensors/guard target | fail closed | same as strict | device owner |
| Serialization/decode | `SerializeError` (`InvalidJson`, `UnknownField`, `VersionMismatch`, `ChecksumMismatch`, `IncompatiblePayload`, `RaptorQFailure`) | malformed payload, schema drift, hash mismatch, incompatible envelope, sidecar/decode proof failure | fail closed for malformed/incompatible payloads | bounded diagnostics for malformed JSON payload prefix, but schema/hash/version compatibility still fail closed | serialization/durability owners |

Anchors:
- `crates/ft-core/src/lib.rs:83`
- `crates/ft-core/src/lib.rs:156`
- `crates/ft-core/src/lib.rs:207`
- `crates/ft-core/src/lib.rs:266`
- `crates/ft-dispatch/src/lib.rs:73`
- `crates/ft-dispatch/src/lib.rs:145`
- `crates/ft-dispatch/src/lib.rs:161`
- `crates/ft-dispatch/src/lib.rs:208`
- `crates/ft-dispatch/src/lib.rs:264`
- `crates/ft-autograd/src/lib.rs:157`
- `crates/ft-autograd/src/lib.rs:273`
- `crates/ft-device/src/lib.rs:8`
- `crates/ft-device/src/lib.rs:40`
- `crates/ft-serialize/src/lib.rs:82`
- `crates/ft-serialize/src/lib.rs:128`
- `crates/ft-serialize/src/lib.rs:148`
- `crates/ft-serialize/src/lib.rs:279`
- `crates/ft-serialize/src/lib.rs:293`
- `crates/ft-serialize/src/lib.rs:328`

### 18.2 Failure-mode matrix (trigger -> impact -> detection -> recovery -> ownership)

| Failure mode | Trigger | Impact surface | Detection path | Recovery semantics | Ownership |
|---|---|---|---|---|---|
| metadata topology invalid | rank/stride mismatch or overflow while validating `TensorMeta` | tensor creation/indexing rejects request; no downstream dispatch/autograd | unit/property guards + tensor-meta conformance mismatch record | caller repairs shape/stride/offset inputs and retries | tensor core |
| dispatch keyset corruption | unknown bits, missing backend/type, or incompatible set | operation cannot route to kernel; strict branch blocks composite fallback | dispatch conformance with `expected_error_observed` and parity checks | strict: explicit fix required; hardened: bounded composite fallback where configured | dispatch |
| scheduler graph integrity breach | dependency underflow or unknown node in backward traversal | backward execution aborts; gradient contract cannot be trusted | scheduler conformance + autograd telemetry (`execution_order`, queue stats) | fail closed; repair graph construction/edge accounting before retry | autograd |
| reentrant depth overflow | backward called above allowed depth | strict run halts; hardened may continue with guard flag | strict/hardened reentrant conformance cases + policy telemetry | strict: lower depth or adjust policy; hardened: bounded fallback with explicit reason code | autograd |
| device mismatch | tensor devices differ during guard/compat checks | op fails before arithmetic semantics are evaluated | unit guard checks (partial coverage) + future conformance negative fixture | fail closed; align devices before retry | device |
| checkpoint schema/integrity drift | unknown fields/version mismatch/checksum mismatch | decode rejected; snapshot replay blocked | serialization conformance (`serialization_expectation_mismatch`) + decode tests | fail closed; regenerate payload with compatible schema/hash | serialization |
| malformed checkpoint payload | JSON parse failure | decode rejected with diagnostic payload prefix in hardened path | hardened malformed decode test + serialization conformance logging | strict: parse failure; hardened: bounded diagnostic + reject | serialization |
| RaptorQ sidecar/decode failure | sidecar generation or decode-proof validation fails | durability evidence becomes non-trustworthy | sidecar generation tests + serialization conformance sidecar repeat checks | fail closed on proof generation; adjust symbol/repair parameters and regenerate | serialization + durability |
| legacy oracle subprocess instability | subprocess fails/returns malformed response | differential signal downgraded to unavailable/noisy evidence | conformance reason codes (`legacy_oracle_unavailable`) + e2e forensics summary | classify and reroute; timeout/kill envelope still pending as follow-on | conformance |

Anchors:
- `crates/ft-conformance/src/lib.rs:1477`
- `crates/ft-conformance/src/lib.rs:1635`
- `crates/ft-conformance/src/lib.rs:1685`
- `crates/ft-conformance/src/lib.rs:1755`
- `crates/ft-conformance/src/lib.rs:1845`
- `crates/ft-conformance/src/lib.rs:1916`
- `crates/ft-conformance/src/lib.rs:2194`
- `crates/ft-conformance/src/lib.rs:2705`

### 18.3 User-facing error semantics and fail-closed doctrine

- Default contract is typed fail-closed behavior for unknown/incompatible state across tensor metadata, dispatch keysets, device guards, and serialization schema/hash checks.
- Hardened mode may add bounded diagnostics or bounded fallback only where explicitly encoded:
  - dispatch composite fallback branch in `dispatch_scalar_binary_with_keyset`
  - autograd reentrant depth overflow fallback under hardened policy
  - serialization malformed JSON diagnostics (`payload_prefix`) while still rejecting decode
- No hardened branch is allowed to silently coerce unknown bits/schema drift into success; unknown incompatible features remain hard failures.

Anchors:
- `crates/ft-dispatch/src/lib.rs:264`
- `crates/ft-dispatch/src/lib.rs:387`
- `crates/ft-dispatch/src/lib.rs:408`
- `crates/ft-autograd/src/lib.rs:556`
- `crates/ft-autograd/src/lib.rs:582`
- `crates/ft-serialize/src/lib.rs:293`
- `crates/ft-serialize/src/lib.rs:457`
- `crates/ft-serialize/src/lib.rs:472`
- `crates/ft-serialize/src/lib.rs:480`
- `crates/ft-serialize/src/lib.rs:497`

### 18.4 Mapping to negative tests, differential checks, and failure-injection e2e hooks

| Error family | Unit/property negative-test anchor | Differential/adversarial anchor | E2E/failure-injection hook | Gap/owner |
|---|---|---|---|---|
| `TensorMetaError` | tensor metadata/index bound tests in `ft-core` | tensor-meta conformance suite and metamorphic checks | e2e matrix entries from tensor-meta case logs | mature path; extend overflow edge corpus (`bd-3v0.12.6`) |
| `DispatchKeyError` / `DispatchError::Key` | dispatch key unknown/composite strict-vs-hardened tests | dispatch conformance case loader + expected-error branch | e2e matrix includes dispatch packet cases and reason codes | mature path; expand malformed keyset corpus (`bd-3v0.13.6`) |
| `AutogradError` | reentrant and unknown-node unit tests | scheduler conformance with strict/hardened telemetry assertions | e2e matrix scheduler packet entries | dependency-underflow dedicated adversarial fixture pending (`bd-3v0.12.6`) |
| `SerializeError` | strict unknown field, version/hash mismatch, hardened malformed payload tests | serialization conformance + sidecar determinism checks | e2e matrix serialization packet entries | mature path; add incompatible-payload adversarial fixture (`bd-3v0.17.6`) |
| `DeviceError` / `TensorCompatError` | only positive-path unit checks present today | no dedicated conformance negative fixture yet | no dedicated failure-injection scenario yet | explicit gap; add fixtures and e2e scenarios (`bd-3v0.20`, `bd-3v0.23.10`) |
| oracle transport errors | legacy oracle unavailable/malformed response branches | differential runs produce `legacy_oracle_unavailable` reason class | e2e summary `failed_entries` + replay command extraction | timeout/cancel branch still pending (`bd-3v0.23.8` follow-on) |

Anchors:
- `crates/ft-core/src/lib.rs:441`
- `crates/ft-dispatch/src/lib.rs:379`
- `crates/ft-dispatch/src/lib.rs:387`
- `crates/ft-dispatch/src/lib.rs:408`
- `crates/ft-autograd/src/lib.rs:556`
- `crates/ft-autograd/src/lib.rs:582`
- `crates/ft-autograd/src/lib.rs:606`
- `crates/ft-device/src/lib.rs:74`
- `crates/ft-serialize/src/lib.rs:457`
- `crates/ft-serialize/src/lib.rs:472`
- `crates/ft-serialize/src/lib.rs:480`
- `crates/ft-serialize/src/lib.rs:497`
- `crates/ft-conformance/src/lib.rs:538`
- `crates/ft-conformance/src/lib.rs:600`
- `crates/ft-conformance/src/lib.rs:2644`
- `crates/ft-conformance/src/lib.rs:2681`

### 18.5 Forensic logging requirements by failure class

All failure classes must emit deterministic replay fields already carried by conformance logs:
- `suite`
- `scenario_id`
- `mode`
- `packet_id`
- `reason_code`
- `artifact_refs`
- `replay_command`
- environment stamp fields (`rustc_version`, `os`, `arch`) in e2e matrix entries

Failure-specific additions required for this pass:
- dispatch failures: include keyset bits and whether hardened fallback branch executed
- scheduler failures: include `reentrant_guard_triggered`, dependency and queue counters
- serialization failures: include decode mode + schema/hash classification (`unknown_field`, `version_mismatch`, `checksum_mismatch`, `invalid_json`, `raptorq_failure`)
- oracle transport failures: include subprocess exit status / stderr digest when available

Anchors:
- `crates/ft-conformance/src/logging.rs:11`
- `crates/ft-conformance/src/logging.rs:24`
- `crates/ft-conformance/src/lib.rs:322`
- `crates/ft-conformance/src/lib.rs:329`
- `crates/ft-conformance/src/lib.rs:1943`
- `crates/ft-conformance/src/lib.rs:1981`

## 19. Full-Agent Deep Dive Pass A: Structure Specialist Findings (`bd-3v0.23.15`)

### 19.1 Module-boundary and ownership coherence verdict

| Layer | Primary ownership | Boundary verdict | Key anchors |
|---|---|---|---|
| tensor primitives + metadata invariants | `ft-core` | coherent: no external crate mutates core tensor invariants directly | `crates/ft-core/src/lib.rs:10` |
| device guard boundary | `ft-device` | coherent but thin: explicit guard exists, negative-path conformance is sparse | `crates/ft-device/src/lib.rs:7` |
| kernel leaf operations | `ft-kernel-cpu` | coherent: kernel crate remains leaf and typed-error only | `crates/ft-kernel-cpu/src/lib.rs:7` |
| dispatch selection + policy split | `ft-dispatch` | coherent: key-model and strict/hardened split centralized | `crates/ft-dispatch/src/lib.rs:8` |
| graph/tape scheduling | `ft-autograd` | coherent: scheduler invariants and policy gates local to autograd crate | `crates/ft-autograd/src/lib.rs:1` |
| runtime evidence + durability envelope contracts | `ft-runtime` | partially integrated: contracts exist but limited cross-crate consumption | `crates/ft-runtime/src/lib.rs:5` |
| checkpoint/decode + sidecar generation | `ft-serialize` | coherent: serialization and sidecar proofs are isolated and deterministic | `crates/ft-serialize/src/lib.rs:14` |
| user session facade | `ft-api` | coherent: API delegates to lower layers and records evidence | `crates/ft-api/src/lib.rs:7` |
| conformance/differential/e2e orchestration | `ft-conformance` | coherent: central harness composes all packet suites and logs replay fields | `crates/ft-conformance/src/lib.rs:335` |

### 19.2 Structural sections to coverage-plan traceability check

| Structural domain | Unit/property traceability | Differential/metamorphic/adversarial traceability | E2E/logging traceability | Verdict |
|---|---|---|---|---|
| scalar autograd pipeline (`ft-api` -> `ft-autograd` -> `ft-dispatch`) | scalar and autograd unit tests + conformance suite | differential scalar checks vs legacy oracle | e2e matrix records scalar scenarios with replay commands | strong |
| tensor metadata/state invariants | core tensor-meta unit checks and conformance cases | metamorphic offset-shift + oracle guard paths | e2e matrix includes tensor-meta packet logs | strong |
| dispatch key routing and fallback policy | dispatch unit tests for unknown bits/strict-hardened split | dispatch differential checks and expected-error paths | e2e logs include dispatch case reason codes | strong |
| scheduler ordering/reentrant semantics | autograd scheduler unit tests (reentrant + unknown-node) | scheduler differential comparisons and telemetry matching | scheduler packet scenarios in e2e matrix | strong |
| serialization schema/hash/sidecar durability checks | strict/hardened decode + checksum/version tests | serialization differential checks and sidecar determinism assertions | serialization packet scenarios in e2e matrix | strong |
| runtime evidence ledger + durability envelope | unit-level runtime tests only | no dedicated differential assertion on ledger durability events | no explicit e2e matrix assertion for durability ledger entries | weak (remediation required) |

Anchors:
- `crates/ft-conformance/src/lib.rs:392`
- `crates/ft-conformance/src/lib.rs:422`
- `crates/ft-conformance/src/lib.rs:449`
- `crates/ft-conformance/src/lib.rs:476`
- `crates/ft-conformance/src/lib.rs:503`
- `crates/ft-conformance/src/lib.rs:530`
- `crates/ft-conformance/src/lib.rs:610`
- `crates/ft-conformance/src/logging.rs:11`

### 19.3 Contradictions and omissions with actionable remediation notes

| Finding | Why it is a structural contradiction/omission | Actionable remediation | Owner beads |
|---|---|---|---|
| durability envelope contract exists in `ft-runtime` but is not asserted by conformance/e2e harness | architecture claims durable evidence path, but only runtime unit tests exercise it | add conformance fixture/case that invokes sidecar generation and asserts runtime ledger durability entry + scrub status propagation | `bd-3v0.9`, `bd-3v0.10`, `bd-3v0.17.7` |
| API ledger entries are validated only in `ft-api` unit tests, not in integration conformance/e2e flows | deterministic audit-log contract is stated as cross-cutting, but integration-level log guarantees are currently indirect | extend scalar/scheduler conformance cases to assert session ledger snapshots and include ledger references in `StructuredCaseLog.artifact_refs` | `bd-3v0.20`, `bd-3v0.23.10`, `bd-3v0.12.7` |
| device mismatch and tensor compatibility negatives are structurally part of boundary doctrine but weakly represented in conformance/e2e | boundary appears explicit in code, but harness does not yet carry dedicated adversarial fixtures for these failures | add dedicated negative fixtures and e2e failure-injection scenarios for `DeviceError::Mismatch` and `TensorCompatError` branches | `bd-3v0.20`, `bd-3v0.23.10`, `bd-3v0.12.6` |

Anchors:
- `crates/ft-runtime/src/lib.rs:21`
- `crates/ft-runtime/src/lib.rs:84`
- `crates/ft-api/src/lib.rs:58`
- `crates/ft-api/src/lib.rs:125`
- `crates/ft-device/src/lib.rs:40`
- `crates/ft-core/src/lib.rs:372`
- `crates/ft-conformance/src/lib.rs:1845`
- `crates/ft-conformance/src/lib.rs:1943`

### 19.4 Final integrated rewrite prerequisites (inputs to `bd-3v0.23.14`)

- Durability bridge prerequisite:
  bind `ft-serialize` sidecar/decode-proof outputs to runtime durability ledger evidence in conformance/e2e artifacts before final doc sign-off.
- Ledger integration prerequisite:
  require at least one conformance/e2e assertion that verifies audit-ledger payload presence and deterministic replay fields per packet scenario.
- Boundary negative-case prerequisite:
  include explicit device/compat failure fixtures in scenario corpus so structure claims and failure doctrine remain aligned.
- Traceability prerequisite:
  each structural claim in sections 2/3/5/6 must continue to map to unit/property + differential/adversarial + e2e/logging anchors with no orphaned domain.

## 20. Unit/E2E Test Corpus and Logging Evidence Crosswalk (`bd-3v0.23.10`)

Machine-diffable source-of-truth artifact:
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

Canonical diff-gate command (stable ordering for CI drift checks):
- `jq -S . artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

### 20.1 Behavior-to-evidence crosswalk summary

| Behavior ID | Major behavior | Unit/property mapping | Differential/adversarial mapping | E2E + scenario-id mapping | Logging/replay obligations |
|---|---|---|---|---|---|
| `BHV-FT-001` | scalar DAC arithmetic + gradients | `ft-api`/`ft-autograd` scalar tests | scalar comparator set in differential report | scalar strict+hardened scenario IDs in full e2e matrix | require `scenario_id`, `seed`, `mode`, `env_fingerprint`, `replay_command`, `reason_code` |
| `BHV-FT-002` | tensor metadata indexing + fail-closed invalid shapes | `ft-core` index/stride guards | tensor_meta local+oracle differential checks + metamorphic branch | tensor_meta scenario IDs (valid + invalid branches) | same required log fields + fixture lineage |
| `BHV-FT-003` | dispatch key routing and strict/hardened fallback split | `ft-dispatch` unknown bits + mode-split tests | dispatch selected-key/fallback differential checks | dispatch scenario IDs including `composite_route_mode_split` | same required log fields + explicit fallback reason code |
| `BHV-FT-004` | scheduler ordering and reentrant policy | autograd deterministic/dependency/reentrant tests | scheduler output/order differential checks | scheduler strict+hardened scenario IDs | same required log fields + scheduler policy context |
| `BHV-FT-005` | serialization decode integrity + sidecar determinism | `ft-serialize` unknown field/version/checksum/malformed tests | serialization decode + sidecar differential checks | serialization strict+hardened checkpoint scenario IDs | same required log fields + decode/sidecar artifact refs |
| `BHV-FT-006` | failure forensics envelope + artifact index UX | triage/index binary unit tests | forensics deterministic ID and reason taxonomy checks | full e2e matrix + crash triage + index generator workflow | failure envelope must carry replay + env fingerprint and evidence-link categories |

### 20.2 Scenario ID, log schema, and replay contract

Scenario ID contract:
- generated by `scenario_id` helper in `crates/ft-conformance/src/lib.rs:2345`
- format: `{suite}/{mode}:{canonical_case_name}`

Structured logging contract (required fields):
- `suite_id`
- `scenario_id`
- `fixture_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `outcome`
- `reason_code`

Anchors:
- `crates/ft-conformance/src/logging.rs:11`
- `crates/ft-conformance/src/logging.rs:24`
- `artifacts/phase2c/TEST_LOG_CONTRACT_V1.md`
- `artifacts/phase2c/FAILURE_FORENSICS_ENVELOPE_SCHEMA_V1.md`

### 20.3 Explicit coverage gaps with priority and dependency linkage

| Gap ID | Gap summary | Priority | Dependency linkage | Closure path |
|---|---|---|---|---|
| `GAP-UX-001` | missing device mismatch / tensor compatibility negative scenarios in conformance + e2e corpus | high | `bd-3v0.20`, `bd-3v0.23.10`, `bd-3v0.12.6` | add fixture rows + scenario IDs + packet-scoped replay hooks |
| `GAP-UX-002` | oracle timeout/cancel branch not represented in crosswalk replay taxonomy | high | `bd-3v0.21`, `bd-3v0.23.10` | add timeout reason family + one-command replay/triage path |
| `GAP-UX-003` | runtime durability-ledger integration not yet cross-linked in conformance/e2e evidence | medium | `bd-3v0.9`, `bd-3v0.10`, `bd-3v0.17.7` | bind sidecar/decode proof events to runtime durability ledger evidence refs |

Gap ledger source:
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`
- `artifacts/phase2c/USER_WORKFLOW_SCENARIO_GAP_LEDGER_V1.md`

## 21. Security/Compatibility Edge Cases and Undefined Zones (`bd-3v0.23.9`)

### 21.1 Security and compatibility edge-case matrix with mode split

| Edge case | Threat/compat risk | Strict mode behavior | Hardened mode behavior | Mitigation status | Anchors |
|---|---|---|---|---|---|
| dispatch keyset contains unknown bits | execution routing corruption via unsupported feature flags | fail closed with `DispatchKeyError::UnknownBits` | same fail-closed behavior | covered in unit + dispatch conformance | `crates/ft-dispatch/src/lib.rs:79`, `crates/ft-dispatch/src/lib.rs:381` |
| composite key requires backend fallback | silent semantic drift if fallback allowed in strict mode | fail closed (`IncompatibleSet`) | bounded fallback permitted and explicitly marked `fallback_used=true` | covered, allowlisted drift only in hardened mode | `crates/ft-dispatch/src/lib.rs:280`, `crates/ft-dispatch/src/lib.rs:286`, `crates/ft-conformance/src/lib.rs:2010` |
| malformed or schema-drifted checkpoint payload | parser confusion, replay mismatch, potential corrupted restore | fail closed on unknown field/version/hash mismatch | malformed JSON gives bounded diagnostics but still rejected; unknown fields/version/hash still fail closed | covered by serialization conformance and decode tests | `crates/ft-serialize/src/lib.rs:282`, `crates/ft-serialize/src/lib.rs:330`, `crates/ft-serialize/src/lib.rs:343`, `crates/ft-serialize/src/lib.rs:473` |
| reentrant autograd depth overflow | scheduler instability or runaway recursion | fail closed (`ReentrantDepthExceeded`) | bounded fallback via `HardenedBoundedFallback` with telemetry flag | covered with explicit strict/hardened tests | `crates/ft-autograd/src/lib.rs:286`, `crates/ft-autograd/src/lib.rs:292`, `crates/ft-autograd/src/lib.rs:602` |
| allowlisted hardened deviations | accidental masking of strict regressions | strict mode never allowlists drift | allowlist active only for hardened mode + explicit drift ID | covered with allowlist parse/contains tests | `crates/ft-conformance/src/lib.rs:2000`, `crates/ft-conformance/src/lib.rs:2011`, `crates/ft-conformance/src/lib.rs:2837` |
| missing security matrix / allowlist files in artifact validator | release-ready report could pass without security controls | validator marks packet/global status `NOT_READY` | same behavior (fail closed) | covered by validator key/file checks | `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:145`, `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:185`, `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs:262` |

### 21.2 Undefined/deferred zones and fail-closed decision register

| Zone | Current decision | Why unresolved | Required follow-up |
|---|---|---|---|
| device mismatch / tensor compat adversarial paths in e2e corpus | deferred with explicit gap (`GAP-UX-001`), currently fail closed in code | conformance fixtures do not yet include dedicated mismatch scenarios | add fixtures + scenario IDs under `bd-3v0.12.6` and update crosswalk (`bd-3v0.23.10`) |
| oracle timeout/cancel branch | deferred with explicit gap (`GAP-UX-002`) | oracle unavailable is logged, but timeout kill-path taxonomy is not yet modeled | extend forensics UX and taxonomy under `bd-3v0.21` follow-on |
| runtime durability-ledger linkage into conformance/e2e | deferred with explicit gap (`GAP-UX-003`) | sidecar artifacts exist; runtime durability evidence not fully cross-linked in e2e index | close with `bd-3v0.9`, `bd-3v0.10`, `bd-3v0.17.7` |
| unknown reason-code emergence | fail closed by reliability budget gate (`max_unknown_reason_codes`) | new reason families can appear as features expand | maintain policy in `RELIABILITY_BUDGET_POLICY_V1.json` and extend checker taxonomy |

Anchors:
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`
- `artifacts/phase2c/USER_WORKFLOW_SCENARIO_GAP_LEDGER_V1.md`
- `artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json`
- `crates/ft-conformance/src/bin/check_reliability_budgets.rs:339`

### 21.3 Threat narratives mapped to adversarial fixtures and e2e abuse scenarios

| Threat narrative | Adversarial fixture or corpus mapping | E2E abuse hook | Forensics artifact |
|---|---|---|---|
| malformed tensor metadata intended to bypass bounds/index checks | `tensor_meta_cases.json` invalid rank/overflow cases + adversarial corpus manifest | `run_e2e_matrix` packet filter `FT-P2C-001` | `artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl` + crash triage |
| dispatch route manipulation via composite/unknown keys | `dispatch_key_cases.json` `composite_route_mode_split` + unknown-bits tests | `run_e2e_matrix` packet filter `FT-P2C-002` | `failure_forensics_index_v1.json` evidence templates for `dispatch_key` |
| scheduler abuse through reentrant depth pressure | `autograd_scheduler_cases.json` + strict/hardened reentrant unit tests | `run_e2e_matrix` packet filter `FT-P2C-004` | e2e matrix + reliability gate scenario counts |
| serialization payload poisoning (unknown fields/version/checksum) | `serialization_cases.json` + strict/hardened decode negative tests + adversarial/fuzz manifest | `run_e2e_matrix` packet filter `FT-P2C-006` | crash triage + failure index + reliability gate report |

### 21.4 Policy override and recovery logging/audit expectations

Mandatory override/recovery audit fields:
- `scenario_id`
- `mode`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`

Additional override-specific requirements:
- when drift is accepted, log must indicate hardened-mode allowlist path (`allowlisted=true`, drift ID present)
- when bounded fallback occurs, log path must carry explicit mode-split reason (e.g., strict fail-closed vs hardened fallback)
- every recovery/override event must be traceable from e2e entry to crash triage and failure-forensics index artifacts

Anchors:
- `crates/ft-conformance/src/lib.rs:1937`
- `crates/ft-conformance/src/lib.rs:1948`
- `crates/ft-conformance/src/lib.rs:1988`
- `crates/ft-conformance/src/lib.rs:2011`
- `crates/ft-conformance/src/logging.rs:20`
- `crates/ft-conformance/src/logging.rs:21`
- `crates/ft-conformance/src/logging.rs:22`
- `crates/ft-conformance/src/logging.rs:24`
- `artifacts/phase2c/e2e_forensics/failure_forensics_index_v1.json`

## 22. Cross-Cutting Validation Gate Note

This pass (`bd-3v0.23.2` + `bd-3v0.23.3` + `bd-3v0.23.4` + `bd-3v0.23.5` + `bd-3v0.23.7` + `bd-3v0.23.8` + `bd-3v0.23.9` + `bd-3v0.23.10` + `bd-3v0.23.15`) is docs/planning only.

Execution evidence is explicitly deferred to implementation/conformance beads:
- unit/property evidence: `bd-3v0.12.5`, `bd-3v0.13.5`, `bd-3v0.14.5`, `bd-3v0.15.5`, `bd-3v0.17.5`
- differential/metamorphic/adversarial evidence: `bd-3v0.12.6`, `bd-3v0.13.6`, `bd-3v0.14.6`, `bd-3v0.15.6`, `bd-3v0.17.6`
- e2e/logging evidence: `bd-3v0.12.7`, `bd-3v0.13.7`, `bd-3v0.14.7`, `bd-3v0.15.7`, `bd-3v0.17.7`
