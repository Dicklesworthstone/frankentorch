# PHASE2C_EXTRACTION_PACKET.md — FrankenTorch

Date: 2026-02-13

Purpose: convert Phase-2 analysis into direct implementation tickets with concrete legacy anchors, target crates, and oracle tests.

## 1. Ticket Packets

| Ticket ID | Subsystem | Legacy anchors (classes/functions) | Target crates | Oracle tests |
|---|---|---|---|---|
| `FT-P2C-001` | Tensor metadata + storage core | `TensorImpl`, `AutogradMetaInterface`, `VariableVersion` in `c10/core/TensorImpl.h` | `ft-core` | tensor construction/view suites in `test/test_torch.py`, tensor metadata tests |
| `FT-P2C-002` | Dispatch key model | `DispatchKey` enum in `DispatchKey.h`; `DispatchKeySet` class in `DispatchKeySet.h`; dispatch macros in `aten/src/ATen/Dispatch.h` | `ft-dispatch` | `test/test_dispatch.py`, `test/test_python_dispatch.py` |
| `FT-P2C-003` | Op schema ingestion | `DispatchKey`/`NativeFunction` classes in `torchgen/model.py`; `- func:` schema entries in `aten/src/ATen/native/native_functions.yaml` | `ft-dispatch`, `ft-kernel-cpu` | schema and dispatch integration tests |
| `FT-P2C-004` | Autograd engine scheduling | `NodeTask`, `ReadyQueue`, `Engine::execute`, `thread_main`, `GraphTask` in `torch/csrc/autograd/engine.h` + `engine.cpp` | `ft-autograd` | `test/test_autograd.py`, `test/autograd/*` |
| `FT-P2C-005` | CPU kernel first-wave semantics | selected kernel contracts from `aten/src/ATen/native/cpu/*` aligned to staged schema closure plan | `ft-kernel-cpu` | op parity suites in `test/test_torch.py` and op-specific tests |
| `FT-P2C-006` | Serialization/checkpoint contract | `THPStorage_writeFileRaw`, `THPStorage_readFileRaw` in `torch/csrc/serialization.cpp` | `ft-serialize`, `ft-core` | `test/test_serialization.py` |
| `FT-P2C-007` | Device guard and backend transitions | `DeviceGuard` and backend transition semantics from `c10/core/*` + backend tests | `ft-device`, `ft-dispatch` | `test/test_cuda.py`, backend-specific suites |
| `FT-P2C-008` | NN module/state contract first-wave | module/state behavior from `torch/nn/*` and state-dict interactions | `ft-nn`, `ft-serialize` | `test/test_nn.py`, `test/nn/*` |

## 2. Packet Definition Template

For each ticket above, deliver all artifacts in the same PR:

1. `legacy_anchor_map.md`: path + line anchors + extracted behavior.
2. `contract_table.md`: input/output/error + dtype/device/grad semantics.
3. `fixture_manifest.json`: oracle mapping and fixture IDs.
4. `parity_gate.yaml`: strict + hardened pass criteria.
5. `risk_note.md`: boundary risks and mitigations.

## 3. Strict/Hardened Expectations per Packet

- Strict mode: exact PyTorch observable behavior for implemented surface, with explicit closure to full drop-in parity.
- Hardened mode: same outward contract with bounded defensive checks (invalid graph/device/state).
- Unknown incompatible schema/serialization/version path: fail-closed.

Foundational policy artifacts:
- `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX_V1.md`
- `artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json`
- `artifacts/phase2c/ESSENCE_EXTRACTION_LEDGER_V1.md`

## 4. Immediate Execution Order

1. `FT-P2C-001`
2. `FT-P2C-002`
3. `FT-P2C-003`
4. `FT-P2C-004`
5. `FT-P2C-005`
6. `FT-P2C-006`
7. `FT-P2C-007`
8. `FT-P2C-008`

## 5. Done Criteria (Phase-2C)

- All 8 packets have extracted anchor maps and contract tables.
- At least one runnable fixture family exists per packet in `ft-conformance`.
- Packet-level parity and gradient report schema is produced for every packet.
- RaptorQ sidecars are generated for fixture bundles and parity reports.

## 6. Per-Ticket Extraction Schema (Mandatory Fields)

Every `FT-P2C-*` packet MUST include:
1. `packet_id`
2. `legacy_paths`
3. `legacy_symbols`
4. `tensor_storage_contract`
5. `dispatch_contract`
6. `error_contract`
7. `grad_graph_contract`
8. `serialization_contract`
9. `strict_mode_policy`
10. `hardened_mode_policy`
11. `sequencing_boundary`
12. `oracle_tests`
13. `performance_sentinels`
14. `compatibility_risks`
15. `raptorq_artifacts`

Missing fields => packet is `NOT READY`.

## 7. Risk Tiering and Gate Escalation

| Ticket | Risk tier | Why | Extra gate |
|---|---|---|---|
| `FT-P2C-001` | Critical | tensor metadata/storage is foundational | metadata invariant ledger |
| `FT-P2C-002` | Critical | dispatch key routing controls execution | dispatch route witness suite |
| `FT-P2C-003` | Critical | op schema ingestion affects kernel binding | schema diff lock |
| `FT-P2C-004` | Critical | autograd scheduling drift is severe | gradient graph replay |
| `FT-P2C-006` | High | serialization compatibility externally visible | checkpoint round-trip gate |
| `FT-P2C-007` | High | device transitions easy to regress | cross-device parity gate |

Critical tickets require strict drift `0` and explicit gradient-drift summary.

## 8. Packet Artifact Topology (Normative)

Directory template:
- `artifacts/phase2c/FT-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FT-P2C-00X/contract_table.md`
- `artifacts/phase2c/FT-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FT-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FT-P2C-00X/risk_note.md`
- `artifacts/phase2c/FT-P2C-00X/parity_report.json`
- `artifacts/phase2c/FT-P2C-00X/parity_report.raptorq.json`
- `artifacts/phase2c/FT-P2C-00X/parity_report.decode_proof.json`
- cross-packet extraction ledger: `artifacts/phase2c/ESSENCE_EXTRACTION_LEDGER_V1.md`

Schema lock reference:
- `artifacts/phase2c/SCHEMA_LOCK_V1.md`

## 9. Optimization and Isomorphism Proof Hooks

Optimization allowed only after strict parity baseline.

Required proof block:
- dispatch ordering preserved
- tensor metadata invariants preserved
- gradient behavior preserved for implemented ops with explicit closure criteria for remaining parity surface
- fixture checksum verification pass/fail

## 10. Packet Readiness Rubric

Packet is `READY_FOR_IMPL` only when:
1. extraction schema complete,
2. fixture manifest includes happy/edge/adversarial paths,
3. strict/hardened gates are machine-checkable,
4. risk note includes compatibility + security mitigations,
5. parity report has RaptorQ sidecar + decode proof.

Machine-check command:

```bash
cargo run -p ft-conformance --bin validate_phase2c_artifacts
```

Non-zero exit means at least one packet is `NOT_READY`.

## 11. Current Packet Status (2026-02-13)

- `FT-P2C-001`: first implementation slice landed (`parity_green`).
- `FT-P2C-002`: dispatch key model landed with strict/hardened route gating (`parity_green`).
- `FT-P2C-004`: autograd scheduling packet landed with dependency scheduler + reentrant policy split (`parity_green`).
- `FT-P2C-006`: serialization packet landed with typed checkpoint envelope + deterministic hash + RaptorQ sidecar/decode proof (`parity_green`).
- Artifact folders:
  - `artifacts/phase2c/FT-P2C-001/`
  - `artifacts/phase2c/FT-P2C-002/`
  - `artifacts/phase2c/FT-P2C-004/`
  - `artifacts/phase2c/FT-P2C-006/`

## 12. Extreme Optimization Evidence Snapshot (2026-02-13)

- Target: `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs`
- Lever applied: single-pass packet file cache (read required packet artifacts once; validate from cached payloads).
- Corpus: synthetic 250-packet tree at `/tmp/ft_phase2c_bench_before_ep1sp0`.
- Before: `68.8 ms` mean (`hyperfine`, 15 runs, warmup 3, ignore failure).
- After: `64.7 ms` mean (same command).
- Delta: `~6.0%` faster wall-time.
- Syscall delta (`strace -c`, same corpus):
  - Before total calls: `12,354`
  - After total calls: `10,098`
  - Improvement: `~18.3%` fewer calls.

Behavior-isomorphism gates run after change:
- `cargo fmt --check`
- `cargo check --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo test -p ft-conformance -- --nocapture`
- `cargo bench`
