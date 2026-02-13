# FT-P2C-004 — Legacy Anchor Map

Packet: Autograd engine scheduling  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `torch/csrc/autograd/engine.h` | 51 | `struct NodeTask` | ready-task payload model |
| `torch/csrc/autograd/engine.h` | 210 | `void compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr)` | dependency counting contract |
| `torch/csrc/autograd/engine.h` | 229 | `virtual void thread_main(const std::shared_ptr<GraphTask>& task)` | scheduler event loop shape |
| `torch/csrc/autograd/engine.h` | 230 | `void reentrant_thread_init()` | reentrant initialization path |
| `torch/csrc/autograd/engine.cpp` | 518 | `auto Engine::thread_main(...) -> void` | worker-side pop/execute/push loop semantics |
| `torch/csrc/autograd/engine.cpp` | 1248 | `auto Engine::compute_dependencies(...)` | concrete dependency traversal logic |
| `torch/csrc/autograd/engine.cpp` | 1286 | `auto Engine::execute(...)` | graph-task execution entry and queue bootstrapping |

## Implemented Mapping

| Rust crate | File | Mapping |
|---|---|---|
| `ft-autograd` | `crates/ft-autograd/src/lib.rs` | deterministic ready-queue scheduler, dependency counters, reentrant mode policy, telemetry |
| `ft-api` | `crates/ft-api/src/lib.rs` | mode-aware `BackwardOptions` plumbing and telemetry evidence logging |
| `ft-conformance` | `crates/ft-conformance/src/lib.rs` | scheduler fixture family and strict/hardened reentrant checks |

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-004`
2. `legacy_paths`: `torch/csrc/autograd/engine.h`, `torch/csrc/autograd/engine.cpp`
3. `legacy_symbols`: `NodeTask`, `compute_dependencies`, `thread_main`, `reentrant_thread_init`, `Engine::execute`
4. `tensor_storage_contract`: unchanged scalar tensor storage from `FT-P2C-001`
5. `dispatch_contract`: unchanged except mode-aware metadata used by scheduling evidence
6. `error_contract`: reentrant depth overflow + dependency underflow are explicit
7. `grad_graph_contract`: dependency-driven deterministic replay with execution-order telemetry
8. `serialization_contract`: no direct packet-local change
9. `strict_mode_policy`: reentrant overflow fails closed
10. `hardened_mode_policy`: bounded reentrant fallback with telemetry flag
11. `excluded_scope`: multithreaded worker pools, CUDA streams, graph task futures
12. `oracle_tests`: `test/test_autograd.py`, `test/autograd/*` (scoped scheduling semantics)
13. `performance_sentinels`: queue push/pop counts, max queue depth, scheduler tail latency
14. `compatibility_risks`: ordering differences vs. full PyTorch graph-task machinery
15. `raptorq_artifacts`: sidecar + decode proof emitted for packet parity report
