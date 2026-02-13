# FT-P2C-006 — Legacy Anchor Map

Packet: Serialization/checkpoint contract  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `torch/csrc/serialization.cpp` | 149 | `void doRead(io fildes, void* raw_buf, size_t nbytes)` | exact-size read contract (EOF fail-closed) |
| `torch/csrc/serialization.cpp` | 197 | `void doWrite(io fildes, void* raw_buf, size_t nbytes)` | exact-size write contract |
| `torch/csrc/serialization.cpp` | 235 | `void THPStorage_writeFileRaw(...)` | raw storage write semantics |
| `torch/csrc/serialization.cpp` | 243 | `size_t size_bytes = self->nbytes();` | byte-accurate payload sizing |
| `torch/csrc/serialization.cpp` | 323 | `c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw(...)` | raw storage read semantics |
| `torch/csrc/serialization.cpp` | 347 | `_storage_nbytes == nbytes` assertion block | strict storage-size compatibility gate |
| `torch/csrc/serialization.cpp` | 369 | `doRead(file, data, storage->nbytes());` | full-byte read obligation |

## Implemented Mapping

| Rust crate | File | Mapping |
|---|---|---|
| `ft-serialize` | `crates/ft-serialize/src/lib.rs` | typed checkpoint envelope, strict/hardened decode, checksum gate, version gate, RaptorQ sidecar + decode proof |
| `ft-conformance` | `crates/ft-conformance/src/lib.rs` | serialization fixture family + sidecar/proof determinism checks |
| `ft-conformance` | `crates/ft-conformance/src/bin/emit_packet_sidecar.rs` | packet parity-report sidecar/proof emitter |

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-006`
2. `legacy_paths`: `torch/csrc/serialization.cpp`
3. `legacy_symbols`: `doRead`, `doWrite`, `THPStorage_writeFileRaw`, `THPStorage_readFileRaw`
4. `tensor_storage_contract`: checkpoint entries preserve node/value/grad typed fields
5. `dispatch_contract`: unchanged by this packet
6. `error_contract`: unknown fields, version mismatch, checksum mismatch are fail-closed
7. `grad_graph_contract`: serialized gradients are optional and typed (`Option<f64>`)
8. `serialization_contract`: strict envelope + bounded hardened diagnostics + deterministic hash
9. `strict_mode_policy`: deny unknown fields and any incompatible schema/version/hash
10. `hardened_mode_policy`: bounded diagnostics while still fail-closing incompatible payloads
11. `excluded_scope`: full `.pt` archive format, tensors > scalar scope, storage alias graph
12. `oracle_tests`: `test/test_serialization.py` (scoped behavioral subset)
13. `performance_sentinels`: checkpoint encode/decode latency + sidecar generation overhead
14. `compatibility_risks`: PyTorch binary format breadth exceeds scoped JSON envelope
15. `raptorq_artifacts`: generated via `ft-serialize::generate_raptorq_sidecar`
