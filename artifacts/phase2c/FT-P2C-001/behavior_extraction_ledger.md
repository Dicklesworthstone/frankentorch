# FT-P2C-001 — Behavior Extraction Ledger

Packet: Tensor metadata + storage core  
Legacy anchor map: `artifacts/phase2c/FT-P2C-001/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C001-B01` | nominal | `sizes()/strides()/storage_offset()` core metadata reads | deterministic scalar add/mul output + gradients | same math contract, defensive policy retained | `ft_api::session_add_backward_records_evidence`, `ft_autograd::add_backward_matches_expected_gradient`, `ft_autograd::mul_backward_matches_expected_gradient` | `scalar_dac/strict:add_basic`=`11230214846748554265`, `scalar_dac/hardened:add_basic`=`7472292100909643468` |
| `FTP2C001-B02` | nominal | contiguous and strided indexing metadata | valid linear index and contiguous flags match expectation | same | `ft_core::custom_strides_validate_and_index_into_storage`, `ft_core::shape_builds_contiguous_strides` | `tensor_meta/strict:contiguous_basic_index`=`4395520818158430929`, `tensor_meta/hardened:strided_offset_index`=`9223797700649824762` |
| `FTP2C001-B03` | edge | version/storage alias fields (`version_counter_`, `storage_`) | out-of-place bumps version; alias preserves storage identity | same | `ft_core::out_of_place_result_gets_new_storage_and_version_bump`, `ft_core::alias_view_shares_storage_identity` | `tensor_meta/strict:scalar_offset_index`=`2717510909112554909` |
| `FTP2C001-B04` | adversarial | rank/stride guard surface (`sizes_and_strides_`) | fail closed with `RankStrideMismatch` | same fail-closed outcome | `ft_core::index_rank_and_bounds_are_guarded` + tensor-meta invalid fixture assertion | `tensor_meta/strict:invalid_rank_stride_mismatch`=`9353830229903822145`, `tensor_meta/hardened:invalid_rank_stride_mismatch`=`5997540812546318856` |
| `FTP2C001-B05` | adversarial | overflow-sensitive offset/index path | fail closed with overflow error family | same fail-closed outcome | `ft_core::custom_strides_validate_and_index_into_storage` overflow branch + conformance invalid overflow case | `tensor_meta/strict:invalid_storage_offset_overflow`=`11931105988727078667`, `tensor_meta/hardened:invalid_storage_offset_overflow`=`1156477838142738040` |
| `FTP2C001-B06` | deferred/compat edge | device/dtype compatibility boundary (`device()`, `data_type_`) | fail closed on mismatch | same (no repair) | add explicit negative fixture coverage for `TensorCompatError`/`DeviceError` | candidate seeds: `tensor_meta/strict:compat_device_mismatch_candidate`=`1609459201001`, `tensor_meta/hardened:compat_device_mismatch_candidate`=`1609459201002` (tracked in `GAP-UX-001`) |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all behavior families):
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `outcome`
- `reason_code`

Behavior-family additions:
- nominal scalar/tensor paths: include differential comparator family (`oracle.scalar_*` / `oracle.tensor_meta`) in linked artifacts.
- adversarial fail-closed paths: require explicit fail reason category (`*_expectation_mismatch` or `expected_error_observed`).
- deferred compatibility paths: must emit gap linkage bead IDs in artifact refs until scenario IDs are added.

Anchors:
- `crates/ft-conformance/src/logging.rs:11`
- `crates/ft-conformance/src/lib.rs:1943`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A.
Execution evidence is carried by downstream packet beads:
- unit/property: `bd-3v0.12.5`
- differential/adversarial: `bd-3v0.12.6`
- e2e/logging: `bd-3v0.12.7`
