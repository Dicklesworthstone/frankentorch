# ft-kernel-cpu Topk Partial Select

- Bead: `frankentorch-kgs4.25`
- Umbrella: `frankentorch-kgs4`
- Skills: `/repeatedly-apply-skill`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-kernel-cpu`
- Target benchmark: `topk_bench::topk_f64_8192x1024_k50_dim1`

## Profile Target

The profiled topk target spent each lane doing a full stable sort over all
1024 values, then copying only `k = 50` selected pairs.

Baseline via `rch`:

```text
worker: vmi1227854
command: rch exec -- cargo bench -p ft-kernel-cpu --bench topk_bench -- topk_f64_8192x1024_k50_dim1 --warm-up-time 1 --measurement-time 4 --sample-size 20
topk_f64_8192x1024_k50_dim1: [107.70 ms 127.21 ms 150.43 ms]
```

## One Lever

For `topk_tensor_contiguous_f64`, use `select_nth_unstable_by` when
`k < dim_size` to partition the exact top-k set under a deterministic total
order. Sort only the selected `k` values when `sorted = true`; for
`sorted = false`, keep the existing original-index order for the selected set.

The `k == dim_size` path keeps the full lane sort because no partition can save
work there.

## Isomorphism Proof

- Ordering: output tensor traversal and writes are unchanged.
- Tie-breaking: the partition comparator orders by value and then original
  index. This is equivalent to the old stable sort because each lane is built
  in ascending original-index order.
- Floating point: no arithmetic was added or removed; values are only compared
  and copied, preserving all bit patterns, including NaN payloads and signed
  zeros.
- NaN semantics: `nan_greatest_cmp_f64` is still the value comparator; NaN is
  greatest in ascending order and therefore first for `largest = true`.
- RNG: not involved.
- Golden output: `ft_kernel_cpu_topk_partial_select_frankentorch-kgs4-25.txt`
  records NaN, duplicate, signed-zero, sorted, and unsorted output contracts
  with sha256
  `5efc6444ba6369c21782eaa0c66117273a73da283ac0c512524f9b6c141f9cdd`.

## Result

After via `rch`:

```text
worker: vmi1156319
command: rch exec -- cargo bench -p ft-kernel-cpu --bench topk_bench -- topk_f64_8192x1024_k50_dim1 --warm-up-time 1 --measurement-time 4 --sample-size 20
topk_f64_8192x1024_k50_dim1: [19.127 ms 20.471 ms 21.789 ms]
```

Delta:

- p50: `127.21 ms -> 20.471 ms`, about 6.2x faster.
- score: Impact 3 * Confidence 3 / Effort 1 = 9.0.
- decision: keep.

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu topk_parallel_matches_serial_bit_exact -- --nocapture` passed on `vmi1156319`: 1/1, 377 filtered.
- `rch exec -- cargo test -p ft-kernel-cpu topk -- --nocapture` passed on `vmi1149989`: 6/6, 372 filtered.
- `rch exec -- cargo check -p ft-kernel-cpu --all-targets` passed on `vmi1153651`.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings` passed on `vmi1153651`.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check` passed; RCH classified fmt as a local non-compilation command.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- `git diff --check` passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs artifacts/optimization/2026-06-03_ft_kernel_cpu_topk_partial_select_frankentorch-kgs4-25.md artifacts/optimization/golden_outputs/ft_kernel_cpu_topk_partial_select_frankentorch-kgs4-25.txt artifacts/optimization/golden_checksums.txt` exited 0 with no critical findings; it reported broad pre-existing warning inventories in `ft-kernel-cpu`.
