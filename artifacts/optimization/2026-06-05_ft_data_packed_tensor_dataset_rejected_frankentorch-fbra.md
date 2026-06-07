# ft-data packed TensorDataset collation rejected

Bead: `frankentorch-fbra`
Date: 2026-06-05
Agent: BlackThrush
Crate: `ft-data`
Target: `dataloader/epoch_2048x256_batch128`
Verdict: rejected and source hunk restored

## Profile-backed target

The ready perf queue was ownership/policy blocked, so this pass used the
profile-backed fallback target from the earlier `ft-data` sampler rejection:
structural `TensorDataset`/`DataLoader` collation.

Fresh baseline via rch Criterion:

```text
worker: ts1
command: RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench dataloader_bench -- dataloader/epoch_2048x256_batch128 --warm-up-time 1 --measurement-time 5 --sample-size 20
dataloader/epoch_2048x256_batch128: [1.8250 ms 1.9114 ms 1.9896 ms]
```

## Candidate

Pack uniform `TensorDataset` tensor columns once at construction, then collate
contiguous ascending batches by copying one large packed slice per tensor.
Malformed or nonuniform datasets fell back to the original per-sample validation
path so error timing and messages stayed deferred to batch collation.

This was a structural memory-layout lever, not another sampler loop tweak.

## Proof while candidate was present

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-data tensor_dataset_packed_collation_golden_summary -- --nocapture
```

Result:

```text
worker: ts1
test tests::tensor_dataset_packed_collation_golden_summary ... ok
```

Golden fixture:

```text
e2134e7d689c121a19b8f692c6d0fb14855be5270285a928678bfbf13a078e2a  artifacts/optimization/golden_outputs/ft_data_packed_tensor_dataset_frankentorch-fbra.txt
```

Isomorphism proof:

- sample index ordering unchanged: `DataLoader` still consumed the same
  `indices[position..batch_end]` slice and advanced `position` before collation;
- tensor ordering and names unchanged: candidate packed tensors in first-sample
  order and emitted names unchanged;
- floating-point bytes unchanged: candidate copied existing `f64` slices without
  arithmetic, rounding, sorting, or reduction;
- RNG unchanged: no sampler or shuffle code was touched;
- tie-breaking unchanged: no comparison or ordering path was introduced.

Additional gates while candidate was present:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-data --all-targets
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
cargo fmt -p ft-data --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check
ubs crates/ft-data/src/lib.rs
```

Results:

```text
cargo check: passed
clippy: passed
fmt: passed
golden checksum verification: passed
diff whitespace: passed
UBS: exit 0, 0 critical, existing warning inventory only
```

## Rebenchmark

Same-worker candidate run:

```text
worker: ts1
command: RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-data --bench dataloader_bench -- dataloader/epoch_2048x256_batch128 --warm-up-time 1 --measurement-time 5 --sample-size 20
dataloader/epoch_2048x256_batch128: [1.9329 ms 2.0904 ms 2.2584 ms]
```

Delta:

```text
mean: 1.9114 ms -> 2.0904 ms
ratio: 0.9144x
score: 0.9144 * 0.95 / 1.0 = 0.87
```

## Decision

Rejected below the required Score >= 2.0 keep gate. The source hunk was
restored, leaving no runtime change.

Next deeper target: stop optimizing per-batch collation copies inside the current
owned-batch API. The next `ft-data` pass should attack a different primitive:
batch-view tensors or arena-backed session ingestion that avoids materializing
new `Vec<f64>` batch payloads while preserving PyTorch-observable batch tensor
values and deterministic DataLoader ordering.
