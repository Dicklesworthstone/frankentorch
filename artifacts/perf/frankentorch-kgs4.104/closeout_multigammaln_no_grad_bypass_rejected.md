# frankentorch-kgs4.104 - multigammaln no-grad bypass rejected

## Target

- Bead: `frankentorch-kgs4.104`
- Profile-backed hotspot: `multigammaln_p4_1m`
- Profile source: `artifacts/perf/frankentorch-special-reprofile-20260615/baseline_special_bench.log`
- Existing profile median on `ovh-a`: `9.6436 ms`
- Local baseline after the 2026-06-16 `ts1` override: `[7.1825 ms 7.4027 ms 7.7425 ms]`

## Candidate Lever

One lever was tested: for inputs that do not require grad, bypass `tensor_apply_function` and compute the same p-term `lgamma_approx` sum directly into a leaf tensor. The tracked/autograd path stayed on the existing `tensor_apply_function` branch.

## Behavior Proof

- Ordering: unchanged candidate map order, one output per input index.
- Floating point: same `constant` initialization and same inner `i=0..p` addition order.
- RNG/ties: no RNG, no ordering ties.
- Autograd: tracked `requires_grad=true` path preserved the existing backward closure.
- Candidate proof test: `multigammaln_log_ndtr_parallel_match_serial_bit_exact`.
- Candidate forward digest: `0x7cbfa4c6ede4804a`.
- Candidate backward digest: `0xba249e682eb30f57`.
- Golden-output SHA-256 over deterministic digest lines: `07daf0ff28696135cc72fd742e595c15a2f04e06743fbadcdad22e44ffbe80b3`.

Proof artifacts:

- `artifacts/perf/frankentorch-kgs4.104/proof_multigammaln_fast_path_pinned.log`
- `artifacts/perf/frankentorch-kgs4.104/proof_multigammaln_fast_path_sha256.txt`

## Benchmark Result

Local Criterion, same isolated target dir:

- Baseline: `[7.1825 ms 7.4027 ms 7.7425 ms]`
- Candidate: `[6.8313 ms 7.0073 ms 7.3314 ms]`
- Median ratio: `1.056x`
- Criterion: `change [-10.233% -3.9994% +2.5257%]`, `p = 0.27`, no significant change detected.

Score: `0.85 = Impact 1.056 * Confidence 0.40 / Effort 0.50`.

Decision: REJECT. The source/test hunk was removed; only evidence is kept.

## Next Route

The saved-tensor bypass is too small for this benchmark. The next pass should target a deeper special-function primitive, starting with the heavier profile rows such as `polygamma2_1m` or iterative gamma-family scalar kernels rather than another autograd wrapper micro-lever.
