# frankentorch-kgs4.106 closeout: polygamma2 unit-interval recurrence

Status: rejected, source hunk removed.

Target: `ft-api` `polygamma2_1m`, selected from profile-backed special-function evidence in `artifacts/perf/frankentorch-special-reprofile-20260615/` and `artifacts/perf/frankentorch-polygamma-nosave-20260615/`.

Local baseline after the `ts1` remote override:

- Command: `env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo bench -j 1 -p ft-api --bench special_bench -- polygamma2_1m --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Result: `polygamma2_1m [6.2319 ms 6.4721 ms 6.6888 ms]`
- Artifact: `artifacts/perf/frankentorch-kgs4.106/pass1_local_baseline_polygamma2_1m.log`
- SHA-256: `e1b64ffa7dcd5ee1b956567357f24e2ef2cd7fb79bb59cddc48b7eb4ba8ea422`

One lever tested:

- For `n=2` and `0<x<1`, replace the dynamic recurrence loop prefix with a straight-line 10-step recurrence prefix.
- Preserve exact `x += 1.0` order, exact `-2.0 / (x*x*x)` arithmetic, and the unchanged asymptotic tail.
- Leave `NaN`, non-positive integer poles, `n!=2`, and outside-unit-interval behavior unchanged.

Behavior proof:

- Command: `env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo test -j 1 -p ft-api polygamma --lib -- --nocapture`
- Result: 5 polygamma tests passed.
- Exact-bit proof: candidate output matched an independent copy of the original recurrence for 16,384 unit-interval values plus outside-domain cases.
- Candidate FNV digest: `0x46fce67a7da6a156`.
- Existing public forward/backward and parallel-vs-serial polygamma proofs passed.
- Artifact: `artifacts/perf/frankentorch-kgs4.106/pass3_proof_polygamma.log`
- SHA-256: `200b4e3972c9b6184e5717b4ff1b05c9a69416e61256260b0efa2e9c840f0c21`

Rebench:

- Command: `env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo bench -j 1 -p ft-api --bench special_bench -- polygamma2_1m --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Result: `polygamma2_1m [6.1669 ms 6.3070 ms 6.5575 ms]`
- Criterion result: change `[-5.7252% -1.6221% +2.8874%]`, `p = 0.51`, no significant change.
- Artifact: `artifacts/perf/frankentorch-kgs4.106/pass4_local_rebench_polygamma2_1m.log`
- SHA-256: `292a70a3717d5a40b958d1aed383f7335d048ca403359ddcd0466b0eeb0d3003`

Score:

- Impact: `1.026` median ratio (`6.4721 ms / 6.3070 ms`).
- Confidence: `0.35` because Criterion reported no significant change.
- Effort: `0.50`.
- Score: `0.72`, below the `2.0` keep threshold.

Decision:

- Reject.
- Remove the source/test hunk.
- Route next work to a deeper scalar special-function primitive rather than another wrapper or recurrence-prefix micro-lever.
