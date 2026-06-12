# frankentorch-8ubk3 Pass 1 Baseline/Profile

Date: 2026-06-12

Scope: baseline/profile only for `frankentorch-8ubk3`, the exact-shift
blocked Francis sweep successor to rejected `frankentorch-fy8to`.

No production source was edited.

## Criterion Baseline

Primary paired worker: `hz1`.

```text
eigvals_f64_256x256 time: [33.839 ms 34.014 ms 34.212 ms]
eig_f64_256x256     time: [68.327 ms 68.927 ms 69.570 ms]
```

Raw logs:

- `artifacts/perf/frankentorch-8ubk3/pass1_bench_eigvals_256.log`
- `artifacts/perf/frankentorch-8ubk3/pass1_bench_eig_256.log`

The raw logs also contain retrieved secondary Criterion chunks from other
workers. Those are routing context only. Keep/reject proof for the first source
lever must rerun immediate before/after on the same selected worker.

## Strict Golden

Command:

```text
rch exec -- cargo run --release -q -p ft-kernel-cpu --example eigvals_golden
```

Worker: `ovh-a`.

Extracted stdout:

```text
frankentorch-l9xod eigvals_golden n=64
eigvals_digest=0xbc0583d464b1a211
eig_digest=0xbc0583d464b1a211
frankentorch-l9xod eigvals_golden n=128
eigvals_digest=0x763c4b15d92c4b89
eig_digest=0x763c4b15d92c4b89
frankentorch-l9xod eigvals_golden n=256
eigvals_digest=0x00b87b4996340204
eig_digest=0x00b87b4996340204
```

Strict stdout SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Artifacts:

- `artifacts/perf/frankentorch-8ubk3/pass1_eigvals_golden.log`
- `artifacts/perf/frankentorch-8ubk3/pass1_eigvals_golden.stdout`
- `artifacts/perf/frankentorch-8ubk3/pass1_eigvals_golden.stdout.sha256`

## Timing/Profile Probe

Command:

```text
rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_timing_probe
```

Worker: `ovh-a`.

```text
n=128   eigvals=    4.15ms  eig=    5.61ms  (vec_machinery=1.45ms)
profile n=128   sweeps=173 defl1=28 defl2=50 fallback=0 exceptional=0 max_width=128 samples=173 truncated=false first_shift=[0..127 x=6.690e1 y=5.117e1 w=1.724e0 exceptional=false]
n=256   eigvals=   29.42ms  eig=   39.49ms  (vec_machinery=10.07ms)
profile n=256   sweeps=319 defl1=14 defl2=121 fallback=0 exceptional=0 max_width=256 samples=319 truncated=false first_shift=[0..255 x=1.290e2 y=1.240e2 w=6.569e1 exceptional=false]
n=512   eigvals=  311.44ms  eig=  408.97ms  (vec_machinery=97.53ms)
profile n=512   sweeps=583 defl1=10 defl2=251 fallback=0 exceptional=0 max_width=512 samples=583 truncated=false first_shift=[0..511 x=2.554e2 y=2.588e2 w=5.377e1 exceptional=false]
n=1024  eigvals= 1979.73ms  eig= 3285.78ms  (vec_machinery=1306.06ms)
profile n=1024  sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0 max_width=1024 samples=1132 truncated=false first_shift=[0..1023 x=5.118e2 y=5.120e2 w=1.282e2 exceptional=false]
```

Raw log:

- `artifacts/perf/frankentorch-8ubk3/pass1_eig_timing_probe.log`

## Diagnosis

The target is still the serial non-symmetric Francis QR sweep stream:

- no fallback deflations
- no exceptional shifts
- 319 sweeps at n=256
- 1132 sweeps at n=1024
- strict digest currently restored after `fy8to` rejection

The next pass must not alter the shift policy. It must first write the proof
contract for exact preservation of shift samples, selected `m`, deflation
counters, complex-pair slots, and stdout SHA before any source hunk.

## Verdict

Productive baseline/profile pass.
