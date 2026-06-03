# frankentorch-kg7s rejection evidence

## Target

- Crate: `ft-api`
- Benchmark: `cargo bench -p ft-api --bench ops_bench -- sinusoidal_pe/4096x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Lever attempted: compute one denominator/angle per even+odd positional-encoding column pair, then write sin/cos slots directly.

## Profile Evidence

Initial clean-HEAD baseline from the bead start note:

- Worker: `vmi1227854`
- Criterion: `[6.4915 ms 6.9369 ms 7.4445 ms]`

Fresh clean-HEAD baseline:

- Worker: `vmi1227854`
- Criterion: `[8.4981 ms 9.3531 ms 10.157 ms]`

Cross-worker after runs were not used as keep proof, but showed the lever was not robust:

- Worker: `vmi1156319`
- Criterion after: `[17.241 ms 18.095 ms 19.099 ms]`

Decisive same-worker rejection pair:

- Worker: `vmi1149989`
- Clean baseline: `[11.161 ms 11.953 ms 12.579 ms]`
- After: `[30.935 ms 32.092 ms 33.856 ms]`
- p50 result: `11.953 ms -> 32.092 ms`, a 2.68x regression.

## Verdict

Rejected. Source lever was reverted and not kept. The source inspection finding is that the existing implementation already precomputes denominator powers once per column outside the position loop, so the attempted pair-loop only removed a tiny setup cost and made the hot row loop slower. The next positional-encoding pass should replace or batch the trigonometric generation strategy instead of micro-tuning the row loop.
