# frankentorch-kgs4.153 — intra-head block parallelism for unmasked f64 flash SDPA

Date: 2026-06-21
Agent: cc
Third application of the nested-BR-block trick (kgs4.151 GQA, kgs4.152 masked dense,
now the unmasked dense inference kernel).

## Lever

`ft_kernel_cpu::sdpa_forward_f64` (the unmasked f64 flash kernel behind the no-grad
`scaled_dot_product_attention` inference fast path, causal and non-causal) parallelised
only over `num_bh = B*H` heads — `B*H=16` on a 64-core host left 48 cores idle. Each head's
`BR`-row blocks are independent, so split them across the pool too, **guarded** by
`num_bh < rayon::current_num_threads()` (head-heavy inputs keep the cheaper serial inner
loop: no regression). The causal case benefits extra: late query blocks attend to more keys
(longer rows), so the serial loop was load-imbalanced — the split balances it across cores.

## Correctness (bit-exact)

Per-`BR`-block work is independent (own scores, own output rows, shared read-only K/V), so
the split changes no reduction order. ft-kernel-cpu lib 530 passed / 0 failed; ft-conformance
green (199 lib + bins/integration/smoke/doctests), which includes the SDPA goldens.

## Measurement (same-host, 32 torch threads = release-scorecard convention)

Shape `[B*H=16, S=512, D=64]`, no-grad f64, `example sdpa_inference_headtohead`.

| Lane | FT before | FT after | PyTorch (32t) | win before → after |
| --- | ---: | ---: | ---: | --- |
| RAW `sdpa_forward_f64` kernel | `4.23 ms` | `2.98–3.07 ms` | — | ~1.4x kernel speedup |
| non-causal (through session) | `7.04 ms` | `6.33–6.56 ms` | `~17.1 ms` | `2.43x` → `2.64–2.69x` faster |
| causal (through session) | `8.64 ms` | `7.53–7.84 ms` | `~16–18 ms` | `1.85x` → `2.27–2.35x` faster |

The causal win improves most (1.85x → 2.3x) thanks to the load-balanced split. Through-session
times also carry fixed `tensor_variable` (~4.6ms input build) + `read_out` (~2.4ms) overhead
that the kernel change does not touch; the RAW-kernel row isolates the ~1.4x kernel gain.

## Win/loss/neutral vs PyTorch (32t): `2W / 0N` (widens two existing wins; no regression)

## Gates

- `cargo test -p ft-kernel-cpu --release --lib`: 530 passed, 0 failed, 2 ignored.
- `cargo test -p ft-conformance --release`: 199 lib + bins/integration/smoke/doctests green.
