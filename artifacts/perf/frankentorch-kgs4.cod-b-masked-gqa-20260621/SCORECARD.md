# frankentorch-kgs4 cod-b masked SDPA GQA verification

Date: 2026-06-21
Agent: IvoryDeer
Target dir: `/data/projects/.rch-targets/frankentorch-cod-b`

## Result

Verdict: mixed. The masked f64 SDPA primary and tensor entry points still beat PyTorch, but the masked GQA path loses badly.

| Lane | FrankenTorch | PyTorch | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| primary masked f64 SDPA | 8.013 ms | 20.846 ms | 2.60x faster | rel-diff 3.29e-14 |
| tensor masked f64 SDPA | 7.916 ms | 21.558 ms | 2.72x faster | rel-diff 3.29e-14 |
| masked f64 GQA | 38.224 ms | 4.545 ms | 8.41x slower | rel-diff 3.19e-14 |

Win/loss/neutral for this proof: 2W / 1L / 0N.

## Evidence

- `local_binary_sdpa_masked_headtohead_three_row.log`: same-host FrankenTorch binary plus local PyTorch comparator.
- `local_pytorch_masked_three_row.log`: local PyTorch-only timing sanity check.
- `rch_sdpa_masked_headtohead_three_row_after_fix.log`: RCH FrankenTorch-only timing on `vmi1153651`; remote PyTorch executable was unavailable there.
- `test_ft_conformance_release.log`: first conformance attempt; failed because the shared checkout had a partial API/kernel masked-SDPA signature mismatch.
- `test_ft_conformance_release_after_maskdiv_fix.log`: `rch exec -- cargo test -p ft-conformance --profile release`; green.

## Conformance

`ft-conformance` release gate passed through RCH on `vmi1227854`:

- lib tests: 199 passed.
- conformance bins/integration/smoke suites: passed.
- doctests: passed.

The shared checkout required a local API compatibility adaptation for a partial broadcast-mask branch before this gate could build; no product source is kept in this evidence commit.

## Next Lever

The radical lever is not another wrapper around the current GQA entry point. The GQA loss is consistent with materializing repeated K/V heads before calling the masked flash kernel. The promising design is a direct grouped masked f64 flash kernel that indexes `kv_head = q_head / group` and never expands K/V. Retry only if that direct grouped kernel is implemented and measured head-to-head against PyTorch GQA.
