#!/usr/bin/env python3
"""Which FrankenTorch conv3d route is more ACCURATE? — w3pol / frankentorch-68pwz.

Reads the raw f64 dumps written by
`crates/ft-api/examples/conv3d_route_torch_arbiter.rs` and scores both routes against
(a) a real torch conv3d in float64 and (b) an EXACT rational reference.

WHY THIS EXISTS. NEGATIVE_EVIDENCE item 68d compared FrankenTorch's direct kernel against
FrankenTorch's streamed kernel, found 4.770e-12 of relative disagreement, and concluded a
"tolerance ratification" was needed before the faster route could be gated in. That framing
is wrong, and it is wrong in a way worth naming: **a disagreement between two of our own
kernels cannot say which one is right.** Neither is a reference. Ratifying a tolerance
would have been ratifying our own drift against itself.

There are two real references, and this script uses both:

  1. TORCH, because this campaign's parity rule is parity with torch.
  2. THE EXACT ANSWER. Every input here is a float64 and therefore an exact rational, so
     the true value of the convolution is exactly computable with `fractions.Fraction` and
     then correctly rounded to float64. Against that, "which route is more accurate" is a
     fact, not a policy. This is the stronger arbiter: it also scores TORCH.

The exact pass is the expensive one, so it runs on a deterministic stratified SAMPLE of
outputs (every Nth, so the sample spans all channels and spatial positions) rather than on
all 122880. The torch pass covers every element.

No numpy: this interpreter's venv has torch without numpy, and installing into a shared
venv to run a probe is not worth the blast radius. stdlib `array` reads the dumps.

Run:
    /data/tmp/torchvenv-2121/bin/python scripts/conv3d_route_torch_arbiter.py
"""

import array
import os
import struct
import sys
import time
from fractions import Fraction

DIR = os.environ.get("CONV3D_ARBITER_DIR", "/data/tmp/conv3d_arbiter_68pwz")
SAMPLE_STRIDE = int(os.environ.get("CONV3D_ARBITER_SAMPLE_STRIDE", "97"))


def load(name):
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        sys.exit(f"missing {path}; run the Rust probe first")
    values = array.array("d")
    with open(path, "rb") as handle:
        values.frombytes(handle.read())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def as_ordinal(value):
    """Sign-magnitude float64 -> a monotone int, so |difference| counts representables."""
    bits = struct.unpack("<q", struct.pack("<d", value))[0]
    return bits if bits >= 0 else -(1 << 63) - bits


def ulp_gap(a, b):
    return abs(as_ordinal(a) - as_ordinal(b))


def relative(a, b):
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def score_against(label, got, ref):
    exact = 0
    max_rel = 0.0
    max_ulp = 0
    total_ulp = 0
    for g, r in zip(got, ref):
        gap = ulp_gap(g, r)
        total_ulp += gap
        if gap == 0:
            exact += 1
        else:
            max_ulp = max(max_ulp, gap)
            max_rel = max(max_rel, relative(g, r))
    n = len(got)
    print(
        f"  {label:<10}  exact {exact:>7}/{n:<7}"
        f"  max_rel {max_rel:.3e}  max_ulp {max_ulp:>6}  mean_ulp {total_ulp / n:>8.3f}"
    )
    return total_ulp / n, max_rel


def main():
    try:
        import torch
    except ImportError:
        sys.exit("no torch here; use /data/tmp/torchvenv-2121/bin/python")

    with open(os.path.join(DIR, "shape.txt")) as handle:
        batch, in_ch, pd, ph, pw, out_ch, k, od, oh, ow = (
            int(v) for v in handle.read().split()
        )

    padded = load("padded.f64")
    weight = load("weight.f64")
    out_stream = load("out_stream.f64")
    out_direct = load("out_direct.f64")

    print("conv3d_route_torch_arbiter (w3pol)")
    print(f"torch {torch.__version__}   threads {torch.get_num_threads()}")
    print(f"input [{batch},{in_ch},{pd},{ph},{pw}]  weight [{out_ch},{in_ch},{k},{k},{k}]")
    print(f"output [{batch},{out_ch},{od},{oh},{ow}] = {len(out_stream)} elements")
    print(f"reduction depth k = {in_ch * k * k * k}")
    print()

    # ---- reference 1: torch, float64, over every element -------------------------------
    # The Rust side already padded the input, so padding=0 here.
    x = torch.frombuffer(bytearray(padded.tobytes()), dtype=torch.float64).reshape(
        batch, in_ch, pd, ph, pw
    )
    w = torch.frombuffer(bytearray(weight.tobytes()), dtype=torch.float64).reshape(
        out_ch, in_ch, k, k, k
    )
    started = time.time()
    ref_t = torch.nn.functional.conv3d(x, w, bias=None, stride=1, padding=0)
    torch_ref = ref_t.contiguous().reshape(-1).tolist()
    print(f"torch conv3d float64: {time.time() - started:.2f}s")
    if len(torch_ref) != len(out_stream):
        sys.exit(f"shape mismatch: torch {len(torch_ref)} vs dump {len(out_stream)}")

    print()
    print("AGAINST TORCH (all elements) -- the reference this campaign's parity rule names:")
    stream_mean_ulp, stream_rel = score_against("streamed", out_stream, torch_ref)
    direct_mean_ulp, direct_rel = score_against("direct", out_direct, torch_ref)

    # ---- reference 2: the exact answer, on a stratified sample -------------------------
    # SAMPLE_STRIDE is coprime with the channel and spatial extents used here, so the
    # sample walks across channels and positions instead of sitting in one plane.
    indices = list(range(0, len(out_stream), SAMPLE_STRIDE))
    print()
    print(
        f"AGAINST THE EXACT ANSWER (Fraction; {len(indices)} sampled outputs, "
        f"stride {SAMPLE_STRIDE}) -- this one also scores torch:"
    )
    started = time.time()
    exact_ref = []
    for flat in indices:
        w_i = flat % ow
        h_i = (flat // ow) % oh
        d_i = (flat // (ow * oh)) % od
        o_i = (flat // (ow * oh * od)) % out_ch
        b_i = flat // (ow * oh * od * out_ch)
        acc = Fraction(0)
        for c in range(in_ch):
            in_plane = ((b_i * in_ch + c) * pd) * ph * pw
            w_base = ((o_i * in_ch + c) * k) * k * k
            for i in range(k):
                row_d = in_plane + (d_i + i) * ph * pw
                for j in range(k):
                    row = row_d + (h_i + j) * pw + w_i
                    wrow = w_base + (i * k + j) * k
                    for l in range(k):
                        acc += Fraction(padded[row + l]) * Fraction(weight[wrow + l])
        # Correctly-rounded float64 of the exact value. Fraction.__float__ rounds to
        # nearest-even, which is exactly the correctly-rounded result we want.
        exact_ref.append(float(acc))
    print(f"  (exact pass {time.time() - started:.1f}s)")

    sample_stream = [out_stream[i] for i in indices]
    sample_direct = [out_direct[i] for i in indices]
    sample_torch = [torch_ref[i] for i in indices]
    ex_stream, _ = score_against("streamed", sample_stream, exact_ref)
    ex_direct, _ = score_against("direct", sample_direct, exact_ref)
    ex_torch, _ = score_against("torch", sample_torch, exact_ref)

    # ---- verdict -----------------------------------------------------------------------
    print()
    print("VERDICT")
    print(
        f"  distance from torch : streamed {stream_mean_ulp:.3f} ulp, "
        f"direct {direct_mean_ulp:.3f} ulp"
    )
    print(
        f"  distance from EXACT : streamed {ex_stream:.3f} ulp, "
        f"direct {ex_direct:.3f} ulp, torch {ex_torch:.3f} ulp"
    )
    print()
    if stream_mean_ulp <= direct_mean_ulp and ex_stream <= ex_direct:
        print(
            "  The STREAMED route is at least as close to torch AND at least as accurate."
            "\n  Re-gating to it is NOT a parity regression. There is nothing to ratify,"
            "\n  and item 68d's 4.770e-12 is our own kernel's error, not a cost of the fix."
        )
    elif ex_stream <= ex_direct:
        print(
            "  The STREAMED route is MORE ACCURATE but further from torch's own rounding."
            "\n  The re-gate improves the answer; whether it improves PARITY is the call."
        )
    else:
        print(
            "  The DIRECT route is more accurate. The re-gate really would cost accuracy"
            "\n  and needs a genuine policy call, as item 68d supposed."
        )
    print()
    print(f"  item 68d's route-vs-route figure was 4.770e-12; here streamed-vs-torch "
          f"max_rel {stream_rel:.3e}, direct-vs-torch max_rel {direct_rel:.3e}")


if __name__ == "__main__":
    main()
