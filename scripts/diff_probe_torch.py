#!/usr/bin/env python3
"""Torch reference for crates/ft-api/examples/diff_probe.rs (Phase-B parity sweep)."""
import torch

torch.set_default_dtype(torch.float64)
NAN, INF = float("nan"), float("inf")


def fmt(t):
    out = []
    for x in t.tolist():
        if x != x:
            out.append("nan")
        elif x == INF:
            out.append("inf")
        elif x == -INF:
            out.append("-inf")
        else:
            out.append(f"{x:.17e}")
    return ",".join(out)


a = torch.tensor([-5.5, -3.0, 3.0, 5.5, -0.0, 0.0, -7.0, 2.5, NAN, INF, -INF, 1.0])
b = torch.tensor([2.0, 2.0, -2.0, 2.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0, 1.0, 0.0])

print("remainder|" + fmt(torch.remainder(a, b)))
print("fmod|" + fmt(torch.fmod(a, b)))
print("floor_divide|" + fmt(torch.floor_divide(a, b)))
print("copysign|" + fmt(torch.copysign(a, b)))
print("nextafter|" + fmt(torch.nextafter(a, b)))
print("hypot|" + fmt(torch.hypot(a, b)))
print("logaddexp|" + fmt(torch.logaddexp(a, b)))
print("fmax|" + fmt(torch.fmax(a, b)))
print("ldexp|" + fmt(torch.ldexp(a, b)))

fa = torch.tensor([1.0, -1.0, 0.0, INF, -INF, INF, -INF, INF, -INF, NAN, 1.0, INF, 5.0, -5.0,
                   7.0, -7.0, 0.3, 2.5, -0.0, 6.5, -6.5])
fb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, NAN, INF, INF, INF,
                   2.0, 2.0, 0.1, 0.5, 3.0, 2.0, 2.0])
print("floor_divide_edge|" + fmt(torch.floor_divide(fa, fb)))

xx = torch.tensor([0.0, 0.0, 2.0, 3.0, 0.5, 1.0])
yy = torch.tensor([0.0, -1.0, 0.0, 2.0, 4.0, -3.0])
print("xlogy|" + fmt(torch.xlogy(xx, yy)))

print("signbit|" + fmt(torch.signbit(a).to(torch.float64)))
c = torch.tensor([0.0, 0.5, 1.0, -1.0, 2.0, -0.5])
print("sinc|" + fmt(torch.sinc(c)))
print("float_power_0.5|" + fmt(torch.float_power(a, 0.5)))
print("nan_to_num|" + fmt(torch.nan_to_num(a, nan=0.0)))
print("heaviside|" + fmt(torch.heaviside(a, b)))

# ── batch 2: NaN-propagation + domain edges ──
ma = torch.tensor([NAN, 1.0, NAN, -INF, INF, 3.0])
mb = torch.tensor([1.0, NAN, NAN, 5.0, 5.0, -3.0])
print("maximum|" + fmt(torch.maximum(ma, mb)))
print("minimum|" + fmt(torch.minimum(ma, mb)))
ya = torch.tensor([0.0, -0.0, 1.0, -1.0, INF, INF, 0.0, NAN])
xa = torch.tensor([1.0, 1.0, 0.0, 0.0, INF, -INF, -1.0, 1.0])
print("atan2|" + fmt(torch.atan2(ya, xa)))
cl = torch.tensor([-1.0, 0.5, 2.0, NAN, INF, -INF])
print("clamp01|" + fmt(torch.clamp(cl, 0.0, 1.0)))
print("clamp_minmax|" + fmt(torch.clamp(cl, 1.0, 0.0)))
print("asin|" + fmt(torch.asin(torch.tensor([2.0, -2.0, 1.0, -1.0, 0.0, NAN]))))
print("acos|" + fmt(torch.acos(torch.tensor([2.0, -2.0, 1.0, -1.0, 0.0, NAN]))))
print("log|" + fmt(torch.log(torch.tensor([-1.0, 0.0, 1.0, INF, -INF, NAN]))))
print("sqrt|" + fmt(torch.sqrt(torch.tensor([-1.0, 0.0, 4.0, INF, -INF, NAN]))))
print("rsqrt|" + fmt(torch.rsqrt(torch.tensor([0.0, 4.0, -1.0, INF, -0.0, NAN]))))
print("log1p|" + fmt(torch.log1p(torch.tensor([-2.0, -1.0, 0.0, INF, -0.5, NAN]))))
print("expm1|" + fmt(torch.expm1(torch.tensor([-INF, 0.0, INF, -1.0, 1.0, NAN]))))
print("lgamma|" + fmt(torch.lgamma(torch.tensor([0.0, -1.0, -2.0, 0.5, INF, -INF]))))
print("digamma|" + fmt(torch.digamma(torch.tensor([0.0, -1.0, -2.0, 0.5, 1.0, INF]))))
print("erfinv|" + fmt(torch.erfinv(torch.tensor([1.0, -1.0, 2.0, -2.0, 0.0, NAN]))))
print("logit|" + fmt(torch.logit(torch.tensor([0.0, 1.0, 0.5, -0.1, 1.1, NAN]))))

# ── batch 3: activation extremes ──
import torch.nn.functional as F
b3 = torch.tensor([-INF, -100.0, -1.0, 0.0, 1.0, 50.0, 100.0, INF, NAN])
print("softplus|" + fmt(F.softplus(b3)))
print("sigmoid|" + fmt(torch.sigmoid(b3)))
print("tanh|" + fmt(torch.tanh(b3)))
print("gelu|" + fmt(F.gelu(b3)))
print("silu|" + fmt(F.silu(b3)))
print("mish|" + fmt(F.mish(b3)))
print("softsign|" + fmt(F.softsign(b3)))
print("erf|" + fmt(torch.erf(b3)))
print("erfc|" + fmt(torch.erfc(b3)))
print("reciprocal|" + fmt(torch.reciprocal(b3)))

# ── batch 4: rounding + exp/log at extremes ──
b4 = torch.tensor([-INF, INF, NAN, -0.0, 2.5, 3.5, -2.5, 0.5, -0.5, 1e20])
print("round|" + fmt(torch.round(b4)))
print("trunc|" + fmt(torch.trunc(b4)))
print("frac|" + fmt(torch.frac(b4)))
print("ceil|" + fmt(torch.ceil(b4)))
print("floor|" + fmt(torch.floor(b4)))
print("sign|" + fmt(torch.sign(b4)))
b4l = torch.tensor([-INF, -1.0, 0.0, -0.0, 1.0, 2.0, 8.0, 710.0, INF, NAN])
print("exp|" + fmt(torch.exp(b4l)))
print("exp2|" + fmt(torch.exp2(b4l)))
print("log2|" + fmt(torch.log2(b4l)))
print("log10|" + fmt(torch.log10(b4l)))
