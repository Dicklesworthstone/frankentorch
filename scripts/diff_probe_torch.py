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
