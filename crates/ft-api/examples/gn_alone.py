# frankentorch-uilzh: the incumbent arm ALONE, at the resized lane's shape, with no
# FrankenTorch in the process. If torch's A/A null goes clean here while it fails at
# 0.867-0.900 inside the harness, the cause is the INTERLEAVING (MossyOtter's item 27)
# and no amount of warm-up fixes it -- which is what FT_H2H_WARMUP=16 already showed.
#
# Deliberately mirrors the harness: 8 threads, same shape and groups, same timed region
# (forward + sum + backward), same warm-up count, same round count.
import time, sys, torch
import torch.nn.functional as Fn

torch.set_num_threads(8)

GN_N, GN_C, GN_H, GN_W, GROUPS = 32, 64, 56, 56, 32
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def seq(n):
    return ((torch.arange(n, dtype=torch.int64) % 251).double()) * 0.001 - 0.12


gnx = seq(GN_N * GN_C * GN_H * GN_W).reshape(GN_N, GN_C, GN_H, GN_W).float()
gnw = (seq(GN_C) * 10.0 + 1.0).float().requires_grad_(True)
gnb = (seq(GN_C) * 3.0).float().requires_grad_(True)


def one_sample():
    x = gnx.clone().requires_grad_(True)
    # Leaf construction outside the timer, as the harness does on both arms.
    t = time.perf_counter()
    out = Fn.group_norm(x, GROUPS, gnw, gnb)
    loss = out.sum()
    loss.backward()
    elapsed = (time.perf_counter() - t) * 1e3
    gnw.grad = None
    gnb.grad = None
    return elapsed


for _ in range(WARMUP):
    one_sample()

samples = [one_sample() for _ in range(ROUNDS)]


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


half = ROUNDS // 2
first, second = samples[:half], samples[half:]
null_median = median(first) / median(second)
null_min = min(first) / min(second)

print(f"torch {torch.__version__} threads=8 shape=[{GN_N},{GN_C},{GN_H},{GN_W}] groups={GROUPS}")
print(f"rounds={ROUNDS} warmup={WARMUP}")
print(f"  median all      {median(samples):.4f} ms")
print(f"  first half      {median(first):.4f} ms   min {min(first):.4f}")
print(f"  second half     {median(second):.4f} ms   min {min(second):.4f}")
print(f"  A/A null median {null_median:.4f}   (>1 = second half FASTER)")
print(f"  A/A null min    {null_min:.4f}")
print("  first 8:", " ".join(f"{v:.3f}" for v in samples[:8]))
print("  last 8: ", " ".join(f"{v:.3f}" for v in samples[-8:]))
