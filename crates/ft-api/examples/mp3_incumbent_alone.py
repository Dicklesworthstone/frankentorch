# Cross-project check: does OUR arm slow the incumbent? Time torch's max_pool3d step
# ALONE at the harness's exact lane shape, no FrankenTorch in the process.
import time, torch
import torch.nn.functional as Fn
torch.set_num_threads(8)
def seq(n): return ((torch.arange(n,dtype=torch.int64)%251).double())*0.001-0.12
mp3 = seq(2*32*16*32*32).reshape(2,32,16,32,32)
def one():
    x = mp3.clone().requires_grad_(True)
    t = time.perf_counter()
    out = Fn.max_pool3d(x,(2,2,2),(2,2,2))
    out.sum().backward()
    return (time.perf_counter()-t)*1e3
for _ in range(32): one()
s = sorted(one() for _ in range(64))
print(f"torch {torch.__version__} max_pool3d [2,32,16,32,32] alone, 64 samples after 32 warmup")
print(f"  min {s[0]:.3f} ms   median {s[len(s)//2]:.3f} ms   p90 {s[int(len(s)*0.9)]:.3f} ms")
