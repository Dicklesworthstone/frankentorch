# Bulk-timed: amortize per-call Python/dispatch overhead over many iterations, and drop
# the per-rep .item() sync that the first probe used.
import time, torch
torch.set_num_threads(8)
N = 1024 * 1024
ITERS = 200

held = torch.empty(N, dtype=torch.float64)

def bulk(fn, iters=ITERS):
    fn()  # warm
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t) * 1e3 / iters)
    return best

print(f"torch {torch.__version__} N={N} f64 = {N*8/2**20:.0f} MiB, bulk min-of-5 over {ITERS} iters")
print(f"  held.zero_()        {bulk(lambda: held.zero_()):.4f} ms/call  (dirty memset)")
print(f"  torch.zeros(N)      {bulk(lambda: torch.zeros(N, dtype=torch.float64)):.4f} ms/call")
print(f"  torch.empty(N)      {bulk(lambda: torch.empty(N, dtype=torch.float64)):.4f} ms/call")
