import os
import time

import torch
import torch.nn.functional as F

# frankentorch-wnku0: this arm self-reports which PyTorch it is, before any
# timing. The bench refuses to parse a run without it, and cross-checks that
# every lane reported the same version.
print(f"PT_TORCH_VERSION {torch.__version__}", flush=True)


BATCH = 32
IN_FEATURES = 512
HIDDEN = 2048


def deterministic_values(n: int, shift: float) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2)


def main() -> None:
    iters = int(os.environ["FT_GAUNTLET_ITERS"])
    torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS", "32")))
    torch.set_num_interop_threads(int(os.environ.get("FT_TORCH_INTEROP_THREADS", "32")))

    base_x = deterministic_values(BATCH * IN_FEATURES, 0.0).reshape(BATCH, IN_FEATURES)
    base_w = deterministic_values(HIDDEN * IN_FEATURES, 1.0).reshape(HIDDEN, IN_FEATURES)
    base_bias = deterministic_values(HIDDEN, 2.0)

    start = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        x = base_x.detach().clone().requires_grad_(True)
        w = base_w.detach().clone().requires_grad_(True)
        bias = base_bias.detach().clone().requires_grad_(True)
        out = F.linear(x, w, bias)
        out.sum().backward()
        checksum += float(x.grad.reshape(-1)[0])
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.12f}")
    print(f"checksum={checksum:.12f}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
