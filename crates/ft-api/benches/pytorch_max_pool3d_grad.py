import os
import time

import torch
import torch.nn.functional as F

# frankentorch-wnku0: this arm self-reports which PyTorch it is, before any
# timing. The bench refuses to parse a run without it, and cross-checks that
# every lane reported the same version.
print(f"PT_TORCH_VERSION {torch.__version__}", flush=True)


def main() -> None:
    iters = int(os.environ["FT_GAUNTLET_ITERS"])
    shape = (2, 32, 16, 32, 32)
    total = shape[0] * shape[1] * shape[2] * shape[3] * shape[4]
    base = torch.arange(total, dtype=torch.float64).reshape(shape)
    base = torch.remainder(base, 251).mul_(0.001).sub_(0.12)

    warmup = base.detach().clone().requires_grad_(True)
    F.max_pool3d(warmup, kernel_size=(2, 2, 2), stride=(2, 2, 2)).sum().backward()

    start = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        x = base.detach().clone().requires_grad_(True)
        out = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        out.sum().backward()
        checksum += float(x.grad.reshape(-1)[0])
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.12f}")
    print(f"checksum={checksum:.12f}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
