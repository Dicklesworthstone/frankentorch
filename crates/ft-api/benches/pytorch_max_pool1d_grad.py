import os
import time

import torch
import torch.nn.functional as F


def main() -> None:
    iters = int(os.environ["FT_GAUNTLET_ITERS"])
    shape = (8, 64, 8192)
    total = shape[0] * shape[1] * shape[2]
    base = torch.arange(total, dtype=torch.float64).reshape(shape)
    base = torch.remainder(base, 251).mul_(0.001).sub_(0.12)

    start = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        x = base.detach().clone().requires_grad_(True)
        out = F.max_pool1d(x, kernel_size=2, stride=2)
        out.sum().backward()
        checksum += float(x.grad.reshape(-1)[0])
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.12f}")
    print(f"checksum={checksum:.12f}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
