import os
import time

import torch
import torch.nn.functional as F

# frankentorch-wnku0: this arm self-reports which PyTorch it is, before any
# timing. The bench refuses to parse a run without it, and cross-checks that
# every lane reported the same version.
print(f"PT_TORCH_VERSION {torch.__version__}", flush=True)


def deterministic_values(total: int, shift: float) -> torch.Tensor:
    idx = torch.arange(total, dtype=torch.float64)
    return torch.sin(idx.mul(0.017).add(shift)).mul_(0.2)


def main() -> None:
    iters = int(os.environ["FT_GAUNTLET_ITERS"])
    torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS", "32")))
    torch.set_num_interop_threads(int(os.environ.get("FT_TORCH_INTEROP_THREADS", "32")))

    x_shape = (8, 64, 32, 32)
    w_shape = (64, 64, 3, 3)
    x_base = deterministic_values(torch.prod(torch.tensor(x_shape)).item(), 0.0).reshape(x_shape)
    w_base = deterministic_values(torch.prod(torch.tensor(w_shape)).item(), 1.0).reshape(w_shape)

    x = x_base.detach().clone().requires_grad_(True)
    weight = w_base.detach().clone().requires_grad_(True)
    F.conv2d(x, weight, None, stride=(1, 1), padding=(1, 1)).sum().backward()

    start = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        x = x_base.detach().clone().requires_grad_(True)
        weight = w_base.detach().clone().requires_grad_(True)
        out = F.conv2d(x, weight, None, stride=(1, 1), padding=(1, 1))
        out.sum().backward()
        checksum += float(x.grad.reshape(-1)[0]) + float(weight.grad.reshape(-1)[0])
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.12f}")
    print(f"checksum={checksum:.12f}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
