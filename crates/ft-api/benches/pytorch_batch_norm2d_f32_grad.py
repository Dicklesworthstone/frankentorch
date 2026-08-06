import os
import time

import torch
import torch.nn.functional as F

# frankentorch-wnku0: this arm self-reports which PyTorch it is, before any
# timing. The bench refuses to parse a run without it, and cross-checks that
# every lane reported the same version.
print(f"PT_TORCH_VERSION {torch.__version__}", flush=True)


N = 32
C = 256
H = 28
W = 28


def deterministic_values(n: int, shift: float) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).mul_(0.017).add_(shift).sin_().mul_(0.2)


def main() -> None:
    iters = int(os.environ["FT_GAUNTLET_ITERS"])
    torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS", "32")))
    torch.set_num_interop_threads(int(os.environ.get("FT_TORCH_INTEROP_THREADS", "32")))

    base_x = deterministic_values(N * C * H * W, 0.0).reshape(N, C, H, W)
    base_running_mean = torch.zeros(C, dtype=torch.float32)
    base_running_var = torch.ones(C, dtype=torch.float32)
    base_weight = deterministic_values(C, 1.0)
    base_bias = deterministic_values(C, 2.0)

    warmup_x = base_x.detach().clone().requires_grad_(True)
    warmup_weight = base_weight.detach().clone().requires_grad_(True)
    warmup_bias = base_bias.detach().clone().requires_grad_(True)
    warmup_mean = base_running_mean.detach().clone()
    warmup_var = base_running_var.detach().clone()
    F.batch_norm(
        warmup_x,
        warmup_mean,
        warmup_var,
        warmup_weight,
        warmup_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
    ).sum().backward()

    start = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        x = base_x.detach().clone().requires_grad_(True)
        weight = base_weight.detach().clone().requires_grad_(True)
        bias = base_bias.detach().clone().requires_grad_(True)
        running_mean = base_running_mean.detach().clone()
        running_var = base_running_var.detach().clone()
        out = F.batch_norm(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            training=True,
            momentum=0.1,
            eps=1e-5,
        )
        out.sum().backward()
        checksum += float(x.grad.reshape(-1)[0] + weight.grad[0] + bias.grad[0])
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.12f}")
    print(f"checksum={checksum:.12f}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
