import os
import time

import torch
import torch.nn.functional as F


BATCH = 2
HEADS = 8
SEQ = 512
D = 64
TOTAL = BATCH * HEADS * SEQ * D


def deterministic_values(shift: float) -> torch.Tensor:
    return (
        torch.arange(TOTAL, dtype=torch.float64)
        .mul_(0.017)
        .add_(shift)
        .sin_()
        .mul_(0.2)
        .reshape(BATCH, HEADS, SEQ, D)
    )


def main() -> None:
    iters = int(os.environ["FT_GAUNTLET_ITERS"])
    torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS", "32")))
    torch.set_num_interop_threads(int(os.environ.get("FT_TORCH_INTEROP_THREADS", "32")))

    base_q = deterministic_values(0.0)
    base_k = deterministic_values(1.0)
    base_v = deterministic_values(2.0)

    warmup_q = base_q.detach().clone().requires_grad_(True)
    warmup_k = base_k.detach().clone().requires_grad_(True)
    warmup_v = base_v.detach().clone().requires_grad_(True)
    F.scaled_dot_product_attention(
        warmup_q,
        warmup_k,
        warmup_v,
        dropout_p=0.0,
        is_causal=False,
    ).sum().backward()

    start = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        q = base_q.detach().clone().requires_grad_(True)
        k = base_k.detach().clone().requires_grad_(True)
        v = base_v.detach().clone().requires_grad_(True)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out.sum().backward()
        checksum += float(q.grad.reshape(-1)[0])
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.12f}")
    print(f"checksum={checksum:.12f}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
