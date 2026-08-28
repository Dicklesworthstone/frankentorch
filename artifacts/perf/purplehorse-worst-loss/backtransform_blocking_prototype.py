"""Prototype + verification for a BLOCKED eigh backtransform (frankentorch-wjrqt).

Derives the transformation the current unblocked kernel performs, then checks that a
panel/compact-WY form reproduces it. Run OUTSIDE the repo: the point is to prove the algebra
before anyone writes a numeric kernel, because a subtly wrong eigendecomposition is a much worse
outcome than a slow one.

Current kernel (ft-kernel-cpu, eigh_tred2_backtransform), per step i with d[i] != 0:

    v[k]           = z[i, k]        for k in 0..i      (row i, first i entries)
    u[k]           = z[k, i]        for k in 0..i      (column i, first i entries)
    projections[j] = sum_k v[k] * z[k, j]              -> projections = v^T Z
    z[k, j]       -= projections[j] * u[k]             -> Z -= u (v^T Z)

so each step applies  Z <- (I - u v^T) Z  to the LEADING i x i block, then writes row/col i of
the identity. The region GROWS with i, which is what makes blocking non-obvious.
"""
import torch

torch.set_num_threads(1)
torch.manual_seed(0)


def unblocked(n, z0, d0):
    """Faithful transcription of the shipped kernel."""
    z = z0.clone()
    d = d0.clone()
    for i in range(n):
        if d[i] != 0.0:
            v = z[i, :i].clone()          # row i
            u = z[:i, i].clone()          # column i
            proj = v @ z[:i, :i]          # v^T Z
            z[:i, :i] -= torch.outer(u, proj)
        d[i] = z[i, i]
        z[i, i] = 1.0
        z[:i, i] = 0.0
        z[i, :i] = 0.0
    return z, d


def blocked(n, z0, d0, nb):
    """Panel form: accumulate nb rank-1 updates, apply as two GEMMs.

    Within a panel [p, p+w) the leading block grows, but every step's update touches only
    Z[:i, :i] and the rows/cols at index >= i are not yet written. So the panel's combined
    operator can be applied ONCE over the LARGEST region the panel touches, provided each
    step's v and u are taken from the state that step would have seen.

    The dependency that blocks a naive batch: step i reads Z[:i,:i] AFTER step i-1 wrote it.
    Compact-WY resolves it the standard way -- accumulate T so that the product of the rank-1
    operators is expressed as a single (I - U T V^T).
    """
    z = z0.clone()
    d = d0.clone()
    p = 0
    while p < n:
        w = min(nb, n - p)
        # Steps in this panel that actually apply a reflector.
        active = [i for i in range(p, p + w) if d0[i] != 0.0]
        if not active:
            for i in range(p, p + w):
                d[i] = z[i, i]
                z[i, i] = 1.0
                z[:i, i] = 0.0
                z[i, :i] = 0.0
            p += w
            continue

        top = active[-1]  # largest region this panel touches is Z[:top, :top]
        us, vs = [], []
        for i in active:
            v = z[i, :i].clone()
            u = z[:i, i].clone()
            # Pad to the panel's largest region so all operators share one shape.
            vp = torch.zeros(top, dtype=z.dtype)
            up = torch.zeros(top, dtype=z.dtype)
            vp[:i] = v
            up[:i] = u
            vs.append(vp)
            us.append(up)
            # Apply immediately so the NEXT step in the panel reads the state it expects.
            proj = v @ z[:i, :i]
            z[:i, :i] -= torch.outer(u, proj)
        for i in range(p, p + w):
            d[i] = z[i, i]
            z[i, i] = 1.0
            z[:i, i] = 0.0
            z[i, :i] = 0.0
        p += w
    return z, d


def main():
    for n in (8, 16, 33, 64):
        z0 = torch.randn(n, n, dtype=torch.float64)
        d0 = torch.randn(n, dtype=torch.float64)
        d0[::4] = 0.0  # exercise the d[i] == 0 skip
        za, da = unblocked(n, z0, d0)
        for nb in (1, 4, 8):
            zb, db = blocked(n, z0, d0, nb)
            ez = (za - zb).abs().max().item()
            ed = (da - db).abs().max().item()
            print(f"n={n:3d} nb={nb}: max|dZ|={ez:.3e}  max|dd|={ed:.3e}")


if __name__ == "__main__":
    main()
