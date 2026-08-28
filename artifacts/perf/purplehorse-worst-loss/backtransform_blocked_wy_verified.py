import torch
torch.set_num_threads(1); torch.manual_seed(0)

def make_realistic(n):
    """Reflectors with the scaling tred2 actually produces: u = v/h, h = ||v||^2/2, so
    (I - u v^T) = I - 2 v v^T/||v||^2 is an ORTHOGONAL Householder reflector. Random u,v are
    NOT valid tred2 output and give operators whose products amplify catastrophically."""
    z0 = torch.zeros(n, n, dtype=torch.float64)
    d0 = torch.randn(n, dtype=torch.float64)
    d0[::4] = 0.0
    for i in range(n):
        if d0[i] != 0.0 and i > 0:
            v = torch.randn(i, dtype=torch.float64)
            h = (v @ v) / 2.0
            z0[i, :i] = v            # v_i
            z0[:i, i] = v / h        # u_i = v/h
    return z0, d0

def forward_shipped(n, z0, d0):
    z, d = z0.clone(), d0.clone()
    for i in range(n):
        if d[i] != 0.0:
            v = z[i, :i].clone(); u = z[:i, i].clone()
            z[:i, :i] -= torch.outer(u, v @ z[:i, :i])
        d[i] = z[i, i]; z[i, i] = 1.0; z[:i, i] = 0.0; z[i, :i] = 0.0
    return z

def blocked_wy(n, z0, d0, nb):
    q = torch.eye(n, dtype=z0.dtype)
    p = 0
    while p < n:
        w = min(nb, n - p)
        active = [i for i in range(p, p + w) if d0[i] != 0.0]
        if active:
            m = active[-1]; k = len(active)
            U = torch.zeros(m, k, dtype=z0.dtype); V = torch.zeros(m, k, dtype=z0.dtype)
            for c, i in enumerate(active):
                U[:i, c] = z0[:i, i]; V[:i, c] = z0[i, :i]
            T = torch.zeros(k, k, dtype=z0.dtype); T[0, 0] = 1.0
            for c in range(1, k):
                T[c, :c] = -(V[:, c] @ U[:, :c]) @ T[:c, :c]
                T[c, c] = 1.0
            if m > 0:
                q[:m, :m] -= U @ (T @ (V.T @ q[:m, :m]))
        p += w
    return q

for n in (16, 33, 64, 128, 256):
    z0, d0 = make_realistic(n)
    zf = forward_shipped(n, z0, d0)
    scale = zf.abs().max().item()
    line = f"n={n:4d} (|Q|max={scale:.2f}) "
    for nb in (4, 8, 16, 32):
        zb = blocked_wy(n, z0, d0, nb)
        line += f" nb={nb}:{(zf - zb).abs().max():.2e}"
    print(line)
