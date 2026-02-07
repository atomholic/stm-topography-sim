import numpy as np


def atom_params_from_types(types, cfg):
    n = len(types)
    A = np.empty(n, dtype=float)
    kappa = np.empty(n, dtype=float)
    for i, t in enumerate(types):
        if t == "adatom":
            A[i] = cfg.A_adatom
            kappa[i] = cfg.kappa_adatom
        elif t == "molecule":
            A[i] = cfg.A_molecule
            kappa[i] = cfg.kappa_molecule
        else:
            A[i] = cfg.A_surface
            kappa[i] = cfg.kappa_surface
    return A, kappa


def compute_rho_grid(x, y, z, positions, A, kappa, cutoff: float | None = None, chunk: int = 256):
    rho = np.zeros_like(x, dtype=float)
    n_atoms = positions.shape[0]
    for start in range(0, n_atoms, chunk):
        end = min(n_atoms, start + chunk)
        pos = positions[start:end]
        A_chunk = A[start:end]
        k_chunk = kappa[start:end]
        dx = x[..., None] - pos[:, 0]
        dy = y[..., None] - pos[:, 1]
        dz = z[..., None] - pos[:, 2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        if cutoff is not None:
            mask = r <= cutoff
        else:
            mask = None
        contrib = (A_chunk * A_chunk) * np.exp(-2.0 * k_chunk * r)
        if mask is not None:
            contrib = np.where(mask, contrib, 0.0)
        rho += contrib.sum(axis=-1)
    return rho
