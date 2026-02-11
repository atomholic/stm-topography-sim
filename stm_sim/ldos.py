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


def compute_ypc2_rho_grid(
    x,
    y,
    z,
    centers,
    orientations_deg,
    cfg,
):
    if centers is None or len(centers) == 0:
        return np.zeros_like(x, dtype=float)

    r0 = float(cfg.ypc2_r0)
    sig_r = float(cfg.ypc2_sig_r)
    ang_mix = float(cfg.ypc2_ang_mix)
    kappa = float(cfg.kappa_molecule)
    amp = float(cfg.A_molecule) ** 2
    center_plateau_amp = float(getattr(cfg, "ypc2_center_plateau_amp", 0.0))
    center_plateau_radius = float(getattr(cfg, "ypc2_center_plateau_radius", 2.0))
    center_plateau_sigma = float(getattr(cfg, "ypc2_center_plateau_sigma", 0.6))
    plateau_amp = float(cfg.ypc2_plateau_amp)
    plateau_radius = float(cfg.ypc2_plateau_radius)
    plateau_sigma = float(cfg.ypc2_plateau_sigma)

    rho = np.zeros_like(x, dtype=float)
    centers_arr = np.asarray(centers, dtype=float)
    if orientations_deg is None or len(orientations_deg) != len(centers_arr):
        orientations = np.zeros((len(centers_arr),), dtype=float)
    else:
        orientations = np.asarray(orientations_deg, dtype=float)

    for (cx, cy, cz), ang_deg in zip(centers_arr, orientations):
        ang = np.deg2rad(ang_deg)
        c = np.cos(ang)
        s = np.sin(ang)
        dx = x - cx
        dy = y - cy
        # rotate by -ang so orientation rotates molecule
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        r = np.sqrt(xr * xr + yr * yr)
        theta = np.arctan2(yr, xr)

        ring = np.exp(-0.5 * ((r - r0) / sig_r) ** 2)
        psi1 = ring * np.cos(4.0 * theta)
        psi2 = ring * np.cos(8.0 * theta)
        psi = psi1 + ang_mix * psi2
        rho_xy = psi * psi
        max_val = float(rho_xy.max())
        if max_val > 0:
            rho_xy = rho_xy / max_val

        if plateau_amp > 0.0:
            if plateau_sigma <= 0:
                plateau = (r <= plateau_radius).astype(float)
            else:
                plateau = 1.0 / (1.0 + np.exp((r - plateau_radius) / plateau_sigma))
            rho_xy = rho_xy + (plateau_amp * plateau)

        dz = np.maximum(0.0, z - cz)
        rho += amp * rho_xy * np.exp(-2.0 * kappa * dz)
        if center_plateau_amp > 0.0:
            if center_plateau_sigma <= 0:
                plateau = (r <= center_plateau_radius).astype(float)
            else:
                plateau = 1.0 / (1.0 + np.exp((r - center_plateau_radius) / center_plateau_sigma))
            rho += (center_plateau_amp * center_plateau_amp) * plateau * np.exp(-2.0 * kappa * dz)

    return rho
