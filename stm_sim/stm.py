import numpy as np

from .geometry import surface_height_at
from .ldos import atom_params_from_types, compute_rho_grid, compute_ypc2_rho_grid


def solve_constant_current(scene, x, y, setpoint, ldos_cfg, stm_cfg):
    molecule_name = str(scene.metadata.get("molecule_name", "")).lower()
    surface_mode = str(getattr(ldos_cfg, "surface_mode", "atomic")).lower()
    base_mask = np.ones(scene.types.shape[0], dtype=bool)
    if molecule_name == "ypc2":
        base_mask &= scene.types != "molecule"
    if surface_mode == "continuum":
        base_mask &= scene.types != "surface"

    positions = scene.positions[base_mask]
    types = scene.types[base_mask]

    A, kappa = atom_params_from_types(types, ldos_cfg)
    xy = np.stack([x, y], axis=-1).reshape(-1, 2)
    base_z = surface_height_at(scene, xy).reshape(x.shape)
    z_lo = base_z - stm_cfg.z_margin_low
    global_max_z = float(scene.positions[:, 2].max())
    z_hi = np.maximum(base_z + stm_cfg.z_margin_high, global_max_z + stm_cfg.z_margin_high)

    for _ in range(stm_cfg.solver_iters):
        z_mid = 0.5 * (z_lo + z_hi)
        rho = compute_rho_grid(
            x,
            y,
            z_mid,
            positions,
            A,
            kappa,
            cutoff=stm_cfg.cutoff,
        )
        if surface_mode == "continuum":
            dz_surface = np.maximum(0.0, z_mid - base_z)
            rho += (float(ldos_cfg.A_surface) ** 2) * np.exp(
                -2.0 * float(ldos_cfg.kappa_surface) * dz_surface
            )
        if molecule_name == "ypc2":
            rho += compute_ypc2_rho_grid(
                x,
                y,
                z_mid,
                scene.metadata.get("molecule_centers", []),
                scene.metadata.get("molecule_orientations_deg", []),
                ldos_cfg,
            )
        higher = rho > setpoint
        z_lo = np.where(higher, z_mid, z_lo)
        z_hi = np.where(higher, z_hi, z_mid)

    return 0.5 * (z_lo + z_hi)
