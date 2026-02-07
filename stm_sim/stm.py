import numpy as np

from .geometry import surface_height_at
from .ldos import atom_params_from_types, compute_rho_grid


def solve_constant_current(scene, x, y, setpoint, ldos_cfg, stm_cfg):
    A, kappa = atom_params_from_types(scene.types, ldos_cfg)
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
            scene.positions,
            A,
            kappa,
            cutoff=stm_cfg.cutoff,
        )
        higher = rho > setpoint
        z_lo = np.where(higher, z_mid, z_lo)
        z_hi = np.where(higher, z_hi, z_mid)

    return 0.5 * (z_lo + z_hi)
