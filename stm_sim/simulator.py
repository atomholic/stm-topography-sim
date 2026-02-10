import warnings
import numpy as np

from .config import SimulationConfig
from .geometry import (
    add_adatoms,
    add_molecules,
    add_step,
    add_surface_roughness,
    add_vacancies,
    build_fcc_surface,
)
from .labels import generate_mask
from .noise import apply_noise
from .stm import solve_constant_current
from .utils import ensure_rng, grid_from_bbox, sample_int, sample_uniform


def build_scene_from_config(cfg: SimulationConfig, rng: np.random.Generator):
    size_angstrom = cfg.surface.size_angstrom
    if hasattr(cfg.surface, "size_nm"):
        size_nm = getattr(cfg.surface, "size_nm")
        if size_nm is not None:
            default_size = SimulationConfig().surface.size_angstrom
            if size_angstrom == default_size:
                size_angstrom = (float(size_nm[0]) * 10.0, float(size_nm[1]) * 10.0)
                cfg.surface.size_angstrom = size_angstrom
                warnings.warn(
                    "cfg.surface.size_nm is deprecated; converted to size_angstrom (Å).",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "cfg.surface.size_nm is ignored; use size_angstrom (Å).",
                    stacklevel=2,
                )
    scene = build_fcc_surface(
        surface=cfg.surface.surface,
        size_angstrom=size_angstrom,
        layers=cfg.surface.layers,
        lattice_constant=cfg.surface.lattice_constant,
    )
    scene.metadata["molecule_name"] = cfg.features.molecule_name

    a_surf = float(scene.metadata.get("a_surf", 0.0))
    scene.metadata["step_exclusion_distance"] = cfg.features.step_exclusion_factor * a_surf

    if rng.random() < cfg.features.step_probability:
        angle = None
        if cfg.features.step_angle_deg_range is not None:
            angle = sample_uniform(rng, cfg.features.step_angle_deg_range)
        add_step(
            scene,
            rng=rng,
            height_layers=cfg.features.step_height_layers,
            angle_deg=angle,
        )

    adatom_count = sample_int(rng, cfg.features.adatom_count)
    add_adatoms(scene, adatom_count, rng=rng)

    vacancy_count = sample_int(rng, cfg.features.vacancy_count)
    add_vacancies(scene, vacancy_count, rng=rng)

    molecule_count = sample_int(rng, cfg.features.molecule_count)
    add_molecules(
        scene,
        molecule_count,
        rng=rng,
        height=cfg.features.molecule_height,
        height_sigma=cfg.features.molecule_height_sigma,
        molecule_name=cfg.features.molecule_name,
        molecule_z_scale=cfg.features.molecule_z_scale,
        molecule_xy_scale=cfg.features.molecule_xy_scale,
    )

    add_surface_roughness(scene, cfg.features.roughness_sigma, rng=rng)
    return scene


def generate_sample(cfg: SimulationConfig, seed: int | None = None, debug: bool = False):
    rng = ensure_rng(seed if seed is not None else cfg.seed)
    scene = build_scene_from_config(cfg, rng)

    base_bbox = scene.metadata.get("surface_bbox", scene.bbox)
    x, y = grid_from_bbox(base_bbox, cfg.image.pixels)
    setpoint = sample_uniform(rng, cfg.stm.setpoint_range)

    if debug:
        print("Scene atomic positions (Angstrom):")
        print(scene.positions)
        molecule_positions = scene.positions[scene.types == "molecule"]
        print("FePc molecule atom positions (Angstrom):")
        print(molecule_positions)
        print("Step edges:", scene.step_edges)

    height = solve_constant_current(scene, x, y, setpoint, cfg.ldos, cfg.stm)
    nx = cfg.image.pixels[1]
    ny = cfg.image.pixels[0]
    xmin, xmax = base_bbox[0]
    ymin, ymax = base_bbox[1]
    dx = (xmax - xmin) / max(1, nx - 1)
    dy = (ymax - ymin) / max(1, ny - 1)
    image = apply_noise(height, cfg.noise, rng, pixel_size=(dx, dy)).astype(np.float32)
    blur_radius = float(np.mean(cfg.noise.tip_sigma))
    mask = generate_mask(
        scene,
        cfg.image.pixels,
        cfg.labels,
        bbox=base_bbox,
        blur_radius=blur_radius,
    ).astype(np.int64)

    molecule_positions = scene.positions[scene.types == "molecule"]
    metadata = {
        "surface": scene.surface,
        "lattice_constant": scene.lattice_constant,
        "size_angstrom": scene.size_angstrom,
        "setpoint": float(setpoint),
        "molecule_name": cfg.features.molecule_name,
        "molecule_z_scale": cfg.features.molecule_z_scale,
        "molecule_xy_scale": cfg.features.molecule_xy_scale,
        "atoms": [
            {"type": t, "position": scene.positions[i].tolist()}
            for i, t in enumerate(scene.types.tolist())
        ],
        "molecule_atoms": molecule_positions.tolist(),
        "vacancies": scene.vacancies.tolist(),
        "step_edges": scene.step_edges,
    }

    return image, mask, metadata
