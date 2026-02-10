# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
    _HAS_TIFF = True
except Exception:
    tifffile = None
    _HAS_TIFF = False

from matplotlib import colormaps

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from stm_sim.config import SimulationConfig
from stm_sim.simulator import build_scene_from_config
from stm_sim.utils import ensure_rng, grid_from_bbox
from stm_sim.geometry import add_molecule_lattice
from stm_sim.stm import solve_constant_current
from stm_sim.noise import apply_noise
from stm_sim.labels import generate_mask


def apply_colormap(image, cmap_name="inferno"):
    cmap = colormaps.get_cmap(cmap_name)
    vmin = float(image.min())
    vmax = float(image.max())
    if vmax <= vmin:
        norm = image * 0
    else:
        norm = (image - vmin) / (vmax - vmin)
    rgba = cmap(norm)  # HxWx4
    rgb = (rgba[..., :3] * 255.0).astype("uint8")
    return rgb


def generate_sample(
    cfg,
    rng,
    spacing,
    grid_n,
    orientation_deg,
    lattice_angle_deg,
    out_dir,
    idx,
    spacing_range=None,
    grid_range=None,
    orientation_range=None,
    lattice_angle_range=None,
):
    scene = build_scene_from_config(cfg, rng)

    if orientation_range is not None:
        orientation_deg = float(rng.uniform(orientation_range[0], orientation_range[1]))
    if lattice_angle_range is not None:
        lattice_angle_deg = float(rng.uniform(lattice_angle_range[0], lattice_angle_range[1]))

    lattice_info = add_molecule_lattice(
        scene,
        molecule_name="FePc",
        grid_n=grid_n,
        spacing=spacing,
        orientation_deg=orientation_deg,
        lattice_angle_deg=lattice_angle_deg,
        height=cfg.features.molecule_height,
        rng=rng,
        random_center=True,
        spacing_range=spacing_range,
        grid_range=grid_range,
        molecule_z_scale=cfg.features.molecule_z_scale,
        molecule_xy_scale=cfg.features.molecule_xy_scale,
    )
    if lattice_info is None:
        lattice_info = {"spacing": spacing, "grid_n": grid_n, "center": None}

    base_bbox = scene.metadata.get("surface_bbox", scene.bbox)
    x, y = grid_from_bbox(base_bbox, cfg.image.pixels)
    setpoint = np.mean(cfg.stm.setpoint_range)
    raw_height = solve_constant_current(scene, x, y, setpoint, cfg.ldos, cfg.stm)

    nx = cfg.image.pixels[1]
    ny = cfg.image.pixels[0]
    dx = (base_bbox[0][1] - base_bbox[0][0]) / max(1, nx - 1)
    dy = (base_bbox[1][1] - base_bbox[1][0]) / max(1, ny - 1)
    image = apply_noise(raw_height, cfg.noise, rng, pixel_size=(dx, dy))
    blur_radius = float(np.mean(cfg.noise.tip_sigma))
    mask = generate_mask(scene, cfg.image.pixels, cfg.labels, bbox=base_bbox, blur_radius=blur_radius)

    np.save(out_dir / f"image_{idx:04d}.npy", image.astype(np.float32))
    np.save(out_dir / f"mask_{idx:04d}.npy", mask.astype(np.int64))

    if _HAS_TIFF:
        tifffile.imwrite(out_dir / f"image_{idx:04d}.tiff", image.astype(np.float32))
        tifffile.imwrite(out_dir / f"mask_{idx:04d}.tiff", mask.astype(np.int64))

        # colored quick-look
        image_rgb = apply_colormap(image, cmap_name="inferno")
        mask_rgb = apply_colormap(mask.astype(float), cmap_name="tab20")
        tifffile.imwrite(out_dir / f"image_{idx:04d}_color.tiff", image_rgb)
        tifffile.imwrite(out_dir / f"mask_{idx:04d}_color.tiff", mask_rgb)

    with open(out_dir / f"meta_{idx:04d}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "index": idx,
                "size_angstrom": cfg.surface.size_angstrom,
                "step_edges": scene.step_edges,
                "setpoint": float(setpoint),
                "spacing": lattice_info.get("spacing", spacing),
                "grid_n": lattice_info.get("grid_n", grid_n),
                "center": lattice_info.get("center"),
                "orientation_deg": orientation_deg,
                "lattice_angle_deg": lattice_angle_deg,
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate FePc layer STM dataset")
    parser.add_argument("--out", type=str, default="dataset_fepc_layer", help="Output directory")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--spacing", type=float, default=15.0, help="FePc lattice spacing (Å)")
    parser.add_argument("--spacing_range", type=float, nargs=2, default=None, help="Spacing range min max (Å)")
    parser.add_argument("--grid_range", type=int, nargs=2, default=None, help="Grid range min max")
    parser.add_argument("--grid", type=int, default=3, help="Grid size (NxN)")
    parser.add_argument("--orientation", type=float, default=0.0, help="Molecule orientation (deg)")
    parser.add_argument(
        "--orientation_range",
        type=float,
        nargs=2,
        default=None,
        help="Orientation range min max (deg)",
    )
    parser.add_argument("--lattice_angle", type=float, default=0.0, help="Lattice rotation (deg)")
    parser.add_argument(
        "--lattice_angle_range",
        type=float,
        nargs=2,
        default=None,
        help="Lattice rotation range min max (deg)",
    )
    args = parser.parse_args()

    cfg = SimulationConfig(seed=args.seed)
    cfg.surface.surface = "100"
    cfg.surface.size_angstrom = (200.0, 200.0)
    cfg.features.step_probability = 0.5
    cfg.features.step_height_layers = 1
    cfg.features.adatom_count = (1, 10)
    cfg.features.vacancy_count = (2, 4)
    cfg.features.molecule_count = (0, 1)
    cfg.features.molecule_height = 1.0

    cfg.noise.tip_sigma = (2.0, 3.0)         # lower = sharper tip
    cfg.noise.tip_instability = (0.0, 0.2)   # lower = more stable
    cfg.noise.height_jitter_sigma = (0.0, 0.01)
    cfg.noise.line_noise_sigma = (0.0, 0.02)
    cfg.noise.line_noise_corr = 8.0
    cfg.stm.cutoff = 5.0
    cfg.stm.solver_iters = 5


    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _HAS_TIFF:
        print("tifffile not installed; TIFF outputs will be skipped.")
    rng = ensure_rng(args.seed)

    spacing_range = tuple(args.spacing_range) if args.spacing_range is not None else None
    grid_range = tuple(args.grid_range) if args.grid_range is not None else None
    orientation_range = (
        tuple(args.orientation_range) if args.orientation_range is not None else None
    )
    lattice_angle_range = (
        tuple(args.lattice_angle_range) if args.lattice_angle_range is not None else None
    )

    for i in range(args.n):
        generate_sample(
            cfg,
            rng,
            args.spacing,
            args.grid,
            args.orientation,
            args.lattice_angle,
            out_dir,
            i,
            spacing_range=spacing_range,
            grid_range=grid_range,
            orientation_range=orientation_range,
            lattice_angle_range=lattice_angle_range,
        )


if __name__ == "__main__":
    main()

