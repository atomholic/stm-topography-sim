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
        molecule_name="YPc2",
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
        adatom_on_top_count=cfg.features.ypc2_adatom_on_top_count,
        adatom_on_top_height=cfg.features.ypc2_adatom_on_top_height,
        adatom_on_top_radial_offset=cfg.features.ypc2_adatom_on_top_radial_offset,
        adatom_on_top_radial_jitter=cfg.features.ypc2_adatom_on_top_radial_jitter,
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
                "molecule_name": "YPc2",
                "ypc2_params": {
                    "r0": cfg.ldos.ypc2_r0,
                    "sig_r": cfg.ldos.ypc2_sig_r,
                    "ang_mix": cfg.ldos.ypc2_ang_mix,
                    "center_amp": cfg.ldos.ypc2_center_amp,
                    "center_kappa": cfg.ldos.ypc2_center_kappa,
                    "plateau_amp": cfg.ldos.ypc2_plateau_amp,
                    "plateau_radius": cfg.ldos.ypc2_plateau_radius,
                    "plateau_sigma": cfg.ldos.ypc2_plateau_sigma,
                },
                "adatom_on_top": {
                    "count_range": cfg.features.ypc2_adatom_on_top_count,
                    "height": cfg.features.ypc2_adatom_on_top_height,
                    "radial_offset": cfg.features.ypc2_adatom_on_top_radial_offset,
                    "radial_jitter": cfg.features.ypc2_adatom_on_top_radial_jitter,
                },
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate YPc2 layer STM dataset")
    parser.add_argument("--out", type=str, default="dataset_ypc2_layer", help="Output directory")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--spacing", type=float, default=15.0, help="YPc2 lattice spacing (Å)")
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
    parser.add_argument("--ypc2_r0", type=float, default=None, help="YPc2 ring radius (Å)")
    parser.add_argument("--ypc2_sig_r", type=float, default=None, help="YPc2 ring thickness (Å)")
    parser.add_argument("--ypc2_ang_mix", type=float, default=None, help="YPc2 angular mix")
    parser.add_argument("--ypc2_center_amp", type=float, default=None, help="YPc2 center amp")
    parser.add_argument("--ypc2_center_kappa", type=float, default=None, help="YPc2 center kappa")
    parser.add_argument("--adatom_count", type=int, nargs=2, default=None, help="Adatom count range min max")
    parser.add_argument("--vacancy_count", type=int, nargs=2, default=None, help="Vacancy count range min max")
    parser.add_argument("--molecule_count", type=int, nargs=2, default=None, help="Random molecule count range min max")
    parser.add_argument("--step_probability", type=float, default=None, help="Step probability (0-1)")
    parser.add_argument("--step_height_layers", type=int, default=None, help="Step height in layers")
    parser.add_argument("--roughness_sigma", type=float, default=None, help="Surface roughness sigma (Å)")
    parser.add_argument("--adatom_on_top", type=int, nargs=2, default=None, help="Adatom-on-top count range min max")
    parser.add_argument("--adatom_on_top_height", type=float, default=2.0, help="Adatom-on-top height (Å)")
    parser.add_argument("--adatom_on_top_offset", type=float, default=None, help="Adatom-on-top radial offset (Å)")
    parser.add_argument("--adatom_on_top_jitter", type=float, default=None, help="Adatom-on-top radial jitter (Å)")
    parser.add_argument("--vibration_amp", type=float, nargs=2, default=None, help="Vibration amplitude range (Å)")
    parser.add_argument("--vibration_wavelength", type=float, nargs=2, default=None, help="Vibration wavelength range (Å)")
    parser.add_argument("--vibration_angle", type=float, nargs=2, default=None, help="Vibration angle range (deg)")
    parser.add_argument("--slope_x", type=float, nargs=2, default=None, help="Slope X range (Å across image)")
    parser.add_argument("--slope_y", type=float, nargs=2, default=None, help="Slope Y range (Å across image)")
    parser.add_argument("--realistic", action="store_true", help="Apply realism preset tuned to STM appearance")
    args = parser.parse_args()

    cfg = SimulationConfig(seed=args.seed)
    cfg.surface.surface = "100"
    cfg.surface.size_angstrom = (120.0, 120.0)
    cfg.features.step_probability = 0.5
    cfg.features.step_height_layers = 1
    cfg.features.adatom_count = (1, 10)
    cfg.features.vacancy_count = (2, 4)
    cfg.features.molecule_count = (0, 1)
    cfg.features.molecule_height = 5.0
    cfg.features.molecule_name = "YPc2"
    cfg.features.ypc2_adatom_on_top_count = (0, 0)
    cfg.features.ypc2_adatom_on_top_height = float(args.adatom_on_top_height)
    cfg.features.ypc2_adatom_on_top_radial_offset = 0.0
    cfg.features.ypc2_adatom_on_top_radial_jitter = 0.0

    cfg.ldos.A_molecule = 1.0
    cfg.ldos.ypc2_r0 = 5.8
    cfg.ldos.ypc2_sig_r = 1.75
    cfg.ldos.ypc2_ang_mix = 0.0
    cfg.ldos.ypc2_center_amp = 1.2
    cfg.ldos.ypc2_center_kappa = 3.0

    if args.ypc2_r0 is not None:
        cfg.ldos.ypc2_r0 = float(args.ypc2_r0)
    if args.ypc2_sig_r is not None:
        cfg.ldos.ypc2_sig_r = float(args.ypc2_sig_r)
    if args.ypc2_ang_mix is not None:
        cfg.ldos.ypc2_ang_mix = float(args.ypc2_ang_mix)
    if args.ypc2_center_amp is not None:
        cfg.ldos.ypc2_center_amp = float(args.ypc2_center_amp)
    if args.ypc2_center_kappa is not None:
        cfg.ldos.ypc2_center_kappa = float(args.ypc2_center_kappa)

    if args.adatom_count is not None:
        cfg.features.adatom_count = (int(args.adatom_count[0]), int(args.adatom_count[1]))
    if args.vacancy_count is not None:
        cfg.features.vacancy_count = (int(args.vacancy_count[0]), int(args.vacancy_count[1]))
    if args.molecule_count is not None:
        cfg.features.molecule_count = (int(args.molecule_count[0]), int(args.molecule_count[1]))
    if args.step_probability is not None:
        cfg.features.step_probability = float(args.step_probability)
    if args.step_height_layers is not None:
        cfg.features.step_height_layers = int(args.step_height_layers)
    if args.roughness_sigma is not None:
        cfg.features.roughness_sigma = float(args.roughness_sigma)

    if args.adatom_on_top is not None:
        cfg.features.ypc2_adatom_on_top_count = (int(args.adatom_on_top[0]), int(args.adatom_on_top[1]))
    if args.adatom_on_top_offset is not None:
        cfg.features.ypc2_adatom_on_top_radial_offset = float(args.adatom_on_top_offset)
    if args.adatom_on_top_jitter is not None:
        cfg.features.ypc2_adatom_on_top_radial_jitter = float(args.adatom_on_top_jitter)

    cfg.noise.tip_sigma = (0.5, 1.2)
    cfg.noise.gaussian_noise_sigma = (0.0, 0.0)
    cfg.noise.line_noise_sigma = (0.0, 0.0)
    cfg.noise.height_jitter_sigma = (0.0, 0.0)
    cfg.noise.tip_instability = (0.0, 0.0)
    cfg.noise.drift_shear = (0.0, 0.0)
    cfg.noise.background_sigma = (0.0, 0.0)
    cfg.noise.background_corr = 15.0
    cfg.noise.vibration_amp = (0.0, 0.0)
    cfg.noise.vibration_wavelength = (8.0, 20.0)
    cfg.noise.vibration_angle_deg = (0.0, 180.0)
    cfg.noise.slope_x = (0.0, 0.0)
    cfg.noise.slope_y = (0.0, 0.0)
    cfg.stm.cutoff = 5.0
    cfg.stm.solver_iters = 5
    cfg.stm.setpoint_range = (0.005, 0.005)

    if args.vibration_amp is not None:
        cfg.noise.vibration_amp = (float(args.vibration_amp[0]), float(args.vibration_amp[1]))
    if args.vibration_wavelength is not None:
        cfg.noise.vibration_wavelength = (
            float(args.vibration_wavelength[0]),
            float(args.vibration_wavelength[1]),
        )
    if args.vibration_angle is not None:
        cfg.noise.vibration_angle_deg = (float(args.vibration_angle[0]), float(args.vibration_angle[1]))
    if args.slope_x is not None:
        cfg.noise.slope_x = (float(args.slope_x[0]), float(args.slope_x[1]))
    if args.slope_y is not None:
        cfg.noise.slope_y = (float(args.slope_y[0]), float(args.slope_y[1]))

    if args.realistic:
        cfg.noise.tip_sigma = (1.0, 1.6)
        cfg.noise.gaussian_noise_sigma = (0.0, 0.01)
        cfg.noise.line_noise_sigma = (0.005, 0.02)
        cfg.noise.height_jitter_sigma = (0.0, 0.015)
        cfg.noise.tip_instability = (0.0, 0.1)
        cfg.noise.drift_shear = (0.0, 0.01)
        cfg.noise.background_sigma = (0.008, 0.03)
        cfg.noise.background_corr = 10.0
        cfg.noise.vibration_amp = (0.0, 0.02)
        cfg.noise.vibration_wavelength = (8.0, 16.0)
        cfg.noise.vibration_angle_deg = (0.0, 180.0)
        cfg.noise.slope_x = (-0.3, 0.3)
        cfg.noise.slope_y = (-0.3, 0.3)
        cfg.ldos.A_adatom = 2.0
        cfg.ldos.kappa_adatom = 0.95
        cfg.ldos.ypc2_plateau_amp = 0.35
        cfg.ldos.ypc2_plateau_radius = 6.5
        cfg.ldos.ypc2_plateau_sigma = 0.7
        cfg.features.ypc2_adatom_on_top_radial_offset = 4.5
        cfg.features.ypc2_adatom_on_top_radial_jitter = 0.6

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
