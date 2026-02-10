from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SurfaceConfig:
    surface: str = "100"
    # Size in Angstroms (Å)
    size_angstrom: Tuple[float, float] = (50.0, 50.0)
    layers: int = 1
    lattice_constant: float = 4.09


@dataclass
class FeatureConfig:
    adatom_count: Tuple[int, int] = (0, 6)
    vacancy_count: Tuple[int, int] = (0, 6)
    step_probability: float = 0.0
    step_height_layers: int = 1
    step_angle_deg_range: Tuple[float, float] = (0.0, 180.0)
    step_exclusion_factor: float = 1.0
    roughness_sigma: float = 0.02
    molecule_count: Tuple[int, int] = (0, 2)
    molecule_name: str = "FePc"
    molecule_z_scale: float = 1.0
    molecule_xy_scale: float = 1.0
    # Height from local surface to molecule top (Å)
    molecule_height: float = 2.0
    molecule_height_sigma: float = 0.15
    # Adatom-on-top for YPc2 lattice (count applies only to lattice placements)
    ypc2_adatom_on_top_count: Tuple[int, int] = (0, 0)
    ypc2_adatom_on_top_height: float = 2.0
    ypc2_adatom_on_top_radial_offset: float = 0.0
    ypc2_adatom_on_top_radial_jitter: float = 0.0


@dataclass
class ImageConfig:
    pixels: Tuple[int, int] = (128, 128)


@dataclass
class LDOSConfig:
    A_surface: float = 1.0
    kappa_surface: float = 1.0
    A_adatom: float = 1.2
    kappa_adatom: float = 1.0
    A_molecule: float = 0.8
    kappa_molecule: float = 1.1
    # Surface LDOS mode: "continuum" (default) or "atomic"
    surface_mode: str = "continuum"
    # YPc2 ring-orbital LDOS parameters (Angstrom-based)
    ypc2_r0: float = 1.8
    ypc2_sig_r: float = 0.45
    ypc2_ang_mix: float = 0.0
    # Optional central lobe to lift the molecule center
    ypc2_center_amp: float = 0.0
    ypc2_center_kappa: float = 1.1
    # Optional plateau term for YPc2 (soft disk)
    ypc2_plateau_amp: float = 0.0
    ypc2_plateau_radius: float = 5.0
    ypc2_plateau_sigma: float = 0.8


@dataclass
class STMConfig:
    setpoint_range: Tuple[float, float] = (0.05, 0.15)
    z_margin_low: float = 0.5
    z_margin_high: float = 6.0
    solver_iters: int = 14
    cutoff: float = 12.0


@dataclass
class NoiseConfig:
    # Tip blur sigma in Angstroms (Å)
    tip_sigma: Tuple[float, float] = (2.5, 4.0)
    gaussian_noise_sigma: Tuple[float, float] = (0.0, 0.02)
    line_noise_sigma: Tuple[float, float] = (0.0, 0.03)
    # Line-noise correlation length in Angstroms (Å)
    line_noise_corr: float = 8.0
    # Low-frequency background variation (Angstroms)
    background_sigma: Tuple[float, float] = (0.0, 0.0)
    background_corr: float = 15.0
    drift_shear: Tuple[float, float] = (0.0, 0.02)
    height_jitter_sigma: Tuple[float, float] = (0.0, 0.02)
    tip_instability: Tuple[float, float] = (0.0, 0.6)
    # Vibration-like sinusoidal noise (Angstroms)
    vibration_amp: Tuple[float, float] = (0.0, 0.0)
    vibration_wavelength: Tuple[float, float] = (8.0, 20.0)
    vibration_angle_deg: Tuple[float, float] = (0.0, 180.0)
    # Global slope in x/y (Angstroms across full image)
    slope_x: Tuple[float, float] = (0.0, 0.0)
    slope_y: Tuple[float, float] = (0.0, 0.0)


@dataclass
class LabelConfig:
    adatom_radius: float = 1.2
    molecule_radius: float = 1.5
    vacancy_radius: float = 1.2
    step_edge_width: float = 2.0
    adatom_on_ypc2_radius: float = 1.2


@dataclass
class SimulationConfig:
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    ldos: LDOSConfig = field(default_factory=LDOSConfig)
    stm: STMConfig = field(default_factory=STMConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    seed: int | None = None
