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
    # Height from local surface to molecule top (Å)
    molecule_height: float = 2.0
    molecule_height_sigma: float = 0.15


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
    drift_shear: Tuple[float, float] = (0.0, 0.02)
    height_jitter_sigma: Tuple[float, float] = (0.0, 0.02)
    tip_instability: Tuple[float, float] = (0.0, 0.6)


@dataclass
class LabelConfig:
    adatom_radius: float = 1.2
    molecule_radius: float = 1.5
    vacancy_radius: float = 1.2
    step_edge_width: float = 2.0


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
