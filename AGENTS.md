# AGENTS.md

## Project Overview
This repository implements a synthetic STM topography simulator to generate labeled datasets for semantic segmentation.

## Recent Changes (Summary)

### Core Simulator
- Implemented modular STM simulator package under `stm_sim/` including:
  - Surface/scene generation (Ag(111)/Ag(100)).
  - Defects: adatoms, vacancies, steps, roughness.
  - Molecules: FePc from `molecules/FePc.mol` and YPc2 from `molecules/YPc2.xyz`.
  - s-wave LDOS approximation and constant-current solver.
  - Noise + tip instability models.
  - Semantic mask generator.
  - Dataset generator scripts.
  - YPc2 ring-orbital LDOS model (center/plateau terms) with per-molecule orientation.
  - Continuum surface LDOS mode (default) to remove atomic corrugation from substrate.
  - Adatom-on-YPc2 lattice support with separate mask label (class 6).
  - Multi-tip (double/triple) convolution mode with per-tip offsets, weights, and blur.
  - Corner step edges with optional rounded height transition and mask rounding.
  - Multi-island YPc2 lattices with per-island orientation/angle and edge-biased removals.
  - YPc2 central plateau LDOS term for flatter molecule center.
  - Adatom-on-YPc2 width control for broader LDOS and mask footprint.

### Units & Grid
- Standardized lengths to **Angstroms ()** in config and geometry.
- Fixed imaging grid to the **surface bounding box** so molecules do not expand the image.
- Blur sigma and line-noise correlation are **-based** and converted to pixels using grid spacing.

### Steps & Terraces
- Single-layer surface by default (`layers = 1`).
- Step edges shift surface atoms upward instead of stacking layers.
- Adatoms/molecules placed on the local terrace height.
- Step edge labels support arbitrary angle steps.

### Visualization
- Added 2D scene plot with color-coded features and legend.
- Added 3D scene plot with optional fixed `zlim`.

### Molecule Support
- Added `molecule_name`, `molecule_z_scale`, and `molecule_xy_scale` in config.
- Molecule loader supports `.mol` and `.xyz` from `molecules/` folder.
- Restored `YPc2.xyz` to uncompressed Z; use `molecule_z_scale` to control Z squeezing.
- `add_molecule_lattice` now supports arbitrary grid sizes via `grid_range` and consistent offsets for any `grid_n`.

### Masks
- Masks now expand by the effective tip blur radius to align with blurred images.
- Adatom-on-YPc2 molecules are labeled as class `6` (`adatom_on_ypc2`).
 - Adatom-on-YPc2 mask radius can expand using the new `adatom_on_top_width`.

### New Notebook
- `notebooks/FePc_Layer_Ag001.ipynb` creates a 20 nm Ag(100) scene with a 3x3 FePc layer (1.5 nm spacing), optional step, and shows 2D/3D + topography.
- `notebooks/YPc2_LDOS_Demo.ipynb` compares YPc2 ring-orbital LDOS vs atom-based LDOS.
- `notebooks/YPc2_Dataset_Generator.ipynb` runs `generate_ypc2_layer.py` with all options exposed as parameters.

### Dataset Scripts
- `scripts/generate_fepc_layer.py` generates datasets of FePc layer images with optional randomized lattice spacing and grid size, TIFF outputs, and metadata.
- `scripts/generate_ypc2_layer.py` generates YPc2 lattice datasets with LDOS and noise controls, adatom-on-top, and realism preset.
  - New CLI args for multi-tip, corner steps, multi-island lattices, YPc2 center plateau, and adatom-on-top width.
- Colored TIFF quicklook outputs (inferno / tab20) if `tifffile` is installed.
- Script auto-skips TIFF outputs if `tifffile` is not available.

### Repo Hygiene
- Added `.gitignore` entries for dataset outputs and Python cache files.

## Usage Notes
- Set molecule type and scaling in config:
  ```python
  cfg.features.molecule_name = "YPc2"
  cfg.features.molecule_z_scale = 1/3
  cfg.features.molecule_xy_scale = 1.0
  ```
- Use `cfg.ldos.surface_mode = "atomic"` to restore atomic corrugation on the substrate.
- Increase blur in  via `cfg.noise.tip_sigma`.
- Fixed grid to surface size: set `cfg.surface.size_angstrom`.

## Known Files
- Core code: `stm_sim/`
- Notebooks: `notebooks/STM_Simulator_Demo.ipynb`, `notebooks/FePc_Layer_Ag001.ipynb`, `notebooks/YPc2_LDOS_Demo.ipynb`, `notebooks/YPc2_Dataset_Generator.ipynb`
- Molecules: `molecules/FePc.mol`, `molecules/YPc2.xyz`
- Scripts: `scripts/generate_mvp.py`, `scripts/generate_fepc_layer.py`, `scripts/generate_ypc2_layer.py`

