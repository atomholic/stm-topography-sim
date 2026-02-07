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
  - Dataset generator script.

### Units & Grid
- Standardized lengths to **Angstroms (Å)** in config and geometry.
- Fixed imaging grid to the **surface bounding box** so molecules do not expand the image.
- Blur sigma and line-noise correlation are now **Å-based** and converted to pixels using grid spacing.

### Steps & Terraces
- Single-layer surface by default (`layers = 1`).
- Step edges shift surface atoms upward instead of stacking layers.
- Adatoms/molecules placed on the local terrace height.
- Step edge labels support arbitrary angle steps.

### Visualization
- Added 2D scene plot with color-coded features and legend.
- Added 3D scene plot with optional fixed `zlim`.

### Molecule Support
- Added `molecule_name` and `molecule_z_scale` in config.
- Molecule loader supports `.mol` and `.xyz` from `molecules/` folder.
- Restored `YPc2.xyz` to uncompressed Z; use `molecule_z_scale` to control Z squeezing.

### Notebooks
- `STM_Simulator_Demo.ipynb` updated for new APIs and optional molecule selection.

### Repo Setup
- New repo folder `stm-topography-sim` created and pushed to GitHub.
- Added `.gitignore` and removed `__pycache__` from version control.

## Usage Notes
- Set molecule type and Z scaling in config:
  ```python
  cfg.features.molecule_name = "YPc2"
  cfg.features.molecule_z_scale = 1/3
  ```
- Increase blur in Å via `cfg.noise.tip_sigma`.
- Fixed grid to surface size: set `cfg.surface.size_angstrom`.

## Known Files
- Core code: `stm_sim/`
- Notebooks: `notebooks/STM_Simulator_Demo.ipynb`
- Molecules: `molecules/FePc.mol`, `molecules/YPc2.xyz`
- Script: `scripts/generate_mvp.py`

