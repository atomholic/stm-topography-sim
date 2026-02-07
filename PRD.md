Yesss, this is shaping up into a super solid MVP spec. I’ll lock in your answers and refine the PRD so it’s implementation-ready for you to move into code.

---

# 🧾 Product Requirements Document (PRD)

**Project:** Synthetic Atomic-Scale STM Image Simulator
**Primary Use:** Generate labeled STM topography images to train a semantic segmentation network
**Initial Scope:** 1,000-image MVP dataset

---

## 1. 🎯 Purpose & Scope

Build a modular simulation platform that generates **physically inspired but computationally lightweight** STM topography images of metal surfaces with adsorbed structures.

The simulator will:

* Model **ideal metal surfaces** (starting with Ag)
* Place **adatoms, defects, steps, and FePc molecules**
* Use **s-wave orbital approximation** + **Tersoff–Hamann** tunneling model
* Produce:

  * Constant-current STM topography images
  * Pixel-level **semantic segmentation masks**

### Fidelity Level (Clarifying Q1)

We will target **qualitative physical realism**, not quantitative reproduction of experimental current values.
That means:

* Correct *relative contrast trends* (atoms brighter than hollow sites, molecules distinct from substrate)
* Physically motivated decay and convolution
* But no DFT-level LDOS or material-specific band structures (for now)

---

## 2. 🧠 Simulation Pipeline Overview

```
Surface + Adsorbate Geometry
        ↓
Approximate LDOS via s-wave orbitals
        ↓
Constant-current STM height solving (Tersoff–Hamann)
        ↓
Imaging artifacts + noise + tip instability
        ↓
Semantic label map generation
        ↓
Dataset export (image + mask + metadata)
```

---

## 3. 🧱 Functional Requirements

### 3.1 Atomic Structure & Scene Generator

#### 3.1.1 Surface Builder

* Lattice: **FCC**
* Initial material: **Ag**
* Surfaces supported:

  * Ag(111) (primary)
  * Ag(100) (optional if easy)
* User parameters:

  * Lateral size (nm or unit cells)
  * Number of layers (z-depth)
  * Lattice constant

Surface atoms remain **ideal lattice positions**, except where modified by features below.

---

#### 3.1.2 Surface Features

| Feature           | Description                | Implementation Note         |
| ----------------- | -------------------------- | --------------------------- |
| Adatoms           | Single Ag atoms on surface | Random adsorption sites     |
| Vacancies         | Missing surface atoms      | Remove atoms from lattice   |
| Step edges        | Height change of 1+ layers | Modify z of half-plane      |
| Surface roughness | Small z jitter             | Gaussian displacement noise |

**Step edges only affect STM via geometry (height), not special electronic states.**

---

#### 3.1.3 Molecules — FePc

We include a built-in **FePc molecular geometry loader**:

* Molecule treated as:

  * **Rigid**
  * Fixed internal coordinates (from file)
* Degrees of freedom per placement:

  * Random lateral (x, y) position
  * Random in-plane rotation
  * Fixed adsorption height (with small variation allowed)

Atoms in FePc are labeled as **“molecule atoms”** (not element-specific for segmentation).

Multiple molecules allowed per image.

---

### 3.2 Electronic Structure Approximation

We approximate LDOS near the Fermi level using **s-wave orbitals only**.

For atom *i*:

[
\psi_i(\mathbf{r}) = A_i \exp(-\kappa_i |\mathbf{r} - \mathbf{R}_i|)
]

LDOS approximation:

[
\rho(\mathbf{r}) \propto \sum_i |\psi_i(\mathbf{r})|^2
]

#### Parameter Groups (Grouped Labels Requirement)

We do **not distinguish elements** in labels, but LDOS amplitudes may differ internally:

| Atom Type     | Role             | Example Parameters        |
| ------------- | ---------------- | ------------------------- |
| Surface atom  | Bulk Ag          | (A_s, \kappa_s)           |
| Adatom        | Protruding metal | Slightly higher (A)       |
| Molecule atom | FePc atoms       | Different (A_m, \kappa_m) |

---

### 3.3 STM Simulation — Constant Current Mode Only

We simulate **constant-current topography** exclusively.

For each lateral pixel (x, y):

Solve for height (z) such that:

[
\rho(x, y, z) = \rho_{setpoint}
]

Where:

* (\rho_{setpoint}) = user-defined current setpoint proxy
* Tip modeled as s-wave (implicit in Tersoff–Hamann)

Bias-dependent contrast differences are **not included** in MVP.

---

### 3.4 Image Formation & Realism Layer

This stage makes outputs look like real STM rather than perfect math.

#### 3.4.1 Tip Convolution

* Lateral Gaussian blur
* Controls effective tip radius

---

#### 3.4.2 Noise Models

| Noise Type     | Description                |
| -------------- | -------------------------- |
| Gaussian noise | Electronics noise          |
| Line noise     | Scan-line correlated noise |
| Drift          | Shear/stretch distortion   |
| Height jitter  | Random z-offset per line   |

Noise levels randomized per image within ranges.

---

#### 3.4.3 Tip Instability (User Requirement #12)

We introduce parameter:

**`tip_instability ∈ [0, 1]`**

| Value | Effect                                            |
| ----- | ------------------------------------------------- |
| 0     | Perfectly stable tip                              |
| 0.5   | Occasional line-wise height jumps                 |
| 1     | Frequent discontinuities, sudden contrast changes |

Implementation ideas:

* Random line offsets in z
* Sudden convolution width change mid-image
* Small lateral shifts between scan lines

This parameter directly modulates probability and magnitude of these artifacts.

---

### 3.5 Semantic Segmentation Labels

**Semantic segmentation only** (no instance masks)

#### Classes (Grouped)

| Class ID | Description                      |
| -------- | -------------------------------- |
| 0        | Background / vacuum (if visible) |
| 1        | Clean metal surface              |
| 2        | Adatoms / clusters               |
| 3        | Molecule (FePc)                  |
| 4        | Vacancies                        |
| 5        | Step edges                       |

Labels generated from **ground-truth geometry projected into image space**, not from noisy image.

---

## 4. 🎛 Dataset Engine (MVP Target: 1,000 Images)

Each image randomized over:

| Parameter                 | Randomized? |
| ------------------------- | ----------- |
| Surface size              | ✓           |
| Step presence             | ✓           |
| Defect density            | ✓           |
| Adatom count              | ✓           |
| Molecule count (FePc)     | ✓           |
| Molecule rotation         | ✓           |
| Setpoint (LDOS threshold) | ✓           |
| Tip radius                | ✓           |
| Noise levels              | ✓           |
| Tip instability           | ✓           |

---

## 5. 💾 Outputs

For each sample:

1. **Topography image**

   * Format: `.npy` or `.tiff`
   * Float32 height map

2. **Segmentation mask**

   * Same resolution
   * Integer class IDs

3. **Metadata JSON**

   * Atomic coordinates
   * Molecule placements
   * All simulation parameters
   * Random seed

---

## 6. ⚙️ Technical Requirements

| Category       | Requirement                                      |
| -------------- | ------------------------------------------------ |
| Language       | Python                                           |
| Acceleration   | **GPU allowed** (for LDOS grid + height solving) |
| Determinism    | Seedable RNG                                     |
| Modularity     | Physics, noise, and geometry decoupled           |
| ML Integration | PyTorch Dataset wrapper                          |

---

## 7. 📦 MVP Deliverables

* Ag surface generator
* FePc molecule loader + placer
* s-wave LDOS calculator
* Constant-current solver
* Noise + tip instability module
* Semantic mask generator
* Script to generate **1,000 images**

---

## 8. 🔜 Post-MVP Extensions (Not Now)

* Orbital symmetry beyond s-wave
* Bias-dependent contrast
* dI/dV maps
* Element-specific segmentation
* Learned noise model from real STM
