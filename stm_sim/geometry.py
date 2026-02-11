import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .utils import ensure_rng, rotation_matrix_z



@dataclass
class Scene:
    positions: np.ndarray
    types: np.ndarray
    surface: str
    lattice_constant: float
    layer_spacing: float
    size_angstrom: Tuple[float, float]
    bbox: Tuple[Tuple[float, float], Tuple[float, float]]
    vacancies: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    step_edges: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _update_bbox(scene: Scene):
    xmin = float(scene.positions[:, 0].min())
    xmax = float(scene.positions[:, 0].max())
    ymin = float(scene.positions[:, 1].min())
    ymax = float(scene.positions[:, 1].max())
    scene.bbox = ((xmin, xmax), (ymin, ymax))


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0:
        return v
    return v / norm


def _min_distance_to_steps(scene: Scene, xy: np.ndarray) -> np.ndarray:
    if not scene.step_edges:
        return np.full((xy.shape[0],), np.inf, dtype=float)
    distances = []
    for edge in scene.step_edges:
        normal = np.array(edge.get("normal", [1.0, 0.0]), dtype=float)
        normal = _normalize(normal)
        point = np.array(edge.get("point", [0.0, 0.0]), dtype=float)
        d = np.abs((xy - point) @ normal)
        distances.append(d)
    return np.min(np.stack(distances, axis=1), axis=1)


def surface_height_at(scene: Scene, xy: np.ndarray) -> np.ndarray:
    surface_mask = scene.types == "surface"
    if not np.any(surface_mask):
        return np.zeros((xy.shape[0],), dtype=float)

    surface_z = scene.positions[surface_mask, 2]
    high_z = float(surface_z.max())
    if not scene.step_edges:
        return np.full((xy.shape[0],), high_z, dtype=float)

    z = np.full((xy.shape[0],), high_z, dtype=float)
    for edge in scene.step_edges:
        height = float(edge.get("height", 0.0))
        normal = np.array(edge.get("normal", [1.0, 0.0]), dtype=float)
        normal = _normalize(normal)
        point = np.array(edge.get("point", [0.0, 0.0]), dtype=float)
        side = (xy - point) @ normal
        corner_point = edge.get("corner_point")
        corner_radius = float(edge.get("corner_radius", 0.0) or 0.0)
        if corner_point is not None and corner_radius > 0:
            cp = np.array(corner_point, dtype=float)
            dx = xy[:, 0] - cp[0]
            dy = xy[:, 1] - cp[1]
            dist = np.sqrt(dx * dx + dy * dy)
            mask_round = dist <= corner_radius
            t_round = 0.5 * (1.0 - np.clip(side / corner_radius, -1.0, 1.0))
            t_sharp = (side <= 0).astype(float)
            t = np.where(mask_round, t_round, t_sharp)
            z = z - height * t
        else:
            z = np.where(side <= 0, z - height, z)
    return z


def _line_bbox_intersections(normal: np.ndarray, point: np.ndarray, bbox):
    normal = _normalize(normal)
    nx, ny = float(normal[0]), float(normal[1])
    x0, y0 = float(point[0]), float(point[1])
    (xmin, xmax), (ymin, ymax) = bbox
    points = []

    if abs(ny) > 1e-12:
        y = y0 - (nx / ny) * (xmin - x0)
        if ymin <= y <= ymax:
            points.append((xmin, y))
        y = y0 - (nx / ny) * (xmax - x0)
        if ymin <= y <= ymax:
            points.append((xmax, y))
    if abs(nx) > 1e-12:
        x = x0 - (ny / nx) * (ymin - y0)
        if xmin <= x <= xmax:
            points.append((x, ymin))
        x = x0 - (ny / nx) * (ymax - y0)
        if xmin <= x <= xmax:
            points.append((x, ymax))

    # pick two farthest points if more than 2
    if len(points) > 2:
        max_dist = -1.0
        p0 = p1 = points[0]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = dx * dx + dy * dy
                if dist > max_dist:
                    max_dist = dist
                    p0, p1 = points[i], points[j]
        points = [p0, p1]
    return points


def build_fcc_surface(
    surface: str = "111",
    size_angstrom: Tuple[float, float] = (50.0, 50.0),
    layers: int = 1,
    lattice_constant: float = 4.09,
) -> Scene:
    surface = surface.strip().lower().replace("(", "").replace(")", "")
    if surface not in ("111", "100"):
        raise ValueError("surface must be '111' or '100'")

    size_x = size_angstrom[0]
    size_y = size_angstrom[1]
    surface_bbox = ((-size_x / 2.0, size_x / 2.0), (-size_y / 2.0, size_y / 2.0))

    positions = []
    if surface == "111":
        a_surf = lattice_constant / math.sqrt(2)
        a1 = np.array([a_surf, 0.0])
        a2 = np.array([0.5 * a_surf, 0.5 * math.sqrt(3) * a_surf])
        d_layer = lattice_constant / math.sqrt(3)
        offsets = [(0.0, 0.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0)]

        nx = int(math.ceil(size_x / a_surf)) + 2
        ny = int(math.ceil(size_y / (0.5 * math.sqrt(3) * a_surf))) + 2

        for layer in range(layers):
            off = offsets[layer % 3]
            shift = off[0] * a1 + off[1] * a2
            z = -layer * d_layer
            for ix in range(nx):
                for iy in range(ny):
                    xy = ix * a1 + iy * a2 + shift
                    if 0 <= xy[0] <= size_x and 0 <= xy[1] <= size_y:
                        positions.append([xy[0], xy[1], z])
    else:
        a_surf = lattice_constant / math.sqrt(2)
        a1 = np.array([a_surf, 0.0])
        a2 = np.array([0.0, a_surf])
        d_layer = lattice_constant / 2.0
        offsets = [(0.0, 0.0)]
        nx = int(math.ceil(size_x / a_surf)) + 2
        ny = int(math.ceil(size_y / a_surf)) + 2
        for layer in range(layers):
            z = -layer * d_layer
            for ix in range(nx):
                for iy in range(ny):
                    x = ix * a_surf
                    y = iy * a_surf
                    if 0 <= x <= size_x and 0 <= y <= size_y:
                        positions.append([x, y, z])

    positions = np.array(positions, dtype=float)
    # center surface around origin for easier sampling
    positions[:, 0] -= size_x / 2.0
    positions[:, 1] -= size_y / 2.0

    types = np.full((positions.shape[0],), "surface", dtype="<U16")
    scene = Scene(
        positions=positions,
        types=types,
        surface=surface,
        lattice_constant=lattice_constant,
        layer_spacing=d_layer,
        size_angstrom=size_angstrom,
        bbox=((0.0, 0.0), (0.0, 0.0)),
        metadata={
            "a_surf": a_surf,
            "a1": a1.tolist(),
            "a2": a2.tolist(),
            "layer_offsets": offsets,
            "top_offset_index": 0,
            "surface_bbox": surface_bbox,
        },
    )
    _update_bbox(scene)
    return scene


def add_adatoms(scene: Scene, count: int, rng=None, height: float | None = None):
    if count <= 0:
        return
    rng = ensure_rng(rng)
    if height is None:
        height = scene.layer_spacing
    positions = []
    min_step_dist = scene.metadata.get("step_exclusion_distance", 0.0)
    attempts = 0
    max_attempts = max(20, count * 40)
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(scene.bbox[0][0], scene.bbox[0][1])
        y = rng.uniform(scene.bbox[1][0], scene.bbox[1][1])
        if min_step_dist > 0 and scene.step_edges:
            d = _min_distance_to_steps(scene, np.array([[x, y]], dtype=float))[0]
            if d < min_step_dist:
                continue
        surface_z = surface_height_at(scene, np.array([[x, y]], dtype=float))[0]
        z = surface_z + height
        positions.append([x, y, z])
    positions = np.array(positions, dtype=float)
    types = np.full((positions.shape[0],), "adatom", dtype=scene.types.dtype)
    scene.positions = np.vstack([scene.positions, positions])
    scene.types = np.concatenate([scene.types, types])
    _update_bbox(scene)


def add_vacancies(scene: Scene, count: int, rng=None):
    if count <= 0:
        return
    rng = ensure_rng(rng)
    surface_idx = np.where(scene.types == "surface")[0]
    if surface_idx.size == 0:
        return
    top_z = float(scene.positions[surface_idx, 2].max())
    candidate_levels = [top_z]
    for edge in scene.step_edges:
        height = float(edge.get("height", 0.0))
        candidate_levels.append(top_z - height)

    z_vals = scene.positions[surface_idx, 2]
    level_mask = np.zeros_like(z_vals, dtype=bool)
    for level in candidate_levels:
        level_mask |= np.isclose(z_vals, level, atol=1e-3)
    top_idx = surface_idx[level_mask]
    if top_idx.size == 0:
        return
    min_step_dist = scene.metadata.get("step_exclusion_distance", 0.0)
    if min_step_dist > 0 and scene.step_edges:
        top_xy = scene.positions[top_idx, :2]
        d = _min_distance_to_steps(scene, top_xy)
        top_idx = top_idx[d >= min_step_dist]
        if top_idx.size == 0:
            return
    count = min(count, top_idx.size)
    remove_idx = rng.choice(top_idx, size=count, replace=False)
    scene.vacancies = np.vstack([scene.vacancies, scene.positions[remove_idx]])
    keep_mask = np.ones(scene.positions.shape[0], dtype=bool)
    keep_mask[remove_idx] = False
    scene.positions = scene.positions[keep_mask]
    scene.types = scene.types[keep_mask]
    _update_bbox(scene)


def add_step(
    scene: Scene,
    rng=None,
    axis: str | None = None,
    position: float | None = None,
    height_layers: int = 1,
    angle_deg: float | None = None,
):
    rng = ensure_rng(rng)
    if angle_deg is not None:
        theta = math.radians(float(angle_deg))
        normal = _normalize(np.array([math.cos(theta), math.sin(theta)], dtype=float))
        x0 = rng.uniform(scene.bbox[0][0], scene.bbox[0][1])
        y0 = rng.uniform(scene.bbox[1][0], scene.bbox[1][1])
        point = np.array([x0, y0], dtype=float)
        axis = "angle"
    else:
        if axis is None:
            axis = rng.choice(["x", "y"])
        if position is None:
            if axis == "x":
                position = rng.uniform(scene.bbox[0][0], scene.bbox[0][1])
            else:
                position = rng.uniform(scene.bbox[1][0], scene.bbox[1][1])
        if axis == "x":
            normal = np.array([1.0, 0.0], dtype=float)
            point = np.array([float(position), 0.0], dtype=float)
        else:
            normal = np.array([0.0, 1.0], dtype=float)
            point = np.array([0.0, float(position)], dtype=float)
    if height_layers <= 0:
        return

    surface_mask = scene.types == "surface"
    if not np.any(surface_mask):
        return

    tol = 1e-3
    top_z = float(scene.positions[surface_mask, 2].max())
    top_mask = surface_mask & np.isclose(scene.positions[:, 2], top_z, atol=tol)

    xy = scene.positions[:, :2]
    side_mask = (xy - point) @ normal > 0

    # remove any lower-layer surface atoms on the raised side
    lower_mask = surface_mask & side_mask & (scene.positions[:, 2] < top_z - tol)
    if np.any(lower_mask):
        keep_mask = ~lower_mask
        scene.positions = scene.positions[keep_mask]
        scene.types = scene.types[keep_mask]
        surface_mask = scene.types == "surface"
        top_z = float(scene.positions[surface_mask, 2].max())
        top_mask = surface_mask & np.isclose(scene.positions[:, 2], top_z, atol=tol)
        xy = scene.positions[:, :2]
        side_mask = (xy - point) @ normal > 0

    # shift the top layer upward on the raised side (single-layer terrace model)
    height = height_layers * scene.layer_spacing
    scene.positions[top_mask & side_mask, 2] += height
    scene.step_edges.append(
        {
            "axis": axis,
            "position": None if angle_deg is not None else float(position),
            "height": float(height),
            "normal": normal.tolist(),
            "point": point.tolist(),
            "angle_deg": None if angle_deg is None else float(angle_deg),
        }
    )
    _update_bbox(scene)


def add_corner_step(
    scene: Scene,
    rng=None,
    height_layers: int = 1,
    base_angle_deg: float | None = None,
    corner_angle_deg: float | None = None,
    corner_point: Tuple[float, float] | None = None,
    round_radius: float = 0.0,
):
    rng = ensure_rng(rng)
    if height_layers <= 0:
        return

    surface_mask = scene.types == "surface"
    if not np.any(surface_mask):
        return

    if base_angle_deg is None:
        base_angle_deg = float(rng.uniform(0.0, 180.0))
    if corner_angle_deg is None:
        corner_angle_deg = float(rng.uniform(60.0, 120.0))

    theta1 = math.radians(float(base_angle_deg))
    theta2 = math.radians(float(base_angle_deg) + float(corner_angle_deg))
    normal1 = _normalize(np.array([math.cos(theta1), math.sin(theta1)], dtype=float))
    normal2 = _normalize(np.array([math.cos(theta2), math.sin(theta2)], dtype=float))

    if corner_point is None:
        x0 = rng.uniform(scene.bbox[0][0], scene.bbox[0][1])
        y0 = rng.uniform(scene.bbox[1][0], scene.bbox[1][1])
        point = np.array([x0, y0], dtype=float)
    else:
        point = np.array([float(corner_point[0]), float(corner_point[1])], dtype=float)

    tol = 1e-3
    top_z = float(scene.positions[surface_mask, 2].max())
    top_mask = surface_mask & np.isclose(scene.positions[:, 2], top_z, atol=tol)

    xy = scene.positions[:, :2]
    side1 = (xy - point) @ normal1 > 0
    side2 = (xy - point) @ normal2 > 0
    raised_mask = side1 & side2

    # remove any lower-layer surface atoms on the raised corner
    lower_mask = surface_mask & raised_mask & (scene.positions[:, 2] < top_z - tol)
    if np.any(lower_mask):
        keep_mask = ~lower_mask
        scene.positions = scene.positions[keep_mask]
        scene.types = scene.types[keep_mask]
        surface_mask = scene.types == "surface"
        top_z = float(scene.positions[surface_mask, 2].max())
        top_mask = surface_mask & np.isclose(scene.positions[:, 2], top_z, atol=tol)
        xy = scene.positions[:, :2]
        side1 = (xy - point) @ normal1 > 0
        side2 = (xy - point) @ normal2 > 0
        raised_mask = side1 & side2

    height = height_layers * scene.layer_spacing
    scene.positions[top_mask & raised_mask, 2] += height

    edge_common = {
        "height": float(height),
        "point": point.tolist(),
        "corner_point": point.tolist(),
        "corner_radius": float(round_radius),
    }
    scene.step_edges.append(
        {
            "axis": "corner",
            "position": None,
            "normal": normal1.tolist(),
            "angle_deg": float(base_angle_deg),
            **edge_common,
        }
    )
    scene.step_edges.append(
        {
            "axis": "corner",
            "position": None,
            "normal": normal2.tolist(),
            "angle_deg": float(base_angle_deg + float(corner_angle_deg)),
            **edge_common,
        }
    )
    _update_bbox(scene)


def add_surface_roughness(scene: Scene, sigma: float, rng=None):
    if sigma <= 0:
        return
    rng = ensure_rng(rng)
    mask = scene.types == "surface"
    scene.positions[mask, 2] += rng.normal(0.0, sigma, size=mask.sum())
    _update_bbox(scene)


def load_xyz(path: Path) -> Tuple[np.ndarray, List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    try:
        count = int(lines[0])
        start = 2
    except ValueError:
        count = len(lines)
        start = 0
    coords = []
    elements = []
    for line in lines[start : start + count]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=float), elements


def load_mol(path: Path) -> Tuple[np.ndarray, List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if len(lines) < 4:
        raise ValueError(f"Invalid mol file: {path}")

    counts_line = lines[3].strip()
    parts = counts_line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid counts line in mol file: {path}")
    try:
        n_atoms = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in mol file: {path}") from exc

    coords = []
    elements = []
    start = 4
    end = start + n_atoms
    if end > len(lines):
        raise ValueError(f"Mol file truncated: {path}")

    for line in lines[start:end]:
        parts = line.split()
        if len(parts) < 4:
            continue
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        elements.append(parts[3])

    return np.array(coords, dtype=float), elements


def load_molecule(name: str) -> np.ndarray:
    repo_root = Path(__file__).resolve().parent.parent
    mol_path = repo_root / "molecules" / f"{name}.mol"
    if mol_path.exists():
        coords, _ = load_mol(mol_path)
        return coords

    xyz_path = repo_root / "molecules" / f"{name}.xyz"
    if xyz_path.exists():
        coords, _ = load_xyz(xyz_path)
        return coords

    if name.lower() == "fepc":
        asset_path = Path(__file__).parent / "assets" / "fepc.xyz"
        coords, _ = load_xyz(asset_path)
        return coords

    raise FileNotFoundError(f"Could not find molecule file for '{name}' in molecules/")


def load_fepc() -> np.ndarray:
    return load_molecule("FePc")


def add_molecules(
    scene: Scene,
    count: int,
    rng=None,
    height: float = 2.0,
    height_sigma: float = 0.1,
    molecule_name: str = "FePc",
    molecule_z_scale: float = 1.0,
    molecule_xy_scale: float = 1.0,
):
    if count <= 0:
        return
    rng = ensure_rng(rng)
    coords = load_molecule(molecule_name)
    coords = coords - coords.mean(axis=0, keepdims=True)
    if molecule_z_scale != 1.0 or molecule_xy_scale != 1.0:
        coords = coords.copy()
    if molecule_xy_scale != 1.0:
        coords[:, 0] *= float(molecule_xy_scale)
        coords[:, 1] *= float(molecule_xy_scale)
    if molecule_z_scale != 1.0:
        coords[:, 2] *= float(molecule_z_scale)
    min_step_dist = scene.metadata.get("step_exclusion_distance", 0.0)
    attempts = 0
    max_attempts = max(20, count * 40)
    placed_count = 0
    centers = []
    orientations = []
    while placed_count < count and attempts < max_attempts:
        attempts += 1
        angle = rng.uniform(0.0, 2 * math.pi)
        rot = rotation_matrix_z(angle)
        rotated = coords @ rot.T
        x = rng.uniform(scene.bbox[0][0], scene.bbox[0][1])
        y = rng.uniform(scene.bbox[1][0], scene.bbox[1][1])
        if min_step_dist > 0 and scene.step_edges:
            d = _min_distance_to_steps(scene, np.array([[x, y]], dtype=float))[0]
            if d < min_step_dist:
                continue
        surface_z = surface_height_at(scene, np.array([[x, y]], dtype=float))[0]
        z = surface_z + height + rng.normal(0.0, height_sigma)
        placed = rotated + np.array([x, y, z])
        types = np.full((placed.shape[0],), "molecule", dtype=scene.types.dtype)
        scene.positions = np.vstack([scene.positions, placed])
        scene.types = np.concatenate([scene.types, types])
        centers.append([x, y, z])
        orientations.append(math.degrees(angle))
        placed_count += 1
    if centers:
        scene.metadata.setdefault("molecule_centers", []).extend(centers)
        scene.metadata.setdefault("molecule_orientations_deg", []).extend(orientations)
    _update_bbox(scene)


def add_molecule_lattice(
    scene: Scene,
    molecule_name: str = "FePc",
    grid_n: int = 3,
    spacing: float = 15.0,
    orientation_deg: float = 0.0,
    lattice_angle_deg: float = 0.0,
    height: float = 2.0,
    rng=None,
    random_center: bool = True,
    spacing_range: Tuple[float, float] | None = None,
    grid_range: Tuple[int, int] | None = None,
    molecule_z_scale: float = 1.0,
    molecule_xy_scale: float = 1.0,
    adatom_on_top_count: Tuple[int, int] | int | None = None,
    adatom_on_top_height: float = 2.0,
    adatom_on_top_radial_offset: float = 0.0,
    adatom_on_top_radial_jitter: float = 0.0,
    adatom_on_top_width: float = 0.0,
    island_count: Tuple[int, int] | int | None = None,
    island_min_distance: float = 1.0,
    island_orientation_range: Tuple[float, float] | None = None,
    island_lattice_angle_range: Tuple[float, float] | None = None,
    edge_remove_count: Tuple[int, int] | int | None = None,
    interior_remove_count: Tuple[int, int] | int | None = None,
    edge_remove_edge_prob: float = 0.9,
    orientation_relative_to_lattice: bool = False,
):
    rng = ensure_rng(rng)
    coords = load_molecule(molecule_name)
    coords = coords - coords.mean(axis=0, keepdims=True)
    if molecule_z_scale != 1.0 or molecule_xy_scale != 1.0:
        coords = coords.copy()
    if molecule_xy_scale != 1.0:
        coords[:, 0] *= float(molecule_xy_scale)
        coords[:, 1] *= float(molecule_xy_scale)
    if molecule_z_scale != 1.0:
        coords[:, 2] *= float(molecule_z_scale)

    bbox = scene.metadata.get("surface_bbox", scene.bbox)
    (xmin, xmax), (ymin, ymax) = bbox

    min_step_dist = scene.metadata.get("step_exclusion_distance", 0.0)
    existing_island_bboxes = []

    def sample_grid_and_spacing():
        local_grid_n = grid_n
        local_spacing = spacing
        if grid_range is not None:
            gmin, gmax = int(grid_range[0]), int(grid_range[1])
            local_grid_n = int(rng.integers(gmin, gmax + 1))
        if spacing_range is not None:
            local_spacing = float(rng.uniform(spacing_range[0], spacing_range[1]))
        return local_grid_n, local_spacing

    def sample_orientation(local_lattice_angle: float):
        if island_orientation_range is not None:
            base = float(rng.uniform(island_orientation_range[0], island_orientation_range[1]))
        else:
            base = float(orientation_deg)
        if orientation_relative_to_lattice:
            return base + float(local_lattice_angle)
        return base

    def sample_lattice_angle():
        if island_lattice_angle_range is not None:
            return float(rng.uniform(island_lattice_angle_range[0], island_lattice_angle_range[1]))
        return float(lattice_angle_deg)

    def choose_center(max_extent):
        if random_center:
            if xmax - xmin < 2.0 * max_extent or ymax - ymin < 2.0 * max_extent:
                return None
            cx = rng.uniform(xmin + max_extent, xmax - max_extent)
            cy = rng.uniform(ymin + max_extent, ymax - max_extent)
        else:
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
        return cx, cy

    def island_bbox_for(center, max_extent):
        cx, cy = center
        return (cx - max_extent, cx + max_extent, cy - max_extent, cy + max_extent)

    def overlaps_existing(bbox_new):
        if not existing_island_bboxes:
            return False
        for (x0, x1, y0, y1) in existing_island_bboxes:
            if (
                bbox_new[0] <= x1 + island_min_distance
                and bbox_new[1] >= x0 - island_min_distance
                and bbox_new[2] <= y1 + island_min_distance
                and bbox_new[3] >= y0 - island_min_distance
            ):
                return True
        return False

    def build_centers(local_grid_n, local_spacing, local_lattice_angle):
        offsets = np.arange(local_grid_n, dtype=float) - (local_grid_n - 1) / 2.0
        max_extent = max(abs(o) for o in offsets) * local_spacing if local_grid_n > 0 else 0.0
        lat_rot = rotation_matrix_z(np.deg2rad(local_lattice_angle))

        for _ in range(100):
            center = choose_center(max_extent)
            if center is None:
                return None, None, None, None
            cx, cy = center
            centers = []
            grid_indices = []
            for i in range(local_grid_n):
                for j in range(local_grid_n):
                    v = np.array([offsets[i] * local_spacing, offsets[j] * local_spacing, 0.0]) @ lat_rot.T
                    centers.append((cx + v[0], cy + v[1]))
                    grid_indices.append((i, j))
            centers = np.array(centers, dtype=float)
            if (
                centers[:, 0].min() < xmin
                or centers[:, 0].max() > xmax
                or centers[:, 1].min() < ymin
                or centers[:, 1].max() > ymax
            ):
                continue
            if min_step_dist > 0 and scene.step_edges:
                d = _min_distance_to_steps(scene, centers)
                if np.any(d < min_step_dist):
                    continue
            bbox_new = island_bbox_for((cx, cy), max_extent)
            if overlaps_existing(bbox_new):
                continue
            return centers, grid_indices, (cx, cy), max_extent
        return None, None, None, None

    def sample_count(spec):
        if spec is None:
            return 0
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            low, high = int(spec[0]), int(spec[1])
            if high < low:
                low, high = high, low
            return int(rng.integers(low, high + 1))
        return int(spec)

    def apply_removals(centers, grid_indices, local_grid_n):
        if centers is None or len(centers) == 0:
            return centers, 0, 0
        edge_idx = [k for k, (i, j) in enumerate(grid_indices) if i == 0 or j == 0 or i == local_grid_n - 1 or j == local_grid_n - 1]
        interior_idx = [k for k in range(len(grid_indices)) if k not in edge_idx]
        remove_set = set()

        edge_remove = sample_count(edge_remove_count)
        interior_remove = sample_count(interior_remove_count)

        for _ in range(edge_remove):
            use_edge = rng.random() < edge_remove_edge_prob
            pool = edge_idx if use_edge else interior_idx
            pool = [idx for idx in pool if idx not in remove_set]
            if not pool:
                continue
            pick = int(rng.choice(pool))
            remove_set.add(pick)

        if interior_remove > 0:
            pool = [idx for idx in interior_idx if idx not in remove_set]
            if pool:
                count = min(interior_remove, len(pool))
                picks = rng.choice(pool, size=count, replace=False)
                for pick in picks.tolist():
                    remove_set.add(int(pick))

        if remove_set:
            keep = [i for i in range(len(centers)) if i not in remove_set]
            centers = centers[keep]
        return centers, edge_remove, interior_remove

    if island_count is None:
        island_total = 1
    else:
        island_total = sample_count(island_count)
    island_total = max(1, island_total)

    for _ in range(island_total):
        local_grid_n, local_spacing = sample_grid_and_spacing()
        local_lattice_angle = sample_lattice_angle()
        local_orientation = sample_orientation(local_lattice_angle)

        centers, grid_indices, center_xy, max_extent = build_centers(
            local_grid_n, local_spacing, local_lattice_angle
        )
        if centers is None:
            continue

        centers, edge_removed, interior_removed = apply_removals(centers, grid_indices, local_grid_n)
        if centers is None or len(centers) == 0:
            continue

        bbox_new = island_bbox_for(center_xy, max_extent)
        existing_island_bboxes.append(bbox_new)

        rot = rotation_matrix_z(np.deg2rad(local_orientation))
        placed = []
        centers_with_z = []
        for (x0, y0) in centers:
            mol = coords @ rot.T
            surface_z = surface_height_at(scene, np.array([[x0, y0]]))[0]
            z0 = surface_z + height
            mol = mol + np.array([x0, y0, z0])
            placed.append(mol)
            centers_with_z.append([float(x0), float(y0), float(z0)])

        if placed:
            placed = np.vstack(placed)
            scene.positions = np.vstack([scene.positions, placed])
            scene.types = np.concatenate(
                [scene.types, np.full((placed.shape[0],), "molecule", dtype=scene.types.dtype)]
            )
            scene.metadata.setdefault("molecule_centers", []).extend(centers_with_z)
            scene.metadata.setdefault("molecule_orientations_deg", []).extend(
                [float(local_orientation)] * len(centers_with_z)
            )
            scene.metadata.setdefault("lattice_islands", []).append(
                {
                    "center": [float(center_xy[0]), float(center_xy[1])],
                    "grid_n": int(local_grid_n),
                    "spacing": float(local_spacing),
                    "orientation_deg": float(local_orientation),
                    "lattice_angle_deg": float(local_lattice_angle),
                    "edge_removed": int(edge_removed),
                    "interior_removed": int(interior_removed),
                }
            )

            if adatom_on_top_count is not None:
                count = sample_count(adatom_on_top_count)
                count = max(0, min(count, len(centers_with_z)))
                if count > 0:
                    idx = rng.choice(len(centers_with_z), size=count, replace=False)
                    adatom_positions = []
                    width = float(adatom_on_top_width)
                    ring_radius = 0.5 * width if width > 0 else 0.0
                    ring_fracs = [0.33, 0.66, 1.0] if ring_radius > 0 else []
                    for i in idx:
                        cx, cy, cz = centers_with_z[int(i)]
                        if adatom_on_top_radial_offset > 0 or adatom_on_top_radial_jitter > 0:
                            angle = rng.uniform(0.0, 2 * math.pi)
                            offset = float(adatom_on_top_radial_offset)
                            if adatom_on_top_radial_jitter > 0:
                                offset += rng.normal(0.0, float(adatom_on_top_radial_jitter))
                            cx = cx + offset * math.cos(angle)
                            cy = cy + offset * math.sin(angle)
                        adatom_positions.append([cx, cy, cz + float(adatom_on_top_height)])
                        if ring_fracs:
                            for frac in ring_fracs:
                                rr = ring_radius * frac
                                if rr <= 0:
                                    continue
                                ring_points = int(max(8, min(48, (2 * math.pi * rr) / 2.0)))
                                for k in range(ring_points):
                                    ang = 2.0 * math.pi * k / ring_points
                                    adatom_positions.append(
                                        [cx + rr * math.cos(ang), cy + rr * math.sin(ang), cz + float(adatom_on_top_height)]
                                    )
                    adatom_positions = np.array(adatom_positions, dtype=float)
                    scene.positions = np.vstack([scene.positions, adatom_positions])
                    scene.types = np.concatenate(
                        [scene.types, np.full((adatom_positions.shape[0],), "adatom", dtype=scene.types.dtype)]
                    )
                    scene.metadata.setdefault("adatom_on_top_centers", []).extend(
                        [centers_with_z[int(i)] for i in idx]
                    )
                    if width > 0:
                        scene.metadata["adatom_on_top_width"] = float(width)

        _update_bbox(scene)


def plot_scene_top_view(scene: Scene, ax=None, show: bool = False):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    pos = scene.positions
    types = scene.types
    z = pos[:, 2]

    surface_z = pos[types == "surface", 2]
    if surface_z.size > 0:
        z_min = float(surface_z.min())
        z_max = float(surface_z.max())
    else:
        z_min = float(z.min())
        z_max = float(z.max())
    terrace_mid = 0.5 * (z_min + z_max)
    has_step = abs(z_max - z_min) > 1e-6
    color_map = {
        "surface": ("#6b7280", 3.0, 0.5, "Surface"),
        "adatom": ("#ff6b6b", 18.0, 1.0, "Adatom"),
        "molecule": ("#00b4d8", 8.0, 0.9, "Molecule"),
    }

    for label, (color, size, alpha, legend_label) in color_map.items():
        sel = types == label
        if np.any(sel):
            edge = "black" if label == "adatom" else "none"
            if has_step:
                size_scale = np.where(z[sel] >= terrace_mid, 4.0, 1.0)
            else:
                size_scale = 1.0
            ax.scatter(
                pos[sel, 0],
                pos[sel, 1],
                s=size * size_scale,
                c=color,
                alpha=alpha,
                linewidths=0.4 if edge != "none" else 0,
                edgecolors=edge,
                label=legend_label,
            )

    if scene.vacancies.size > 0:
        ax.scatter(
            scene.vacancies[:, 0],
            scene.vacancies[:, 1],
            s=12.0,
            c="#2ecc71",
            alpha=0.9,
            linewidths=0,
            label="Vacancy",
        )

    if scene.step_edges:
        xmin, xmax = scene.bbox[0]
        ymin, ymax = scene.bbox[1]
        for edge in scene.step_edges:
            normal = np.array(edge.get("normal", [1.0, 0.0]), dtype=float)
            point = np.array(edge.get("point", [0.0, 0.0]), dtype=float)
            pts = _line_bbox_intersections(normal, point, scene.bbox)
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                ax.plot([x1, x2], [y1, y2], color="#f4a261", linestyle="--", linewidth=1.5)
        ax.plot([], [], color="#f4a261", linestyle="--", linewidth=1.5, label="Step edge")

    ax.set_aspect("equal")
    ax.set_title("Atomic Positions (Top View)")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    if show:
        plt.show()
    return fig, ax


def plot_scene_3d(
    scene: Scene,
    views: List[Tuple[float, float]] | None = None,
    show: bool = False,
    zlim: Tuple[float, float] | None = None,
):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    if views is None:
        views = [(25.0, 45.0)]

    n_views = len(views)
    fig = plt.figure(figsize=(5 * n_views, 5))
    axes = []

    pos = scene.positions
    types = scene.types
    color_map = {
        "surface": ("#6b7280", 6.0, 0.45, "Surface"),
        "adatom": ("#ff6b6b", 25.0, 1.0, "Adatom"),
        "molecule": ("#00b4d8", 10.0, 0.9, "Molecule"),
    }

    for idx, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, n_views, idx, projection="3d")
        axes.append(ax)

        for label, (color, size, alpha, legend_label) in color_map.items():
            sel = types == label
            if np.any(sel):
                ax.scatter(
                    pos[sel, 0],
                    pos[sel, 1],
                    pos[sel, 2],
                    s=size,
                    c=color,
                    alpha=alpha,
                    linewidths=0,
                    label=legend_label,
                )

        if scene.vacancies.size > 0:
            ax.scatter(
                scene.vacancies[:, 0],
                scene.vacancies[:, 1],
                scene.vacancies[:, 2],
                s=18.0,
                c="#2ecc71",
                alpha=0.9,
                linewidths=0,
                label="Vacancy",
            )

        if scene.step_edges:
            surface_mask = scene.types == "surface"
            z_mid = float(scene.positions[surface_mask, 2].mean()) if np.any(surface_mask) else 0.0
            for edge in scene.step_edges:
                normal = np.array(edge.get("normal", [1.0, 0.0]), dtype=float)
                point = np.array(edge.get("point", [0.0, 0.0]), dtype=float)
                pts = _line_bbox_intersections(normal, point, scene.bbox)
                if len(pts) == 2:
                    (x1, y1), (x2, y2) = pts
                    ax.plot([x1, x2], [y1, y2], [z_mid, z_mid], color="#f4a261", linestyle="--", linewidth=1.5)
            ax.plot([], [], [], color="#f4a261", linestyle="--", linewidth=1.5, label="Step edge")

        ax.set_title(f"3D View (elev={elev:.0f}, azim={azim:.0f})")
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        ax.view_init(elev=elev, azim=azim)

    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    if show:
        plt.show()
    return fig, axes
