import numpy as np

from .utils import grid_from_bbox


def _mark_circle(mask, x, y, positions, radius, class_id):
    if positions is None or positions.size == 0:
        return
    dx = x[..., None] - positions[:, 0]
    dy = y[..., None] - positions[:, 1]
    dist2 = dx * dx + dy * dy
    min_dist2 = dist2.min(axis=-1)
    mask[min_dist2 <= radius * radius] = class_id


def generate_mask(scene, pixels, label_cfg, bbox=None, blur_radius: float = 0.0):
    x, y = grid_from_bbox(bbox if bbox is not None else scene.bbox, pixels)
    mask = np.full(x.shape, 1, dtype=np.int64)

    # background (outside bbox) is not applicable since grid is within bbox

    # step edges
    corner_points = []
    for edge in scene.step_edges:
        width = label_cfg.step_edge_width + 2.0 * blur_radius
        normal = np.array(edge.get("normal", [1.0, 0.0]), dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue
        normal = normal / normal_norm
        point = np.array(edge.get("point", [0.0, 0.0]), dtype=float)
        dist = np.abs((x - point[0]) * normal[0] + (y - point[1]) * normal[1])
        mask[dist <= width * 0.5] = 5
        corner_point = edge.get("corner_point")
        corner_radius = edge.get("corner_radius", 0.0)
        if corner_point is not None and corner_radius and corner_radius > 0:
            corner_points.append((corner_point, float(corner_radius)))

    if corner_points:
        for corner_point, corner_radius in corner_points:
            cp = np.array(corner_point, dtype=float)
            dx = x - cp[0]
            dy = y - cp[1]
            dist2 = dx * dx + dy * dy
            radius = corner_radius + blur_radius
            mask[dist2 <= radius * radius] = 5

    # vacancies
    if scene.vacancies.size > 0:
        _mark_circle(mask, x, y, scene.vacancies, label_cfg.vacancy_radius + blur_radius, 4)

    # adatoms
    adatom_positions = scene.positions[scene.types == "adatom"]
    _mark_circle(mask, x, y, adatom_positions, label_cfg.adatom_radius + blur_radius, 2)

    # molecules
    molecule_positions = scene.positions[scene.types == "molecule"]
    _mark_circle(mask, x, y, molecule_positions, label_cfg.molecule_radius + blur_radius, 3)

    # adatom-on-YPc2: label the whole molecule footprint as class 6
    adatom_on_ypc2 = np.array(scene.metadata.get("adatom_on_top_centers", []), dtype=float)
    adatom_on_ypc2_radius = max(label_cfg.molecule_radius, label_cfg.adatom_on_ypc2_radius)
    width = float(scene.metadata.get("adatom_on_top_width", 0.0) or 0.0)
    if width > 0:
        adatom_on_ypc2_radius = max(adatom_on_ypc2_radius, 0.5 * width)
    _mark_circle(
        mask,
        x,
        y,
        adatom_on_ypc2,
        adatom_on_ypc2_radius + blur_radius,
        6,
    )

    return mask
