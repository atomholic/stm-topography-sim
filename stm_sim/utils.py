import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def ensure_rng(seed=None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def sample_uniform(rng: np.random.Generator, spec):
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return rng.uniform(spec[0], spec[1])
    return spec


def sample_int(rng: np.random.Generator, spec):
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        low, high = int(spec[0]), int(spec[1])
        if high < low:
            low, high = high, low
        return int(rng.integers(low, high + 1))
    return int(spec)


def grid_from_bbox(bbox, pixels: Tuple[int, int]):
    (xmin, xmax), (ymin, ymax) = bbox
    nx = int(pixels[1])
    ny = int(pixels[0])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    return xg, yg


def rotation_matrix_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    if radius is None:
        radius = int(max(1, math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def convolve1d_same(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if kernel.size == 1:
        return arr.copy()
    pad = kernel.size // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis, padded)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.copy()
    kernel = gaussian_kernel1d(sigma)
    tmp = convolve1d_same(image, kernel, axis=0)
    out = convolve1d_same(tmp, kernel, axis=1)
    return out
