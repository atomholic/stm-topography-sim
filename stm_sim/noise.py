import numpy as np

from .utils import gaussian_blur, gaussian_kernel1d, sample_uniform


def add_gaussian_noise(image: np.ndarray, sigma: float, rng) -> np.ndarray:
    if sigma <= 0:
        return image
    return image + rng.normal(0.0, sigma, size=image.shape)


def add_line_noise(image: np.ndarray, sigma: float, corr: float, rng) -> np.ndarray:
    if sigma <= 0:
        return image
    ny = image.shape[0]
    noise = rng.normal(0.0, sigma, size=ny)
    kernel = gaussian_kernel1d(corr) if corr > 0 else np.array([1.0], dtype=float)
    smooth = np.convolve(noise, kernel, mode="same")
    return image + smooth[:, None]


def add_background_variation(image: np.ndarray, sigma: float, corr: float, rng) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = rng.normal(0.0, sigma, size=image.shape)
    if corr > 0:
        noise = gaussian_blur(noise, corr)
    return image + noise


def add_vibration_noise(
    image: np.ndarray,
    amp: float,
    wavelength: float,
    angle_deg: float,
    pixel_size: tuple[float, float] | None,
) -> np.ndarray:
    if amp <= 0 or wavelength <= 0:
        return image
    ny, nx = image.shape
    if pixel_size is None:
        dx = dy = 1.0
    else:
        dx, dy = float(pixel_size[0]), float(pixel_size[1])
        if dx <= 0 or dy <= 0:
            dx = dy = 1.0
    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(angle_deg)
    proj = X * np.cos(theta) + Y * np.sin(theta)
    phase = 2.0 * np.pi * proj / wavelength
    return image + amp * np.sin(phase)


def apply_slope(image: np.ndarray, slope_x: float, slope_y: float) -> np.ndarray:
    if slope_x == 0 and slope_y == 0:
        return image
    ny, nx = image.shape
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y)
    return image + slope_x * X + slope_y * Y


def apply_height_jitter(image: np.ndarray, sigma: float, rng) -> np.ndarray:
    if sigma <= 0:
        return image
    ny = image.shape[0]
    offsets = rng.normal(0.0, sigma, size=ny)
    return image + offsets[:, None]


def apply_drift(image: np.ndarray, shear: float, rng) -> np.ndarray:
    if shear == 0:
        return image
    ny, nx = image.shape
    max_shift = int(abs(shear) * nx)
    if max_shift == 0:
        return image
    shifts = np.linspace(-max_shift, max_shift, ny).astype(int)
    out = np.empty_like(image)
    for i in range(ny):
        out[i] = np.roll(image[i], shifts[i])
    return out


def apply_tip_instability(image: np.ndarray, strength: float, rng) -> np.ndarray:
    if strength <= 0:
        return image
    out = image.copy()
    ny, nx = out.shape

    p_jump = min(0.5, 0.05 + 0.4 * strength)
    jump_rows = rng.random(ny) < p_jump
    jump_vals = rng.normal(0.0, 0.15 * strength, size=ny)
    out[jump_rows] += jump_vals[jump_rows, None]

    shift_std = strength * 2.0
    shifts = rng.normal(0.0, shift_std, size=ny).astype(int)
    out = np.vstack([np.roll(out[i], shifts[i]) for i in range(ny)])

    if rng.random() < strength:
        start = int(rng.integers(0, max(1, ny // 2)))
        length = int(rng.integers(max(2, ny // 8), max(3, ny // 3)))
        end = min(ny, start + length)
        out[start:end] = gaussian_blur(out[start:end], sigma=0.6 + 1.0 * strength)

    return out


def _shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if dx == 0 and dy == 0:
        return image
    ix = int(np.floor(dx))
    iy = int(np.floor(dy))
    fx = dx - ix
    fy = dy - iy

    base = np.roll(image, shift=(iy, ix), axis=(0, 1))
    if fx == 0 and fy == 0:
        return base

    base_x = np.roll(image, shift=(iy, ix + 1), axis=(0, 1))
    base_y = np.roll(image, shift=(iy + 1, ix), axis=(0, 1))
    base_xy = np.roll(image, shift=(iy + 1, ix + 1), axis=(0, 1))

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy
    return w00 * base + w10 * base_x + w01 * base_y + w11 * base_xy


def _sample_radial_offset(rng, offset_range):
    if offset_range is None:
        return 0.0, 0.0
    if isinstance(offset_range, (tuple, list)) and len(offset_range) == 2:
        r = float(rng.uniform(offset_range[0], offset_range[1]))
    else:
        r = float(offset_range)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return r * np.cos(angle), r * np.sin(angle)


def _sigma_to_pixels(sigma_angstrom: float, pixel_size: tuple[float, float] | None) -> float:
    if pixel_size is None:
        return sigma_angstrom
    px = 0.5 * (pixel_size[0] + pixel_size[1])
    if px <= 0:
        return sigma_angstrom
    return sigma_angstrom / px


def apply_noise(image: np.ndarray, noise_cfg, rng, pixel_size: tuple[float, float] | None = None) -> np.ndarray:
    tip_sigma = sample_uniform(rng, noise_cfg.tip_sigma)
    gaussian_sigma = sample_uniform(rng, noise_cfg.gaussian_noise_sigma)
    line_sigma = sample_uniform(rng, noise_cfg.line_noise_sigma)
    background_sigma = sample_uniform(rng, noise_cfg.background_sigma)
    drift_shear = sample_uniform(rng, noise_cfg.drift_shear)
    height_sigma = sample_uniform(rng, noise_cfg.height_jitter_sigma)
    tip_instability = sample_uniform(rng, noise_cfg.tip_instability)
    vibration_amp = sample_uniform(rng, noise_cfg.vibration_amp)
    vibration_wavelength = sample_uniform(rng, noise_cfg.vibration_wavelength)
    vibration_angle = sample_uniform(rng, noise_cfg.vibration_angle_deg)
    slope_x = sample_uniform(rng, noise_cfg.slope_x)
    slope_y = sample_uniform(rng, noise_cfg.slope_y)

    tip_mode = str(getattr(noise_cfg, "tip_mode", "single")).lower()
    if tip_mode == "multi":
        count_spec = getattr(noise_cfg, "tip_count", (2, 3))
        if isinstance(count_spec, (tuple, list)) and len(count_spec) == 2:
            n_tips = int(rng.integers(int(count_spec[0]), int(count_spec[1]) + 1))
        else:
            n_tips = int(count_spec)
        n_tips = max(1, n_tips)

        weights = []
        shifts = []
        sigmas = []
        z_decay = float(getattr(noise_cfg, "tip_z_decay", 1.0))
        weight_spec = getattr(noise_cfg, "tip_weight_range", (0.0, 0.7))
        offset_spec = getattr(noise_cfg, "tip_offset_range", (0.0, 1.0))
        z_spec = getattr(noise_cfg, "tip_z_range", (0.0, 1.0))

        for i in range(n_tips):
            if i == 0:
                dx_a, dy_a = 0.0, 0.0
                dz = 0.0
                w = 1.0
            else:
                dx_a, dy_a = _sample_radial_offset(rng, offset_spec)
                dz = sample_uniform(rng, z_spec)
                if isinstance(weight_spec, (tuple, list)) and len(weight_spec) == 2:
                    w = float(rng.uniform(weight_spec[0], weight_spec[1]))
                else:
                    w = float(weight_spec)
                w *= float(np.exp(-abs(float(dz)) / max(1e-6, z_decay)))

            if pixel_size is None:
                dx_px, dy_px = dx_a, dy_a
            else:
                dx_px = dx_a / max(1e-9, float(pixel_size[0]))
                dy_px = dy_a / max(1e-9, float(pixel_size[1]))

            sigma_i = sample_uniform(rng, noise_cfg.tip_sigma)
            sigmas.append(float(sigma_i))
            shifts.append((dx_px, dy_px))
            weights.append(w)

        out = np.zeros_like(image)
        weight_sum = 0.0
        for (dx_px, dy_px), w, sigma_i in zip(shifts, weights, sigmas):
            sigma_px = _sigma_to_pixels(float(sigma_i), pixel_size)
            blurred = gaussian_blur(image, sigma_px)
            out += _shift_image(blurred, dx_px, dy_px) * w
            weight_sum += w
        if weight_sum > 0:
            out /= weight_sum
    else:
        tip_sigma_px = _sigma_to_pixels(float(tip_sigma), pixel_size)
        out = gaussian_blur(image, tip_sigma_px)

    out = add_gaussian_noise(out, gaussian_sigma, rng)
    corr_px = noise_cfg.line_noise_corr
    if pixel_size is not None:
        # line noise varies along rows (y direction)
        py = pixel_size[1]
        if py > 0:
            corr_px = float(noise_cfg.line_noise_corr) / py
    out = add_line_noise(out, line_sigma, corr_px, rng)
    background_corr_px = noise_cfg.background_corr
    if pixel_size is not None:
        py = pixel_size[1]
        if py > 0:
            background_corr_px = float(noise_cfg.background_corr) / py
    out = add_background_variation(out, background_sigma, background_corr_px, rng)
    out = add_vibration_noise(out, vibration_amp, vibration_wavelength, vibration_angle, pixel_size)
    out = apply_slope(out, slope_x, slope_y)
    out = apply_drift(out, drift_shear, rng)
    out = apply_height_jitter(out, height_sigma, rng)
    out = apply_tip_instability(out, tip_instability, rng)
    return out
