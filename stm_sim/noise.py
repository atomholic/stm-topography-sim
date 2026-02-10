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
