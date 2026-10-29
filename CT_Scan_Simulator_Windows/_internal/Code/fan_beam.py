from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from skimage.transform import iradon

@dataclass
class FanBeamGeometry:
    image_size: int = 256
    d_so: float = 400.0          
    d_sd: float = 800.0          
    num_detectors: int = 365     
    fan_angle_deg: float = 60.0  
    num_angles: int = 360        

    @property
    def gamma_deg(self) -> np.ndarray:
        half_fan = self.fan_angle_deg / 2.0
        return np.linspace(-half_fan, half_fan, self.num_detectors)

    @property
    def gamma_rad(self) -> np.ndarray:
        return np.deg2rad(self.gamma_deg)

    @property
    def beta_deg(self) -> np.ndarray:
        return np.linspace(0.0, 360.0, self.num_angles, endpoint=False)

    @property
    def beta_rad(self) -> np.ndarray:
        return np.deg2rad(self.beta_deg)

def forward_fan_beam(phantom: np.ndarray, geom: FanBeamGeometry, step_size: float = 0.5) -> np.ndarray:
    n = geom.image_size
    center = (n - 1) / 2.0
    radius = n / 2.0
    l_min = max(0.0, geom.d_so - radius * 1.05)
    l_max = geom.d_so + radius * 1.05
    l_samples = np.arange(l_min, l_max, step_size)
    num_samples = len(l_samples)
    gamma_rad = geom.gamma_rad[:, None]
    l_grid = l_samples[None, :]
    fan_sinogram = np.zeros((geom.num_detectors, geom.num_angles), dtype=np.float64)
    for idx_b, beta in enumerate(geom.beta_rad):
        ray_angle = beta + gamma_rad
        x_phys = geom.d_so * np.cos(beta) - l_grid * np.cos(ray_angle)
        y_phys = geom.d_so * np.sin(beta) - l_grid * np.sin(ray_angle)
        col_coords = center + x_phys
        row_coords = center - y_phys
        coords = np.vstack([row_coords.ravel(), col_coords.ravel()])
        sampled_vals = map_coordinates(phantom, coords, order=1, mode="constant", cval=0.0)
        sampled_vals = sampled_vals.reshape(geom.num_detectors, num_samples)
        fan_sinogram[:, idx_b] = np.sum(sampled_vals, axis=1) * step_size
    return fan_sinogram

def rebin_fan_to_parallel(fan_sinogram: np.ndarray, geom: FanBeamGeometry, num_parallel_angles: int = 180) -> tuple[np.ndarray, np.ndarray]:
    n = geom.image_size
    half_len = (n - 1) / 2.0
    s_coords = np.linspace(-half_len, half_len, n)
    theta_deg = np.linspace(0.0, 180.0, num_parallel_angles, endpoint=False)
    beta_ext = np.concatenate([geom.beta_deg - 360.0, geom.beta_deg, geom.beta_deg + 360.0])
    sino_ext = np.concatenate([fan_sinogram, fan_sinogram, fan_sinogram], axis=1)
    interpolator = RegularGridInterpolator((geom.gamma_deg, beta_ext), sino_ext, method="linear", bounds_error=False, fill_value=0.0)
    s_grid, theta_grid = np.meshgrid(s_coords, theta_deg, indexing="ij")
    gamma_query_deg = np.rad2deg(np.arcsin(np.clip(s_grid / geom.d_so, -0.99, 0.99)))
    beta_query_deg = (theta_grid - gamma_query_deg) % 360.0
    query_points = np.stack([gamma_query_deg.ravel(), beta_query_deg.ravel()], axis=-1)
    parallel_sinogram = interpolator(query_points).reshape(n, num_parallel_angles)
    return parallel_sinogram, theta_deg

def reconstruct_fan_beam_rebinned(fan_sinogram: np.ndarray, geom: FanBeamGeometry, num_parallel_angles: int = 180, filter_name: str = "ramp") -> np.ndarray:
    parallel_sino, theta_deg = rebin_fan_to_parallel(fan_sinogram, geom, num_parallel_angles)
    return iradon(parallel_sino, theta=theta_deg, output_size=geom.image_size, filter_name=filter_name, interpolation="linear", circle=True)

def reconstruct_fan_beam_direct_fbp(
    fan_sinogram: np.ndarray,
    geom: FanBeamGeometry,
    filter_name: str | None = "hann",
) -> np.ndarray:
    n = geom.image_size
    num_det, num_angles = fan_sinogram.shape
    d_gamma = geom.gamma_rad[1] - geom.gamma_rad[0]
    d_beta = geom.beta_rad[1] - geom.beta_rad[0]
    cos_weight = np.cos(geom.gamma_rad)[:, None]
    weighted_sino = fan_sinogram * cos_weight
    n_fft = int(2 ** np.ceil(np.log2(2 * num_det)))
    freqs = np.fft.fftfreq(n_fft, d=d_gamma * geom.d_so)
    ramp_filter = np.abs(freqs)
    if filter_name is None:
        freq_filter = np.ones_like(ramp_filter)
    elif filter_name == "ramp":
        freq_filter = ramp_filter
    elif filter_name in ("hann", "hamming"):
        window_base = 0.5 * (1.0 + np.cos(np.pi * freqs / np.max(np.abs(freqs))))
        if filter_name == "hamming":
            window_base = 0.54 + 0.46 * np.cos(
                np.pi * freqs / np.max(np.abs(freqs))
            )
        freq_filter = ramp_filter * window_base
    else:
        raise ValueError(
            "filter_name phải là 'ramp', 'hann', 'hamming' hoặc None."
        )
    sino_padded = np.zeros((n_fft, num_angles), dtype=np.float64)
    sino_padded[:num_det, :] = weighted_sino
    sino_fft = np.fft.fft(sino_padded, axis=0)
    filtered_sino = np.real(np.fft.ifft(sino_fft * freq_filter[:, None], axis=0))[:num_det, :]
    center = (n - 1) / 2.0
    col_idx, row_idx = np.meshgrid(np.arange(n), np.arange(n))
    x_grid = col_idx - center
    y_grid = center - row_idx
    fov_mask = (x_grid ** 2 + y_grid ** 2) <= (n / 2.0) ** 2
    rec_img = np.zeros((n, n), dtype=np.float64)
    gamma_min = geom.gamma_rad[0]
    for idx_b, beta in enumerate(geom.beta_rad):
        sx, sy = geom.d_so * np.cos(beta), geom.d_so * np.sin(beta)
        dx, dy = sx - x_grid, sy - y_grid
        dist_sq = dx ** 2 + dy ** 2
        dist = np.sqrt(dist_sq)
        proj_along = dx * np.cos(beta) + dy * np.sin(beta)
        proj_perp = -dx * np.sin(beta) + dy * np.cos(beta)
        gamma_pixel = np.arctan2(proj_perp, proj_along)
        det_float_idx = (gamma_pixel - gamma_min) / d_gamma
        valid = (det_float_idx >= 0) & (det_float_idx < num_det - 1) & fov_mask
        idx0 = np.floor(det_float_idx[valid]).astype(int)
        frac = det_float_idx[valid] - idx0
        col_filtered = filtered_sino[:, idx_b]
        interp_val = (1.0 - frac) * col_filtered[idx0] + frac * col_filtered[idx0 + 1]
        weight = (geom.d_so ** 2) / np.maximum(dist_sq[valid], 1.0)
        rec_img[valid] += weight * interp_val * d_beta
    return rec_img * 0.5