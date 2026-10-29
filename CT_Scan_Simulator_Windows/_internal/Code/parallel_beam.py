"""
File: parallel_beam.py
Mô tả: Mô phỏng thu nhận dữ liệu chiếu chùm tia song song (Parallel-beam CT)
và tái tạo ảnh bằng phương pháp Chiếu ngược có lọc (Filtered Backprojection - FBP)
cùng Chiếu ngược không lọc (Unfiltered Backprojection - UBP).
"""

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon


def create_phantom(size: int = 256) -> np.ndarray:
    """
    Tạo ảnh phantom Shepp-Logan chuẩn hóa về kích thước (size x size)
    và miền giá trị [0.0, 1.0].
    """
    raw_phantom = shepp_logan_phantom()
    phantom = resize(raw_phantom, (size, size), anti_aliasing=True, preserve_range=True)
    phantom = np.clip(phantom, 0.0, 1.0).astype(np.float64)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    circle_mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 2) ** 2
    return phantom * circle_mask


def forward_parallel_beam(phantom: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính biến đổi Radon (Sinogram) cho cấu hình chùm tia song song.
    
    Parameters:
        phantom (np.ndarray): Ảnh đối tượng 2D kích thước N x N.
        theta (np.ndarray): Mảng các góc chiếu (đơn vị: độ), nằm trong [0, 180).
        
    Returns:
        sinogram (np.ndarray): Ma trận dữ liệu chiếu kích thước [N_detectors, N_angles].
        detector_coords (np.ndarray): Tọa độ không gian của dãy đầu dò s (đơn vị: pixel).
    """
    sinogram = radon(phantom, theta=theta, circle=True)
    num_detectors = sinogram.shape[0]
    half_len = (num_detectors - 1) / 2.0
    detector_coords = np.linspace(-half_len, half_len, num_detectors)
    return sinogram, detector_coords


def reconstruct_parallel_fbp(
    sinogram: np.ndarray,
    theta: np.ndarray,
    output_size: int = 256,
    filter_name: str = "ramp",
    interpolation: str = "linear",
) -> np.ndarray:
    """
    Tái tạo ảnh từ sinogram chùm tia song song bằng Filtered Backprojection (FBP).
    
    Parameters:
        sinogram (np.ndarray): Ma trận sinogram [N_detectors, N_angles].
        theta (np.ndarray): Mảng góc chiếu (độ).
        output_size (int): Kích thước ảnh đầu ra (output_size x output_size).
        filter_name (str | None): Tên bộ lọc ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', hoặc None cho UBP).
        interpolation (str): Phương pháp nội suy ('linear', 'nearest', 'cubic').
        
    Returns:
        rec_img (np.ndarray): Ảnh tái tạo 2D.
    """
    rec_img = iradon(
        sinogram,
        theta=theta,
        output_size=output_size,
        filter_name=filter_name,
        interpolation=interpolation,
        circle=True,
    )
    return rec_img