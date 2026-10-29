"""
File: evaluation.py
Mô tả: Các hàm đánh giá định lượng chất lượng ảnh tái tạo CT (RMSE, Relative Error,
PSNR, SSIM, và CNR). Tuân thủ quy trình chuẩn hóa miền giá trị thống nhất theo
ảnh gốc để không làm sai lệch kết quả so sánh giữa các thuật toán.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate_reconstruction(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    clip_negative: bool = True,
) -> dict:
    assert reference.shape == reconstructed.shape, "Kích thước ảnh tái tạo và ảnh gốc phải khớp nhau!"
    n = reference.shape[0]
    center = (n - 1) / 2.0
    y, x = np.ogrid[:n, :n]
    fov_mask = ((x - center) ** 2 + (y - center) ** 2) <= (n / 2.0) ** 2

    ref_eval = reference * fov_mask
    rec_eval = reconstructed.copy()
    if clip_negative:
        rec_eval = np.clip(rec_eval, 0.0, None)
    rec_eval = rec_eval * fov_mask

    diff_fov = rec_eval[fov_mask] - ref_eval[fov_mask]
    rmse = float(np.sqrt(np.mean(diff_fov ** 2)))
    reference_norm = float(np.linalg.norm(ref_eval[fov_mask]))
    rel_error = float(np.linalg.norm(diff_fov) / max(reference_norm, 1e-12))

    data_range = float(np.max(ref_eval) - np.min(ref_eval))
    if data_range <= 0:
        data_range = 1.0

    mse_fov = float(np.mean(diff_fov ** 2))
    psnr = float(10.0 * np.log10((data_range ** 2) / max(mse_fov, 1e-12)))

    ssim = float(structural_similarity(ref_eval, rec_eval, data_range=data_range))

    return {
        "RMSE": round(rmse, 5),
        "RelError": round(rel_error, 5),
        "PSNR_dB": round(psnr, 2),
        "SSIM": round(ssim, 4),
    }