"""
File: iterative_reconstruction.py
Mô tả: Triển khai thuật toán tái tạo ảnh lặp đại số SART (Simultaneous Algebraic
Reconstruction Technique) cho mô hình y = A*x + epsilon với ràng buộc không âm (x >= 0).
Bao gồm kiểm định tính nhất quán của cặp toán tử chiếu thẳng (A) và chiếu ngược (A^T).
"""

import numpy as np
from skimage.transform import radon, iradon


def create_circular_mask(size: int) -> np.ndarray:
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = ((x - center) ** 2 + (y - center) ** 2) <= (size / 2.0) ** 2
    return mask.astype(np.float64)


def forward_operator_A(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return radon(x, theta=theta, circle=True)


def adjoint_operator_AT(y: np.ndarray, theta: np.ndarray, output_size: int) -> np.ndarray:
    raw_bp = iradon(
        y,
        theta=theta,
        output_size=output_size,
        filter_name=None,
        interpolation="linear",
        circle=True,
    )
    scale_correction = (2.0 * len(theta)) / np.pi
    return raw_bp * scale_correction


def verify_operators(size: int = 64, num_angles: int = 30, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    mask = create_circular_mask(size)
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

    x_test = rng.random((size, size)) * mask
    y_test = rng.random((size, num_angles))

    Ax = forward_operator_A(x_test, theta)
    ATy = adjoint_operator_AT(y_test, theta, output_size=size) * mask

    inner_1 = float(np.sum(Ax * y_test))
    inner_2 = float(np.sum(x_test * ATy))
    rel_diff = abs(inner_1 - inner_2) / max(abs(inner_1), 1e-12)

    return {
        "inner_Ax_y": inner_1,
        "inner_x_ATy": inner_2,
        "relative_adjoint_mismatch": rel_diff,
    }


def sart_reconstruction(
    sinogram: np.ndarray,
    theta: np.ndarray,
    num_iterations: int = 20,
    relaxation: float = 0.5,
    initial_image: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
    record_milestones: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    output_size = sinogram.shape[0]
    num_angles = len(theta)
    fov_mask = create_circular_mask(output_size)

    if initial_image is None:
        x = np.zeros((output_size, output_size), dtype=np.float64)
    else:
        x = initial_image.copy().astype(np.float64) * fov_mask

    if record_milestones is None:
        record_milestones = [num_iterations]

    R_all = forward_operator_A(fov_mask, theta)
    R_safe = np.where(R_all > 1e-3, R_all, np.inf)

    C_per_angle = []
    ones_ray = np.ones((output_size, 1), dtype=np.float64)
    for idx_m in range(num_angles):
        ang_arr = np.array([theta[idx_m]])
        c_m = adjoint_operator_AT(ones_ray, ang_arr, output_size) * fov_mask
        c_m_safe = np.where(c_m > 1e-3, c_m, np.inf)
        C_per_angle.append(c_m_safe)

    step = 7 if np.gcd(7, num_angles) == 1 else 1
    angle_order = [(k * step) % num_angles for k in range(num_angles)]

    history = {
        "iteration": [],
        "residual_norm": [],
        "rmse_vs_gt": [],
        "milestone_images": {},
    }

    for it in range(1, num_iterations + 1):
        for idx_m in angle_order:
            ang_arr = np.array([theta[idx_m]])
            y_m = sinogram[:, idx_m : idx_m + 1]

            y_pred_m = forward_operator_A(x, ang_arr)
            norm_residual = (y_m - y_pred_m) / R_safe[:, idx_m : idx_m + 1]
            back_corr = adjoint_operator_AT(norm_residual, ang_arr, output_size)
            
            x = x + relaxation * (back_corr / C_per_angle[idx_m])
            x = np.clip(x, 0.0, None) * fov_mask

        full_pred = forward_operator_A(x, theta)
        res_norm = float(np.linalg.norm(full_pred - sinogram))
        history["iteration"].append(it)
        history["residual_norm"].append(res_norm)

        if ground_truth is not None:
            valid = fov_mask > 0.5
            rmse_val = float(np.sqrt(np.mean((x[valid] - ground_truth[valid]) ** 2)))
            history["rmse_vs_gt"].append(rmse_val)

        if it in record_milestones:
            history["milestone_images"][it] = x.copy()

    return x, history