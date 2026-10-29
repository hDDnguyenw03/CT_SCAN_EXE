from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parallel_beam import create_phantom, forward_parallel_beam, reconstruct_parallel_fbp
from fan_beam import (
    FanBeamGeometry,
    forward_fan_beam,
    reconstruct_fan_beam_direct_fbp,
)
from iterative_reconstruction import verify_operators, sart_reconstruction
from evaluation import evaluate_reconstruction

def setup_directories(base_dir: Path) -> dict[str, Path]:
    paths = {
        "root": base_dir,
        "report": base_dir / "Report",
        "code": base_dir / "Code",
        "results": base_dir / "Results",
        "figures": base_dir / "Results" / "figures",
        "tables": base_dir / "Results" / "tables",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def plot_fan_beam_geometry(geom: FanBeamGeometry, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    fov_r = geom.image_size / 2.0
    fov_circle = plt.Circle((0, 0), fov_r, color="royalblue", alpha=0.2, label="Vùng chụp FOV")
    ax.add_patch(fov_circle)
    orbit_circle = plt.Circle((0, 0), geom.d_so, color="gray", linestyle="--", fill=False, label="Quỹ đạo nguồn")
    ax.add_patch(orbit_circle)
    beta0 = np.deg2rad(90.0)
    sx, sy = geom.d_so * np.cos(beta0), geom.d_so * np.sin(beta0)
    ax.plot(sx, sy, "ro", markersize=8, label="Tiêu điểm nguồn")
    ax.plot(0, 0, "k+", markersize=10, markeredgewidth=2, label="Tâm quay")
    gamma_vals = geom.gamma_rad
    det_x = sx - geom.d_sd * np.cos(beta0 + gamma_vals)
    det_y = sy - geom.d_sd * np.sin(beta0 + gamma_vals)
    ax.plot(det_x, det_y, "g-", linewidth=3, label="Dãy đầu dò")
    ax.set_xlim(-geom.d_so * 1.25, geom.d_so * 1.25)
    ax.set_ylim(-geom.d_so * 1.25, geom.d_so * 1.25)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def save_reconstruction_figures(
    phantom: np.ndarray,
    parallel_sinogram: np.ndarray,
    fan_sinogram: np.ndarray,
    parallel_reconstructions: dict[int, np.ndarray],
    fan_reconstruction: np.ndarray,
    sart_images: dict[int, np.ndarray],
    sart_history: dict,
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, image, title in (
        (axes[0], parallel_sinogram, "Sinogram chùm song song (60 góc)"),
        (axes[1], fan_sinogram, "Sinogram chùm phân kỳ (fan-beam)"),
    ):
        ax.imshow(image, cmap="gray", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Góc chiếu")
        ax.set_ylabel("Kênh đầu dò")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_sinograms.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    parallel_items = [
        (phantom, "Ảnh gốc"),
        (parallel_reconstructions[30], "FBP - 30 góc"),
        (parallel_reconstructions[60], "FBP - 60 góc"),
        (parallel_reconstructions[180], "FBP - 180 góc"),
    ]
    for ax, (image, title) in zip(axes, parallel_items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "04_parallel_reconstruction.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, title in (
        (axes[0], phantom, "Ảnh gốc"),
        (axes[1], fan_sinogram, "Sinogram fan-beam"),
        (axes[2], fan_reconstruction, "Fan-beam FBP"),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_fan_beam_reconstruction.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, iteration in zip(axes.flat[:3], sorted(sart_images)):
        ax.imshow(sart_images[iteration], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"SART - {iteration} vòng lặp")
        ax.axis("off")
    axes.flat[3].plot(
        sart_history["iteration"],
        sart_history["residual_norm"],
        marker=".",
        markersize=3,
    )
    axes.flat[3].set_title("Chuẩn sai số dữ liệu")
    axes.flat[3].set_xlabel("Vòng lặp")
    axes.flat[3].set_ylabel(r"$\|Ax-b\|_2$")
    axes.flat[3].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_sart_iterations.png", dpi=250)
    plt.close(fig)


def run_all_experiments():
    np.random.seed(2026)
    base_dir = Path(__file__).resolve().parent.parent
    paths = setup_directories(base_dir)
    fig_dir = paths["figures"]

    print("=====================================================================")
    print(" BẮT ĐẦU CHƯƠNG TRÌNH MÔ PHỎNG TÁI TẠO ẢNH CT (PARALLEL, FAN, SART)")
    print("=====================================================================")

    op_check = verify_operators(size=64, num_angles=30, seed=2026)
    print(f"[Kiểm định] <Ax, y> = {op_check['inner_Ax_y']:.4f}, Sai lệch = {op_check['relative_adjoint_mismatch']*100:.2f}%")

    img_size = 256
    phantom = create_phantom(size=img_size)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(phantom, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Ảnh gốc Shepp-Logan Phantom (256x256)")
    fig.colorbar(im, ax=ax)
    fig.savefig(fig_dir / "01_phantom_original.png", dpi=250)
    plt.close(fig)

    metrics_records = []

    print("\n[Trường hợp A] Đang mô phỏng Parallel-beam CT...")
    angle_list = [30, 60, 180]
    parallel_reconstructions = {}
    sinogram_60, theta_60 = None, None

    for n_ang in angle_list:
        theta = np.linspace(0.0, 180.0, n_ang, endpoint=False)
        sino, s_coords = forward_parallel_beam(phantom, theta)
        rec_fbp = reconstruct_parallel_fbp(sino, theta, output_size=img_size, filter_name="ramp")
        parallel_reconstructions[n_ang] = rec_fbp

        if n_ang == 60:
            sinogram_60, theta_60 = sino, theta

        m = evaluate_reconstruction(phantom, rec_fbp)
        metrics_records.append({
            "Phương pháp": "Parallel-beam FBP",
            "Góc chiếu": n_ang,
            "Số vòng lặp": "Không",
            "RMSE": m["RMSE"],
            "RelError": m["RelError"],
            "PSNR (dB)": m["PSNR_dB"],
            "SSIM": m["SSIM"],
            "Nhận xét": f"{n_ang} góc chiếu",
        })

    print("\n[Trường hợp B] Đang mô phỏng Fan-beam CT...")
    fan_geom = FanBeamGeometry(image_size=img_size)
    plot_fan_beam_geometry(fan_geom, fig_dir / "02_fan_beam_geometry.png")
    
    fan_sino = forward_fan_beam(phantom, fan_geom, step_size=0.5)
    rec_fan_direct = reconstruct_fan_beam_direct_fbp(fan_sino, fan_geom)
    m_fan_direct = evaluate_reconstruction(phantom, rec_fan_direct)
    metrics_records.append({
        "Phương pháp": "Fan-beam Direct FBP",
        "Góc chiếu": 360,
        "Số vòng lặp": "Không",
        "RMSE": m_fan_direct["RMSE"],
        "RelError": m_fan_direct["RelError"],
        "PSNR (dB)": m_fan_direct["PSNR_dB"],
        "SSIM": m_fan_direct["SSIM"],
        "Nhận xét": "Hình học phân kỳ",
    })

    print("\n[Trường hợp C] Đang chạy lặp SART (60 góc chiếu)...")
    milestones = [5, 15, 30]
    _, sart_hist = sart_reconstruction(
        sinogram_60,
        theta_60,
        num_iterations=30,
        relaxation=0.4,
        ground_truth=phantom,
        record_milestones=milestones,
    )
    save_reconstruction_figures(
        phantom,
        sinogram_60,
        fan_sino,
        parallel_reconstructions,
        rec_fan_direct,
        sart_hist["milestone_images"],
        sart_hist,
        fig_dir,
    )
    
    for k_iter in milestones:
        img_k = sart_hist["milestone_images"][k_iter]
        m_k = evaluate_reconstruction(phantom, img_k)
        metrics_records.append({
            "Phương pháp": "Iterative SART",
            "Góc chiếu": 60,
            "Số vòng lặp": k_iter,
            "RMSE": m_k["RMSE"],
            "RelError": m_k["RelError"],
            "PSNR (dB)": m_k["PSNR_dB"],
            "SSIM": m_k["SSIM"],
            "Nhận xét": f"SART {k_iter} vòng lặp",
        })

    df_metrics = pd.DataFrame(metrics_records)
    csv_main_path = paths["results"] / "metrics.csv"
    df_metrics.to_csv(csv_main_path, index=False, encoding="utf-8-sig")

    print("\n=====================================================================")
    print(" BẢNG TỔNG HỢP CHỈ SỐ ĐÁNH GIÁ ĐỊNH LƯỢNG:")
    print("=====================================================================")
    print(df_metrics.to_string(index=False))
    print(f"\n[Hoàn tất] Đã lưu kết quả tại: {paths['results']}")

if __name__ == "__main__":
    run_all_experiments()