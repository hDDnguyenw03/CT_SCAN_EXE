"""Streamlit dashboard for user-configurable CT reconstruction simulations."""

import io
import hashlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from evaluation import evaluate_reconstruction
from export_results import create_results_zip
from fan_beam import (
    FanBeamGeometry,
    forward_fan_beam,
    reconstruct_fan_beam_direct_fbp,
)
from iterative_reconstruction import sart_reconstruction
from parallel_beam import forward_parallel_beam, reconstruct_parallel_fbp
from phantom import (
    MAX_UPLOAD_BYTES,
    create_builtin_phantom,
    create_custom_phantom,
    decode_uploaded_image,
    image_preview_rgb,
    prepare_phantom,
)


FILTER_LABELS = {
    "ramp": "Ramp",
    "hann": "Hann",
    "hamming": "Hamming",
    None: "None (không lọc)",
}
METHODS = {
    "Parallel-beam FBP": "parallel",
    "Fan-beam Direct FBP": "fan",
    "Iterative SART": "sart",
}
BUILTIN_PHANTOMS = (
    "Shepp–Logan phantom",
    "Modified Shepp–Logan phantom",
    "Phantom hình học (tròn và elip)",
    "Lát cắt với các vùng suy giảm tương đối",
)
SHAPE_LABELS = {
    "Circle": "Hình tròn",
    "Ellipse": "Hình elip",
    "Rectangle": "Hình chữ nhật",
}


def _initialize_session() -> None:
    if "phantom" not in st.session_state:
        initial = create_builtin_phantom(BUILTIN_PHANTOMS[1], 128)
        st.session_state["phantom"] = initial
        st.session_state["phantom_name"] = BUILTIN_PHANTOMS[1]
        st.session_state["phantom_size"] = 128
        st.session_state["phantom_source_image"] = None
        st.session_state["phantom_metadata"] = None
        st.session_state["phantom_warnings"] = []
        st.session_state["phantom_results"] = {}
        st.session_state["phantom_structures"] = [
            {
                "shape": "Ellipse",
                "center_x": 0.0,
                "center_y": 0.0,
                "radius_x": 0.68,
                "radius_y": 0.78,
                "angle": 0.0,
                "value": 0.25,
            },
            {
                "shape": "Circle",
                "center_x": -0.2,
                "center_y": 0.12,
                "radius_x": 0.18,
                "radius_y": 0.18,
                "angle": 0.0,
                "value": 0.85,
            },
        ]
    st.session_state.setdefault("builtin_phantom_choice", BUILTIN_PHANTOMS[1])


def _reset_phantom() -> None:
    initial = create_builtin_phantom(BUILTIN_PHANTOMS[1], 128)
    st.session_state["phantom"] = initial
    st.session_state["phantom_name"] = BUILTIN_PHANTOMS[1]
    st.session_state["phantom_size"] = 128
    st.session_state["phantom_source_image"] = None
    st.session_state["phantom_metadata"] = None
    st.session_state["phantom_warnings"] = []
    st.session_state["phantom_results"] = {}
    st.session_state["phantom_structures"] = [
        {
            "shape": "Ellipse",
            "center_x": 0.0,
            "center_y": 0.0,
            "radius_x": 0.68,
            "radius_y": 0.78,
            "angle": 0.0,
            "value": 0.25,
        },
        {
            "shape": "Circle",
            "center_x": -0.2,
            "center_y": 0.12,
            "radius_x": 0.18,
            "radius_y": 0.18,
            "angle": 0.0,
            "value": 0.85,
        },
    ]
    st.session_state["builtin_phantom_choice"] = BUILTIN_PHANTOMS[1]
    for key in ("phantom_size_choice", "phantom_normalize", "phantom_source_choice"):
        st.session_state.pop(key, None)
    st.session_state.pop("uploaded_phantom", None)
    st.session_state.pop("_decoded_upload_cache", None)
    for key in list(st.session_state.keys()):
        if key.startswith("edit_"):
            st.session_state.pop(key, None)


def _decode_upload_file(uploaded_file) -> tuple[np.ndarray, dict]:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("Tệp ảnh vượt quá giới hạn 32 MB để bảo vệ bộ nhớ.")
    content = uploaded_file.getvalue()
    fingerprint = hashlib.sha256(content).digest()
    cached = st.session_state.get("_decoded_upload_cache")
    if cached is not None and cached[0] == fingerprint:
        return cached[1], cached[2]
    image, metadata = decode_uploaded_image(content)
    st.session_state["_decoded_upload_cache"] = (fingerprint, image, metadata)
    return image, metadata


def _load_selected_phantom(
    source: str,
    builtin_name: str,
    size: int,
    normalize: bool,
    uploaded_file,
) -> None:
    if size > 256:
        st.warning(
            "512×512 images require substantially more time and memory, especially for "
            "Fan-beam and SART. Use a smaller number of projection angles if needed."
        )

    warnings = []
    source_image = None
    metadata = None
    if source == "Phantom dựng sẵn":
        image = create_builtin_phantom(builtin_name, size)
        name = builtin_name
    elif source == "Tải ảnh":
        if uploaded_file is None:
            raise ValueError("Chọn một tệp PNG, JPG/JPEG hoặc TIFF trước khi tải phantom.")
        image, metadata = _decode_upload_file(uploaded_file)
        source_image = image.copy()
        name = uploaded_file.name
    else:
        image = create_custom_phantom(size, st.session_state["phantom_structures"])
        name = "Phantom tự thiết kế"

    processed, warnings = prepare_phantom(image, size, normalize=normalize)
    if (
        metadata
        and metadata["non_finite_count"]
        and not any("NaN/Inf" in warning for warning in warnings)
    ):
        warnings.append(
            f"Phát hiện {metadata['non_finite_count']} giá trị NaN/Inf trong dữ liệu ảnh gốc."
        )
    if np.min(processed) < 0:
        warnings.append(
            "Phantom có giá trị âm; SART hiện áp đặt ràng buộc không âm nên kết quả SART "
            "không thể khớp phần âm."
        )
    st.session_state["phantom"] = processed
    st.session_state["phantom_name"] = name
    st.session_state["phantom_size"] = size
    st.session_state["phantom_source_image"] = source_image
    st.session_state["phantom_metadata"] = metadata
    st.session_state["phantom_warnings"] = warnings
    st.session_state["phantom_results"] = {}
    st.success(f"Đã nạp phantom **{name}** với kích thước {size}×{size}.")


def _simulate(
    phantom: np.ndarray,
    phantom_name: str,
    method: str,
    num_angles: int,
    noise_percent: float,
    filter_name: str | None,
    num_iterations: int,
    relaxation: float,
) -> dict:
    size = int(phantom.shape[0])
    if phantom.shape != (size, size) or not np.isfinite(phantom).all():
        raise ValueError("Phantom phải là ma trận vuông có các giá trị hữu hạn.")

    history = None
    milestones = {}
    if method == "fan":
        geometry = FanBeamGeometry(image_size=size, num_angles=num_angles)
        sinogram = forward_fan_beam(phantom, geometry)
        theta = geometry.beta_deg
    else:
        geometry = None
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        sinogram, _ = forward_parallel_beam(phantom, theta)

    if noise_percent > 0.0:
        noise_scale = float(np.max(np.abs(sinogram))) * noise_percent / 100.0
        rng = np.random.default_rng(seed=42)
        sinogram = sinogram + rng.normal(0.0, noise_scale, sinogram.shape)

    if method == "parallel":
        reconstruction = reconstruct_parallel_fbp(
            sinogram,
            theta,
            output_size=size,
            filter_name=filter_name,
        )
    elif method == "fan":
        if geometry is None:
            raise ValueError("Fan-beam reconstruction requires fan-beam geometry.")
        reconstruction = reconstruct_fan_beam_direct_fbp(
            sinogram,
            geometry,
            filter_name=filter_name,
        )
    elif method == "sart":
        milestones_list = sorted(
            {1, max(1, num_iterations // 2), num_iterations}
        )
        reconstruction, history = sart_reconstruction(
            sinogram,
            theta,
            num_iterations=num_iterations,
            relaxation=relaxation,
            ground_truth=phantom,
            record_milestones=milestones_list,
        )
        milestones = history["milestone_images"]
    else:
        raise ValueError(f"Phương pháp tái tạo không được hỗ trợ: {method}")

    metrics = evaluate_reconstruction(phantom, reconstruction)
    return {
        "phantom": phantom.copy(),
        "phantom_name": phantom_name,
        "sinogram": sinogram,
        "reconstruction": reconstruction,
        "metrics": metrics,
        "method": method,
        "method_label": next(label for label, key in METHODS.items() if key == method),
        "size": size,
        "num_angles": num_angles,
        "noise_percent": noise_percent,
        "filter_name": filter_name,
        "num_iterations": num_iterations,
        "relaxation": relaxation,
        "history": history,
        "milestones": milestones,
    }


def _show_preview() -> None:
    phantom = st.session_state["phantom"]
    st.subheader("Phantom hiện tại")
    columns = st.columns(2)
    source_image = st.session_state["phantom_source_image"]
    if source_image is not None:
        columns[0].image(
            image_preview_rgb(source_image),
            caption="Ảnh tải lên trước tiền xử lý",
            alt="Ảnh phantom tải lên trước tiền xử lý",
        )
    else:
        columns[0].image(
            phantom,
            caption="Phantom đầu vào",
            alt="Phantom đầu vào trước mô phỏng",
            clamp=True,
        )
    columns[1].image(
        phantom,
        caption=f"Phantom sau tiền xử lý · {phantom.shape[1]}×{phantom.shape[0]}",
        alt="Phantom sau tiền xử lý",
        clamp=True,
    )
    st.caption(
        f"Tên: {st.session_state['phantom_name']} · "
        f"Kiểu: {phantom.dtype} · "
        f"Miền pixel sau xử lý: [{phantom.min():.5g}, {phantom.max():.5g}]"
    )

    metadata = st.session_state["phantom_metadata"]
    if metadata:
        st.write(
            {
                "Tên tệp": st.session_state["phantom_name"],
                "Định dạng": metadata["format"],
                "Kích thước gốc": f"{metadata['width']}×{metadata['height']}",
                "Số kênh màu": metadata["channels"],
                "Kiểu dữ liệu": metadata["dtype"],
                "Miền giá trị pixel": f"[{metadata['pixel_min']:.5g}, {metadata['pixel_max']:.5g}]",
                "Pixel NaN/Inf": metadata["non_finite_count"],
            }
        )
        st.info(
            "Ảnh tải lên chỉ là ảnh tham chiếu của mô hình mô phỏng; PNG/JPG/TIFF thông "
            "thường không được coi là ảnh CT chuẩn hoặc đơn vị HU."
        )
    for message in st.session_state["phantom_warnings"]:
        st.warning(message)


def _display_result(result: dict) -> None:
    sinogram = result["sinogram"]
    sino_min = float(np.min(sinogram))
    sino_max = float(np.max(sinogram))
    sino_range = sino_max - sino_min
    sinogram_display = (
        (sinogram - sino_min) / sino_range
        if sino_range > 0.0
        else np.zeros_like(sinogram)
    )
    st.subheader(result["method_label"])
    st.caption(
        f"{result['size']}×{result['size']} px · {result['num_angles']} góc · "
        f"nhiễu {result['noise_percent']:.1f}% · "
        f"bộ lọc {FILTER_LABELS[result['filter_name']]}"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    display_min = float(np.min(result["phantom"]))
    display_max = float(np.max(result["phantom"]))
    if display_max <= display_min:
        display_max = display_min + 1.0
    axes[0].imshow(
        result["phantom"],
        cmap="gray",
        vmin=display_min,
        vmax=display_max,
        aspect="equal",
    )
    axes[0].set_title("Phantom đầu vào")
    axes[1].imshow(sinogram_display, cmap="gray", aspect="auto")
    axes[1].set_title(f"Sinogram ({result['method_label']})")
    axes[2].imshow(
        result["reconstruction"],
        cmap="gray",
        vmin=display_min,
        vmax=display_max,
        aspect="equal",
    )
    axes[2].set_title("Ảnh tái tạo")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    st.pyplot(figure, alt="Phantom đầu vào, sinogram và ảnh tái tạo")
    plt.close(figure)

    error_figure, error_axis = plt.subplots(figsize=(6, 6))
    error_axis.imshow(
        np.abs(result["phantom"] - result["reconstruction"]),
        cmap="inferno",
        aspect="equal",
    )
    error_axis.set_title("|Phantom − ảnh tái tạo|")
    error_axis.axis("off")
    error_figure.tight_layout()
    st.pyplot(error_figure, alt="Bản đồ sai số tuyệt đối")
    plt.close(error_figure)

    st.caption("Các chỉ số so sánh với phantom đầu vào của mô hình mô phỏng.")
    metrics = result["metrics"]
    columns = st.columns(4)
    columns[0].metric("RMSE", f"{metrics['RMSE']:.5f}")
    columns[1].metric("Sai số tương đối", f"{metrics['RelError']:.5f}")
    columns[2].metric("PSNR", f"{metrics['PSNR_dB']:.2f} dB")
    columns[3].metric("SSIM", f"{metrics['SSIM']:.4f}")

    if result["method"] == "sart" and result["history"]:
        history = result["history"]
        st.subheader("Các mốc SART và độ phù hợp dữ liệu")
        count = len(result["milestones"]) + 1
        figure, axes = plt.subplots(1, count, figsize=(4 * count, 4))
        axes = np.atleast_1d(axes)
        for axis, (iteration, image) in zip(axes, result["milestones"].items()):
            axis.imshow(image, cmap="gray", aspect="equal")
            axis.set_title(f"{iteration} vòng lặp")
            axis.axis("off")
        axes[-1].plot(history["iteration"], history["residual_norm"])
        axes[-1].set_title("Chuẩn residual ||Ax−b||₂")
        axes[-1].set_xlabel("Vòng lặp")
        axes[-1].grid(alpha=0.25)
        figure.tight_layout()
        st.pyplot(figure, alt="Các mốc ảnh SART và chuẩn residual")
        plt.close(figure)


def _comparison_png(results: dict[str, dict]) -> bytes:
    figure, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4))
    axes = np.atleast_1d(axes)
    phantom = next(iter(results.values()))["phantom"]
    display_min = float(np.min(phantom))
    display_max = float(np.max(phantom))
    if display_max <= display_min:
        display_max = display_min + 1.0
    axes[0].imshow(phantom, cmap="gray", vmin=display_min, vmax=display_max)
    axes[0].set_title("Phantom đầu vào")
    axes[0].axis("off")
    for axis, result in zip(axes[1:], results.values()):
        axis.imshow(
            result["reconstruction"],
            cmap="gray",
            vmin=display_min,
            vmax=display_max,
        )
        axis.set_title(result["method_label"])
        axis.axis("off")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    return buffer.getvalue()


def _display_results() -> None:
    results = st.session_state["phantom_results"]
    if not results:
        st.info("Cấu hình phantom, sau đó chọn **Run Reconstruction** hoặc **Compare Methods**.")
        return

    if len(results) > 1:
        st.subheader("So sánh phương pháp")
        metric_rows = [
            {"Phương pháp": result["method_label"], **result["metrics"]}
            for result in results.values()
        ]
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True)
        method_key = st.selectbox(
            "Xem chi tiết kết quả",
            options=list(results),
            format_func=lambda key: results[key]["method_label"],
        )
    else:
        method_key = next(iter(results))

    _display_result(results[method_key])
    if len(results) > 1:
        st.subheader("Hình so sánh các phương pháp")
        st.image(
            _comparison_png(results),
            caption="Cùng phantom đầu vào, với dữ liệu chiếu đúng hình học của từng phương pháp.",
            alt="So sánh phantom đầu vào và ảnh tái tạo",
        )

    reference_name = st.session_state["phantom_name"]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", reference_name).strip("_") or "phantom"
    first_result = next(iter(results.values()))
    config = {
        "phantom_name": reference_name,
        "phantom_shape": list(st.session_state["phantom"].shape),
        "phantom_dtype": str(st.session_state["phantom"].dtype),
        "reference_note": "Uploaded raster images are simulation references, not calibrated CT/HU data.",
        "methods": [
            {
                "method": result["method_label"],
                "geometry": "fan-beam" if result["method"] == "fan" else "parallel-beam",
                "num_angles": result["num_angles"],
                "noise_percent": result["noise_percent"],
                "filter": result["filter_name"],
                "num_iterations": result["num_iterations"] if result["method"] == "sart" else None,
                "relaxation": result["relaxation"] if result["method"] == "sart" else None,
            }
            for result in results.values()
        ],
    }
    archive = create_results_zip(
        reference_name,
        st.session_state["phantom"],
        results,
        config,
        _comparison_png(results),
    )
    st.download_button(
        "Export Results",
        data=archive,
        file_name=f"{safe_name}_{first_result['size']}px_results.zip",
        mime="application/zip",
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="CT Reconstruction Dashboard", layout="wide")
    _initialize_session()
    st.title("Mô phỏng tái tạo ảnh CT")
    st.write(
        "Chọn hoặc tải phantom, cấu hình phép chiếu và chạy FBP chùm song song, "
        "fan-beam trực tiếp hoặc SART trên cùng dữ liệu."
    )

    st.header("Input Phantom / Phantom đầu vào")
    reset_column, _, _ = st.columns([1, 2, 2])
    if reset_column.button("Reset Phantom"):
        _reset_phantom()
        st.rerun()

    source = st.selectbox(
        "Nguồn phantom",
        options=["Phantom dựng sẵn", "Tải ảnh", "Tự thiết kế"],
        key="phantom_source_choice",
    )
    source_column, option_column, size_column = st.columns([2, 2, 1])
    builtin_name = BUILTIN_PHANTOMS[1]
    uploaded_file = None
    if source != "Tải ảnh":
        st.session_state.pop("_decoded_upload_cache", None)
    if source == "Phantom dựng sẵn":
        builtin_name = source_column.selectbox(
            "Chọn phantom dựng sẵn",
            BUILTIN_PHANTOMS,
            key="builtin_phantom_choice",
        )
    elif source == "Tải ảnh":
        uploaded_file = source_column.file_uploader(
            "Tải ảnh phantom (PNG, JPG/JPEG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="uploaded_phantom",
            help="DICOM không được hỗ trợ. Không diễn giải ảnh raster thông thường thành HU.",
        )
        if uploaded_file is not None:
            try:
                preview_image, _ = _decode_upload_file(uploaded_file)
                st.image(
                    image_preview_rgb(preview_image),
                    caption="Ảnh tải lên (xem trước)",
                    alt="Xem trước ảnh phantom được chọn",
                )
            except ValueError as exc:
                st.error(str(exc))
    else:
        option_column.caption(
            "Tọa độ tâm x/y và bán trục được tính theo nửa chiều rộng/cao ảnh "
            "(tọa độ −1…1; bán trục 0…1). Các hình được vẽ theo thứ tự, hình sau ghi đè hình trước."
        )

        structures = st.session_state["phantom_structures"]
        shape_options = ["Circle", "Ellipse", "Rectangle"]
        with st.form("add_phantom_structure"):
            st.markdown("**Thêm cấu trúc**")
            add_shape = st.selectbox(
                "Loại hình", shape_options, format_func=lambda value: SHAPE_LABELS[value]
            )
            add_columns = st.columns(3)
            add_center_x = add_columns[0].number_input(
                "Tâm x", min_value=-1.0, max_value=1.0, value=0.0, step=0.05
            )
            add_center_y = add_columns[1].number_input(
                "Tâm y", min_value=-1.0, max_value=1.0, value=0.0, step=0.05
            )
            add_radius_x = add_columns[2].number_input(
                "Bán trục x", min_value=0.01, max_value=1.0, value=0.2, step=0.05
            )
            add_columns = st.columns(3)
            add_radius_y = add_columns[0].number_input(
                "Bán trục y", min_value=0.01, max_value=1.0, value=0.2, step=0.05
            )
            add_angle = add_columns[1].number_input(
                "Góc xoay (°)", min_value=-180.0, max_value=180.0, value=0.0, step=5.0
            )
            add_value = add_columns[2].number_input(
                "Cường độ tương đối", min_value=0.0, max_value=1.0, value=0.8, step=0.05
            )
            add_submitted = st.form_submit_button("Thêm cấu trúc", width="stretch")
        if add_submitted:
            radius_y = add_radius_x if add_shape == "Circle" else add_radius_y
            structures.append(
                {
                    "shape": add_shape,
                    "center_x": add_center_x,
                    "center_y": add_center_y,
                    "radius_x": add_radius_x,
                    "radius_y": radius_y,
                    "angle": add_angle,
                    "value": add_value,
                }
            )
            st.session_state["phantom_structures"] = structures
            st.rerun()

        if structures:
            labels = [
                f"{index + 1}. {SHAPE_LABELS[item['shape']]}, giá trị {item['value']:.2f}"
                for index, item in enumerate(structures)
            ]
            selected_structure = st.selectbox("Cấu trúc cần sửa/xóa", range(len(structures)), format_func=lambda index: labels[index])
            item = structures[selected_structure]
            with st.form("edit_phantom_structure"):
                edit_columns = st.columns(3)
                edit_center_x = edit_columns[0].number_input(
                    "Tâm x", min_value=-1.0, max_value=1.0, value=float(item["center_x"]), step=0.05, key=f"edit_x_{selected_structure}"
                )
                edit_center_y = edit_columns[1].number_input(
                    "Tâm y", min_value=-1.0, max_value=1.0, value=float(item["center_y"]), step=0.05, key=f"edit_y_{selected_structure}"
                )
                edit_radius_x = edit_columns[2].number_input(
                    "Bán trục x", min_value=0.01, max_value=1.0, value=float(item["radius_x"]), step=0.05, key=f"edit_rx_{selected_structure}"
                )
                edit_columns = st.columns(3)
                edit_radius_y = edit_columns[0].number_input(
                    "Bán trục y", min_value=0.01, max_value=1.0, value=float(item["radius_y"]), step=0.05, key=f"edit_ry_{selected_structure}"
                )
                edit_angle = edit_columns[1].number_input(
                    "Góc xoay (°)", min_value=-180.0, max_value=180.0, value=float(item["angle"]), step=5.0, key=f"edit_angle_{selected_structure}"
                )
                edit_value = edit_columns[2].number_input(
                    "Cường độ tương đối", min_value=0.0, max_value=1.0, value=float(item["value"]), step=0.05, key=f"edit_value_{selected_structure}"
                )
                edit_columns = st.columns(2)
                update_submitted = edit_columns[0].form_submit_button("Cập nhật cấu trúc", width="stretch")
                delete_submitted = edit_columns[1].form_submit_button("Xóa cấu trúc", width="stretch")
            if update_submitted:
                updated = dict(item)
                updated.update(
                    {
                        "center_x": edit_center_x,
                        "center_y": edit_center_y,
                        "radius_x": edit_radius_x,
                        "radius_y": edit_radius_x if item["shape"] == "Circle" else edit_radius_y,
                        "angle": edit_angle,
                        "value": edit_value,
                    }
                )
                structures[selected_structure] = updated
                st.session_state["phantom_structures"] = structures
                st.rerun()
            if delete_submitted:
                structures.pop(selected_structure)
                st.session_state["phantom_structures"] = structures
                st.rerun()
        if structures:
            structure_preview = create_custom_phantom(
                st.session_state["phantom_size"], structures
            )
            st.image(
                structure_preview,
                caption="Xem trước thiết kế hiện tại",
                alt="Phantom được tạo từ các cấu trúc hình học",
                clamp=True,
            )
        else:
            st.info("Thêm ít nhất một cấu trúc để tạo phantom.")

    size = size_column.selectbox(
        "Kích thước ảnh",
        options=[128, 256, 512],
        index=[128, 256, 512].index(
            st.session_state["phantom_size"]
            if st.session_state["phantom_size"] in (128, 256, 512)
            else 128
        ),
        key="phantom_size_choice",
        format_func=lambda value: f"{value} × {value} px",
    )
    normalize = option_column.checkbox(
        "Chuẩn hóa min–max về [0, 1]",
        value=True,
        key="phantom_normalize",
        help="Thay đổi tuyến tính miền pixel để mô phỏng tương đối; không tạo ra đơn vị HU hay hệ số suy giảm vật lý.",
    )
    if st.button("Load Phantom", type="primary"):
        try:
            _load_selected_phantom(
                source, builtin_name, size, normalize, uploaded_file
            )
        except MemoryError:
            st.error("Không đủ bộ nhớ để nạp phantom. Hãy giảm kích thước ảnh đầu vào.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))

    _show_preview()

    st.header("Cấu hình CT và tái tạo")
    with st.form("simulation_controls"):
        method_label = st.selectbox(
            "Phương pháp tái tạo", options=list(METHODS), key="selected_method"
        )
        method = METHODS[method_label]
        columns = st.columns(3)
        num_angles = columns[0].slider(
            "Số góc chiếu (Nθ)", min_value=10, max_value=360, value=180
        )
        noise_percent = columns[1].slider(
            "Nhiễu Gaussian (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
        )
        filter_name = "hann"
        if method in ("parallel", "fan"):
            filter_name = columns[2].selectbox(
                "Bộ lọc FBP",
                options=["ramp", "hann", "hamming", None],
                format_func=lambda value: FILTER_LABELS[value],
            )
        else:
            columns[2].caption("SART dùng dữ liệu chiếu parallel-beam.")
        num_iterations = 10
        relaxation = 0.5
        if method == "sart":
            iteration_columns = st.columns(2)
            num_iterations = iteration_columns[0].slider(
                "Số vòng lặp SART", min_value=1, max_value=50, value=10
            )
            relaxation = iteration_columns[1].slider(
                "Hệ số thư giãn λ",
                min_value=0.1,
                max_value=1.5,
                value=0.5,
                step=0.1,
            )
        run_column, compare_column = st.columns(2)
        run_reconstruction = run_column.form_submit_button(
            "Run Reconstruction", type="primary", width="stretch"
        )
        compare_methods = compare_column.form_submit_button(
            "Compare Methods", width="stretch"
        )

    if run_reconstruction or compare_methods:
        methods = list(METHODS.values()) if compare_methods else [method]
        computed = {}
        st.session_state["phantom_results"] = {}
        with st.spinner("Đang tạo dữ liệu chiếu và tái tạo ảnh..."):
            try:
                for selected_method in methods:
                    computed[selected_method] = _simulate(
                        phantom=st.session_state["phantom"],
                        phantom_name=st.session_state["phantom_name"],
                        method=selected_method,
                        num_angles=num_angles,
                        noise_percent=noise_percent,
                        filter_name=filter_name,
                        num_iterations=num_iterations,
                        relaxation=relaxation,
                    )
            except MemoryError:
                st.error(
                    "Không đủ bộ nhớ để hoàn tất mô phỏng. Hãy giảm kích thước phantom, "
                    "số góc hoặc số vòng lặp."
                )
            except (ValueError, RuntimeError) as exc:
                st.error(f"Mô phỏng thất bại: {exc}")
            else:
                st.session_state["phantom_results"] = computed
                if compare_methods:
                    st.success("Đã chạy cả ba phương pháp trên cùng phantom đầu vào.")

    st.header("Kết quả và xuất dữ liệu")
    _display_results()


if __name__ == "__main__":
    main()
