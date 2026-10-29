"""Phantom creation, image loading, and shared preprocessing for the dashboard."""

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.transform import resize

MAX_INPUT_PIXELS = 4_000_000
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


_ELLIPSES = {
    "Shepp–Logan phantom": (
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),
        (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
        (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
        (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
        (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
        (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
    ),
    "Modified Shepp–Logan phantom": (
        (2.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.98, 0.6624, 0.874, 0.0, -0.0184, 0.0),
        (-0.02, 0.11, 0.31, 0.22, 0.0, -18.0),
        (-0.02, 0.16, 0.41, -0.22, 0.0, 18.0),
        (0.01, 0.21, 0.25, 0.0, 0.35, 0.0),
        (0.01, 0.046, 0.046, 0.0, 0.1, 0.0),
        (0.01, 0.046, 0.046, 0.0, -0.1, 0.0),
        (0.01, 0.046, 0.023, -0.08, -0.605, 0.0),
        (0.01, 0.023, 0.023, 0.0, -0.606, 0.0),
        (0.01, 0.023, 0.046, 0.06, -0.605, 0.0),
    ),
}


def _validate_size(size: int) -> int:
    if not isinstance(size, (int, np.integer)) or not 32 <= size <= 512:
        raise ValueError("Kích thước phantom phải nằm trong khoảng 32–512 pixel.")
    return int(size)


def _ellipse_phantom(size: int, ellipses: tuple[tuple, ...]) -> np.ndarray:
    axis = (np.arange(size) - (size - 1) / 2.0) / (size / 2.0)
    x, y = np.meshgrid(axis, -axis)
    image = np.zeros((size, size), dtype=np.float64)
    for value, radius_x, radius_y, center_x, center_y, angle in ellipses:
        angle_rad = np.deg2rad(angle)
        dx = x - center_x
        dy = y - center_y
        rotated_x = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
        rotated_y = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        mask = (rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2 <= 1.0
        image[mask] += value

    low, high = float(image.min()), float(image.max())
    if high > low:
        image = (image - low) / (high - low)
    return image


def create_custom_phantom(size: int, structures: list[dict]) -> np.ndarray:
    """Render circles, ellipses, and rectangles using normalized image coordinates."""
    size = _validate_size(size)
    if not structures:
        raise ValueError("Thêm ít nhất một cấu trúc để tạo phantom tự thiết kế.")
    axis = (np.arange(size) - (size - 1) / 2.0) / (size / 2.0)
    x, y = np.meshgrid(axis, -axis)
    image = np.zeros((size, size), dtype=np.float64)

    for item in structures:
        shape = item["shape"]
        center_x = float(item["center_x"])
        center_y = float(item["center_y"])
        radius_x = float(item["radius_x"])
        radius_y = float(item["radius_y"])
        value = float(item["value"])
        angle = float(item.get("angle", 0.0))
        if shape not in {"Circle", "Ellipse", "Rectangle"}:
            raise ValueError(f"Hình dạng không được hỗ trợ: {shape}")
        if not all(np.isfinite(v) for v in (center_x, center_y, radius_x, radius_y, value, angle)):
            raise ValueError("Thông số cấu trúc phải là các số hữu hạn.")
        if radius_x <= 0.0 or radius_y <= 0.0 or not 0.0 <= value <= 1.0:
            raise ValueError("Bán trục phải dương và cường độ phải thuộc [0, 1].")
        if shape == "Circle" and not np.isclose(radius_x, radius_y):
            raise ValueError("Hình tròn cần hai bán kính bằng nhau.")

        angle_rad = np.deg2rad(angle)
        dx = x - center_x
        dy = y - center_y
        rotated_x = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
        rotated_y = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        if shape == "Rectangle":
            mask = (np.abs(rotated_x) <= radius_x) & (np.abs(rotated_y) <= radius_y)
        else:
            mask = (rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2 <= 1.0
        image[mask] = value
    return image


def create_builtin_phantom(name: str, size: int) -> np.ndarray:
    size = _validate_size(size)
    if name in _ELLIPSES:
        return _ellipse_phantom(size, _ELLIPSES[name])

    axis = (np.arange(size) - (size - 1) / 2.0) / (size / 2.0)
    x, y = np.meshgrid(axis, -axis)
    if name == "Phantom hình học (tròn và elip)":
        structures = [
            {"shape": "Ellipse", "center_x": 0.0, "center_y": 0.0, "radius_x": 0.72, "radius_y": 0.86, "angle": 0.0, "value": 0.25},
            {"shape": "Circle", "center_x": -0.24, "center_y": 0.12, "radius_x": 0.2, "radius_y": 0.2, "angle": 0.0, "value": 0.8},
            {"shape": "Ellipse", "center_x": 0.28, "center_y": -0.18, "radius_x": 0.28, "radius_y": 0.12, "angle": -25.0, "value": 1.0},
            {"shape": "Circle", "center_x": 0.38, "center_y": 0.35, "radius_x": 0.1, "radius_y": 0.1, "angle": 0.0, "value": 0.55},
        ]
        return create_custom_phantom(size, structures)
    if name == "Lát cắt với các vùng suy giảm tương đối":
        image = np.zeros((size, size), dtype=np.float64)
        image[(x / 0.82) ** 2 + (y / 0.82) ** 2 <= 1.0] = 0.18
        image[(x / 0.58) ** 2 + (y / 0.58) ** 2 <= 1.0] = 0.35
        image[((x + 0.22) / 0.19) ** 2 + ((y - 0.12) / 0.19) ** 2 <= 1.0] = 0.72
        image[((x - 0.24) / 0.17) ** 2 + ((y + 0.2) / 0.12) ** 2 <= 1.0] = 1.0
        return image
    raise ValueError(f"Phantom dựng sẵn không được hỗ trợ: {name}")


def decode_uploaded_image(file_bytes: bytes) -> tuple[np.ndarray, dict]:
    if not file_bytes:
        raise ValueError("Tệp ảnh rỗng.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Tệp ảnh vượt quá giới hạn 32 MB để bảo vệ bộ nhớ.")
    try:
        with Image.open(BytesIO(file_bytes)) as source:
            if source.width * source.height > MAX_INPUT_PIXELS:
                raise ValueError(
                    "Ảnh vượt quá giới hạn 4 triệu pixel để bảo vệ bộ nhớ. "
                    "Hãy thu nhỏ ảnh trước khi tải lên."
                )
            image_format = source.format
            source.load()
            if source.mode == "P":
                source = source.convert("RGBA" if "transparency" in source.info else "RGB")
            image = np.asarray(source)
            metadata = {
                "format": image_format or "Không xác định",
                "mode": source.mode,
                "width": int(source.width),
                "height": int(source.height),
                "dtype": str(image.dtype),
            }
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Không thể đọc tệp ảnh. Hãy tải lên PNG, JPG/JPEG hoặc TIFF hợp lệ.") from exc

    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Ảnh phải có dữ liệu 2D xám hoặc 3D màu.")
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    if channels not in (1, 2, 3, 4):
        raise ValueError(f"Số kênh ảnh không được hỗ trợ: {channels}.")
    values = image.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Ảnh không chứa pixel hữu hạn.")
    metadata.update(
        {
            "channels": 1 if channels in (1, 2) else (3 if channels == 3 else 4),
            "pixel_min": float(np.min(values[finite])),
            "pixel_max": float(np.max(values[finite])),
            "non_finite_count": int(values.size - np.count_nonzero(finite)),
        }
    )
    return image, metadata


def prepare_phantom(
    image: np.ndarray, size: int, normalize: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Convert input data to a finite, square float64 image without distortion."""
    size = _validate_size(size)
    values = np.asarray(image)
    if values.size == 0 or values.ndim not in (2, 3):
        raise ValueError("Phantom phải là ảnh 2D xám hoặc ảnh màu không rỗng.")
    if (
        not np.issubdtype(values.dtype, np.number)
        and not np.issubdtype(values.dtype, np.bool_)
    ) or np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError("Pixel ảnh phải có kiểu dữ liệu số thực hoặc số nguyên.")
    if values.shape[0] * values.shape[1] > MAX_INPUT_PIXELS:
        raise ValueError("Ảnh vượt quá giới hạn 4 triệu pixel để bảo vệ bộ nhớ.")
    if values.ndim == 3:
        if values.shape[2] not in (2, 3, 4):
            raise ValueError("Ảnh màu phải có 2, 3 hoặc 4 kênh.")
        original = np.asarray(values)
        values = original.astype(np.float64, copy=False)
        if values.shape[2] in (3, 4):
            rgb = values[..., :3]
            values = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            alpha_channel = 3 if original.shape[2] == 4 else None
        else:
            values = values[..., 0]
            alpha_channel = 1
        if alpha_channel is not None:
            alpha = original[..., alpha_channel].astype(np.float64, copy=False)
            alpha_finite = alpha[np.isfinite(alpha)]
            alpha_max = (
                float(np.iinfo(original.dtype).max)
                if np.issubdtype(original.dtype, np.integer)
                else (float(alpha_finite.max()) if alpha_finite.size else 1.0)
            )
            alpha = np.nan_to_num(alpha / max(alpha_max, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
            if original.shape[2] == 2:
                values = original[..., 0].astype(np.float64, copy=False) * alpha
            else:
                values = values * np.clip(alpha, 0.0, 1.0)
    else:
        values = values.astype(np.float64, copy=False)

    warnings = []
    finite = np.isfinite(values)
    if not finite.all():
        warnings.append(f"Thay thế {int(values.size - finite.sum())} pixel NaN/Inf bằng 0.")
        values = np.nan_to_num(values, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Ảnh đầu vào không có chiều cao hoặc chiều rộng.")

    height, width = values.shape
    scale = min(size / height, size / width)
    new_height = max(1, min(size, int(round(height * scale))))
    new_width = max(1, min(size, int(round(width * scale))))
    if (new_height, new_width) != values.shape:
        values = resize(
            values,
            (new_height, new_width),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
    canvas = np.zeros((size, size), dtype=np.float64)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = values

    axis = np.arange(size)
    x, y = np.meshgrid(axis, axis)
    circle_mask = (x - size // 2) ** 2 + (y - size // 2) ** 2 <= (size // 2) ** 2
    if np.any(np.abs(canvas[~circle_mask]) > 1e-12):
        warnings.append(
            "Đặt pixel ngoài trường nhìn tròn về 0 để tương thích hình học Radon/SART."
        )
    canvas[~circle_mask] = 0.0

    low = float(canvas.min())
    high = float(canvas.max())
    if normalize:
        if high > low:
            canvas = (canvas - low) / (high - low)
        else:
            canvas.fill(0.0)
        warnings.append("Chuẩn hóa min–max toàn ảnh về [0, 1]; giá trị cường độ gốc bị thay đổi.")
    else:
        warnings.append("Giữ miền giá trị pixel; ảnh xám không được xem là hệ số suy giảm tuyến tính.")
        if low < 0:
            warnings.append("Ảnh có giá trị âm; mô hình CT tương đối này không áp đặt ý nghĩa vật lý cho chúng.")

    if not np.isfinite(canvas).all():
        raise ValueError("Tiền xử lý tạo ra giá trị không hữu hạn.")
    return canvas.astype(np.float64, copy=False), warnings


def image_preview_rgb(image: np.ndarray) -> np.ndarray:
    """Create a displayable RGB preview for grayscale or RGB(A) source data."""
    values = np.asarray(image)
    if values.ndim == 2:
        gray = values.astype(np.float64)
        finite = gray[np.isfinite(gray)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        gray = (gray - low) / (high - low) if high > low else np.zeros_like(gray)
        return np.repeat(gray[..., None], 3, axis=2)
    if values.ndim == 3 and values.shape[2] >= 3:
        rgb = values[..., :3].astype(np.float64)
        finite_values = rgb[np.isfinite(rgb)]
        low, high = (float(finite_values.min()), float(finite_values.max())) if finite_values.size else (0.0, 1.0)
        rgb = (rgb - low) / (high - low) if high > low else np.zeros_like(rgb)
        return np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return image_preview_rgb(values[..., 0])
