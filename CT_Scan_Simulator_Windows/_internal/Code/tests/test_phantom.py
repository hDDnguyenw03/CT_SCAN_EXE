import io
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_gui import METHODS, _simulate
from evaluation import evaluate_reconstruction
from export_results import create_results_zip
from phantom import (
    create_builtin_phantom,
    create_custom_phantom,
    decode_uploaded_image,
    prepare_phantom,
)


def _encoded_image(array, image_format):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    return buffer.getvalue()


class PhantomTests(unittest.TestCase):
    def test_all_builtin_phantoms_are_finite_square_images(self):
        names = (
            "Shepp–Logan phantom",
            "Modified Shepp–Logan phantom",
            "Phantom hình học (tròn và elip)",
            "Lát cắt với các vùng suy giảm tương đối",
        )
        images = [create_builtin_phantom(name, 128) for name in names]
        for image in images:
            self.assertEqual(image.shape, (128, 128))
            self.assertTrue(np.isfinite(image).all())
        self.assertFalse(np.array_equal(images[0], images[1]))

    def test_png_grayscale_and_jpeg_rgb_uploads(self):
        gray = np.arange(24 * 24, dtype=np.uint8).reshape(24, 24)
        decoded_gray, gray_info = decode_uploaded_image(_encoded_image(gray, "PNG"))
        processed_gray, _ = prepare_phantom(decoded_gray, 32)
        self.assertEqual(gray_info["channels"], 1)
        self.assertEqual(processed_gray.shape, (32, 32))

        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        rgb[..., 0] = np.tile(np.arange(24, dtype=np.uint8) * 11, (24, 1))
        decoded_rgb, rgb_info = decode_uploaded_image(_encoded_image(rgb, "JPEG"))
        processed_rgb, _ = prepare_phantom(decoded_rgb, 32)
        self.assertEqual(rgb_info["channels"], 3)
        self.assertTrue(np.isfinite(processed_rgb).all())
        self.assertGreater(float(processed_rgb.max()), 0.9)

    def test_rgba_transparency_is_composited_over_black(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[3:5, 3:5, 3] = 255
        processed, _ = prepare_phantom(rgba, 32)
        self.assertEqual(processed.shape, (32, 32))
        self.assertTrue(np.isfinite(processed).all())
        self.assertEqual(float(processed.max()), 1.0)

    def test_non_square_image_preserves_aspect_ratio(self):
        image = np.ones((12, 24), dtype=np.float32)
        processed, _ = prepare_phantom(image, 32, normalize=False)
        rows, columns = np.where(processed > 0.5)
        self.assertAlmostEqual((columns.max() - columns.min() + 1) / (rows.max() - rows.min() + 1), 2.0)

    def test_negative_tiff_values_and_non_finite_replacement(self):
        source = np.array([[-10, 0], [5, 20]], dtype=np.int16)
        decoded, metadata = decode_uploaded_image(_encoded_image(source, "TIFF"))
        processed, warnings = prepare_phantom(decoded, 32, normalize=False)
        self.assertLess(metadata["pixel_min"], 0)
        self.assertLess(float(processed.min()), 0)
        self.assertTrue(any("âm" in warning for warning in warnings))

        image = np.array([[0.0, np.nan], [np.inf, 1.0]])
        finite, warnings = prepare_phantom(image, 32, normalize=False)
        self.assertTrue(np.isfinite(finite).all())
        self.assertTrue(any("NaN/Inf" in warning for warning in warnings))

    def test_corrupt_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Không thể đọc"):
            decode_uploaded_image(b"not an image")
        with self.assertRaisesRegex(ValueError, "rỗng"):
            decode_uploaded_image(b"")

    def test_custom_circle_ellipse_and_rectangle(self):
        structures = [
            {"shape": "Circle", "center_x": -0.3, "center_y": 0.0, "radius_x": 0.2, "radius_y": 0.2, "value": 0.6},
            {"shape": "Ellipse", "center_x": 0.25, "center_y": 0.0, "radius_x": 0.32, "radius_y": 0.12, "angle": 35, "value": 0.8},
            {"shape": "Rectangle", "center_x": 0.0, "center_y": -0.5, "radius_x": 0.2, "radius_y": 0.1, "angle": 15, "value": 1.0},
        ]
        image = create_custom_phantom(128, structures)
        self.assertEqual(image.shape, (128, 128))
        self.assertEqual(float(image.max()), 1.0)
        self.assertGreater(np.count_nonzero(image), 0)
        with self.assertRaisesRegex(ValueError, "ít nhất"):
            create_custom_phantom(128, [])

    def test_switching_builtin_phantoms_does_not_reuse_old_data(self):
        first = create_builtin_phantom("Shepp–Logan phantom", 128)
        second = create_builtin_phantom("Phantom hình học (tròn và elip)", 128)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.shape, second.shape)

    def test_all_methods_use_the_same_phantom(self):
        phantom = create_builtin_phantom("Phantom hình học (tròn và elip)", 32)
        results = {
            method: _simulate(
                phantom=phantom,
                phantom_name="unit-test",
                method=method,
                num_angles=10,
                noise_percent=0.0,
                filter_name="hann",
                num_iterations=2,
                relaxation=0.5,
            )
            for method in METHODS.values()
        }
        self.assertEqual(set(results), {"parallel", "fan", "sart"})
        for result in results.values():
            np.testing.assert_array_equal(result["phantom"], phantom)
            self.assertEqual(result["reconstruction"].shape, phantom.shape)
            self.assertTrue(np.isfinite(result["reconstruction"]).all())
            self.assertIn("RMSE", result["metrics"])

    def test_zero_reference_relative_error_is_finite(self):
        metrics = evaluate_reconstruction(np.zeros((32, 32)), np.zeros((32, 32)))
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertEqual(metrics["RelError"], 0.0)

    def test_export_archive_can_be_reopened(self):
        phantom = create_builtin_phantom("Shepp–Logan phantom", 32)
        result = {
            "parallel": {
                "method": "parallel",
                "method_label": "Parallel-beam FBP",
                "sinogram": np.ones((32, 10)),
                "reconstruction": phantom.copy(),
                "metrics": {"RMSE": 0.0, "RelError": 0.0, "PSNR_dB": 120.0, "SSIM": 1.0},
            }
        }
        archive_bytes = create_results_zip(
            "unit-test",
            phantom,
            result,
            {"phantom_name": "unit-test"},
            b"comparison-png",
        )
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertIn("metrics.csv", archive.namelist())
            self.assertIn(
                "parallel/parallel_unit-test_sinogram.npy", archive.namelist()
            )
            restored = np.load(
                io.BytesIO(archive.read("phantom/input_unit-test_phantom.npy"))
            )
            np.testing.assert_array_equal(restored, phantom)


if __name__ == "__main__":
    unittest.main()
