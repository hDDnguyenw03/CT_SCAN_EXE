"""Portable ZIP export for a dashboard reconstruction session."""

import csv
import io
import json
import zipfile

import numpy as np


def create_results_zip(
    phantom_name: str,
    phantom: np.ndarray,
    results: dict[str, dict],
    config: dict,
    comparison_png: bytes,
) -> bytes:
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in phantom_name
    ).strip("_") or "phantom"
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"phantom/input_{safe_name}_phantom.npy", _npy_bytes(phantom)
        )
        archive.writestr(
            "configuration.json",
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False),
        )
        metrics_buffer = io.StringIO(newline="")
        metric_columns = ["method", "RMSE", "RelError", "PSNR_dB", "SSIM"]
        writer = csv.DictWriter(metrics_buffer, fieldnames=metric_columns)
        writer.writeheader()
        for result in results.values():
            row = {"method": result["method_label"], **result["metrics"]}
            writer.writerow({key: row.get(key, "") for key in metric_columns})
        archive.writestr("metrics.csv", metrics_buffer.getvalue())

        for result in results.values():
            method = result["method"]
            archive.writestr(
                f"{method}/{method}_{safe_name}_sinogram.npy",
                _npy_bytes(result["sinogram"]),
            )
            archive.writestr(
                f"{method}/{method}_{safe_name}_reconstruction.npy",
                _npy_bytes(result["reconstruction"]),
            )
        archive.writestr(f"comparison_{safe_name}.png", comparison_png)
    return archive_buffer.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float64), allow_pickle=False)
    return buffer.getvalue()
