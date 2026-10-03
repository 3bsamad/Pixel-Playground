"""Image resizing primitives and batch helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from tqdm import tqdm

Backend = Literal["auto", "opencv", "pillow"]
Interpolation = Literal["auto", "nearest", "linear", "cubic", "lanczos", "area"]
OnExisting = Literal["overwrite", "skip", "error"]

try:
    import cv2
except ImportError:  # pragma: no cover - covered indirectly when OpenCV is absent
    cv2 = None  # type: ignore[assignment]


@dataclass(slots=True)
class BatchResizeResult:
    """Summary returned by :func:`resize_directory`."""

    discovered: int = 0
    processed: int = 0
    skipped: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def failed_count(self) -> int:
        return len(self.failed)


def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("size must be a (width, height) tuple")

    width, height = size
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("width and height must be positive integers")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    return width, height


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")


def _resolve_backend(backend: Backend) -> Literal["opencv", "pillow"]:
    if backend == "auto":
        return "opencv" if cv2 is not None else "pillow"
    if backend == "opencv":
        if cv2 is None:
            raise ImportError(
                "OpenCV backend requested but OpenCV is not installed. "
                "Install pixel-playground[opencv] or choose backend='pillow'."
            )
        return "opencv"
    if backend == "pillow":
        return "pillow"
    raise ValueError("backend must be one of: auto, opencv, pillow")


def _opencv_interpolation(
    interpolation: Interpolation,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> int:
    assert cv2 is not None
    if interpolation == "auto":
        shrinking = target_size[0] <= source_size[0] and target_size[1] <= source_size[1]
        return cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR

    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    try:
        return mapping[interpolation]
    except KeyError as exc:
        raise ValueError(
            "interpolation must be one of: auto, nearest, linear, cubic, lanczos, area"
        ) from exc


def _pillow_interpolation(
    interpolation: Interpolation,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> Image.Resampling:
    if interpolation == "auto":
        shrinking = target_size[0] <= source_size[0] and target_size[1] <= source_size[1]
        return Image.Resampling.LANCZOS if shrinking else Image.Resampling.BILINEAR

    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "linear": Image.Resampling.BILINEAR,
        "cubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "area": Image.Resampling.BOX,
    }
    try:
        return mapping[interpolation]
    except KeyError as exc:
        raise ValueError(
            "interpolation must be one of: auto, nearest, linear, cubic, lanczos, area"
        ) from exc


def resize(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    backend: Backend = "auto",
    interpolation: Interpolation = "auto",
) -> np.ndarray:
    """Resize a NumPy image to ``size=(width, height)``.

    ``backend='auto'`` uses OpenCV when it is installed and otherwise falls back to Pillow.
    ``interpolation='auto'`` favors area/Lanczos-style filtering when shrinking and a linear
    filter when enlarging.
    """

    _validate_image(image)
    target_size = _validate_size(size)
    source_size = (image.shape[1], image.shape[0])
    resolved_backend = _resolve_backend(backend)

    if resolved_backend == "opencv":
        assert cv2 is not None
        method = _opencv_interpolation(interpolation, source_size, target_size)
        return cv2.resize(image, target_size, interpolation=method)

    method = _pillow_interpolation(interpolation, source_size, target_size)
    try:
        pillow_image = Image.fromarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Pillow could not represent this NumPy array. Try backend='opencv' for this dtype."
        ) from exc
    return np.asarray(pillow_image.resize(target_size, resample=method))


def resize_file(
    input_path: str | Path,
    output_path: str | Path,
    size: tuple[int, int],
    *,
    backend: Backend = "auto",
    interpolation: Interpolation = "auto",
) -> Path:
    """Resize one image file and write it to ``output_path``."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    _validate_size(size)

    with Image.open(input_path) as source:
        image = np.asarray(source)
        resized = resize(image, size, backend=backend, interpolation=interpolation)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(resized).save(output_path)

    return output_path


def _normalize_extensions(extensions: str | Sequence[str]) -> set[str]:
    if isinstance(extensions, str):
        extensions = [extensions]

    normalized = {ext.lower().lstrip(".") for ext in extensions if ext.strip()}
    if not normalized:
        raise ValueError("at least one image extension must be provided")
    return normalized


def resize_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    size: tuple[int, int],
    *,
    backend: Backend = "auto",
    interpolation: Interpolation = "auto",
    extensions: str | Sequence[str] = ("jpg", "jpeg", "png"),
    recursive: bool = False,
    on_existing: OnExisting = "overwrite",
    continue_on_error: bool = False,
    show_progress: bool = True,
) -> BatchResizeResult:
    """Resize matching images in a directory while preserving relative paths."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _validate_size(size)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {input_dir}")
    if on_existing not in {"overwrite", "skip", "error"}:
        raise ValueError("on_existing must be one of: overwrite, skip, error")

    normalized_extensions = _normalize_extensions(extensions)
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower().lstrip(".") in normalized_extensions
    )

    result = BatchResizeResult(discovered=len(paths))
    progress = tqdm(paths, desc="Resizing", unit="image", disable=not show_progress)

    for input_path in progress:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        if output_path.exists():
            if on_existing == "skip":
                result.skipped += 1
                continue
            if on_existing == "error":
                raise FileExistsError(f"output file already exists: {output_path}")

        try:
            resize_file(
                input_path,
                output_path,
                size,
                backend=backend,
                interpolation=interpolation,
            )
            result.processed += 1
        except Exception as exc:
            if not continue_on_error:
                raise
            result.failed.append((input_path, str(exc)))

    return result
