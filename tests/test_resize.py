from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_playground import resize, resize_directory


def test_resize_pillow_uses_width_height_order() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    resized = resize(image, (7, 5), backend="pillow")

    assert resized.shape == (5, 7, 3)


def test_resize_opencv_uses_width_height_order() -> None:
    pytest.importorskip("cv2")
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    resized = resize(image, (7, 5), backend="opencv")

    assert resized.shape == (5, 7, 3)


@pytest.mark.parametrize(
    "size",
    [
        (0, 10),
        (10, 0),
        (-1, 10),
        (10, -1),
    ],
)
def test_resize_rejects_non_positive_sizes(size: tuple[int, int]) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive integers"):
        resize(image, size)


def test_resize_directory_filters_extensions(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (12, 8)).save(input_dir / "keep.png")
    Image.new("RGB", (12, 8)).save(input_dir / "ignore.jpg")

    result = resize_directory(
        input_dir,
        output_dir,
        (6, 4),
        backend="pillow",
        extensions="png",
        show_progress=False,
    )

    assert result.discovered == 1
    assert result.processed == 1
    assert (output_dir / "keep.png").exists()
    assert not (output_dir / "ignore.jpg").exists()


def test_resize_directory_empty_is_valid(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    result = resize_directory(
        input_dir,
        tmp_path / "output",
        (6, 4),
        backend="pillow",
        show_progress=False,
    )

    assert result.discovered == 0
    assert result.processed == 0
    assert result.failed_count == 0


def test_resize_directory_preserves_relative_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    nested = input_dir / "camera_01"
    nested.mkdir(parents=True)
    Image.new("RGB", (12, 8)).save(nested / "frame.png")

    output_dir = tmp_path / "output"
    resize_directory(
        input_dir,
        output_dir,
        (6, 4),
        backend="pillow",
        recursive=True,
        show_progress=False,
    )

    assert (output_dir / "camera_01" / "frame.png").exists()
