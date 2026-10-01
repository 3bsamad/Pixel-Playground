"""Pixel Playground: lightweight utilities for computer-vision datasets."""

from pixel_playground.image.resize import (
    Backend,
    BatchResizeResult,
    Interpolation,
    OnExisting,
    resize,
    resize_directory,
    resize_file,
)

__all__ = [
    "Backend",
    "BatchResizeResult",
    "Interpolation",
    "OnExisting",
    "resize",
    "resize_directory",
    "resize_file",
]

__version__ = "0.1.0"
