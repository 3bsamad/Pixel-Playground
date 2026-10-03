# Pixel Playground

**Pixel Playground** is a lightweight Python toolkit for preparing computer-vision images and datasets. The goal is simple: make common preprocessing jobs reliable, scriptable, and pleasant to use from both Python and the command line.

> **Status:** early alpha (`0.1.0`). The first release establishes the package, CLI, tests, and resizing API. Dataset auditing, validation, tiling, and annotation-aware geometry are next on the roadmap.

## Why Pixel Playground?

Computer-vision projects repeatedly need the same small utilities: resize a directory, preserve its structure, validate inputs, tile large images, transform annotations, inspect a dataset, or convert formats. These tasks are easy to prototype and surprisingly easy to get subtly wrong.

Pixel Playground aims to provide focused building blocks rather than replace OpenCV, Pillow, Albumentations, or full dataset-management platforms.

## Current features

- NumPy image resizing with Pillow or OpenCV
- Automatic interpolation selection for shrinking vs. enlarging
- Batch directory resizing
- Recursive directory processing with relative-path preservation
- Multiple extension filters
- Existing-file policies (`overwrite`, `skip`, `error`)
- Optional continue-on-error behavior
- Python API and command-line interface
- OpenCV is optional; Pillow is the lightweight default fallback

## Development installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/3bsamad/Pixel-Playground.git
cd Pixel-Playground
python -m pip install -e .
```

For the optional OpenCV backend:

```bash
python -m pip install -e ".[opencv]"
```

For development:

```bash
python -m pip install -e ".[dev,opencv]"
```

## Python API

`size` is always expressed as `(width, height)`.

```python
import numpy as np
from pixel_playground import resize

image = np.zeros((1080, 1920, 3), dtype=np.uint8)
resized = resize(image, (640, 640))

print(resized.shape)
# (640, 640, 3)
```

Choose a backend or interpolation method explicitly when needed:

```python
resized = resize(
    image,
    (640, 640),
    backend="opencv",
    interpolation="area",
)
```

Batch process a directory:

```python
from pixel_playground import resize_directory

result = resize_directory(
    "dataset/images",
    "dataset/resized",
    (640, 640),
    recursive=True,
    extensions=("jpg", "jpeg", "png"),
)

print(result.processed, result.skipped, result.failed_count)
```

## Command line

```bash
pixel-playground resize dataset/images dataset/resized \
  --size 640 640 \
  --recursive
```

Filter by one or more extensions:

```bash
pixel-playground resize images resized \
  --size 256 256 \
  --extension jpg \
  --extension png
```

Useful options:

```text
--backend {auto,opencv,pillow}
--interpolation {auto,nearest,linear,cubic,lanczos,area}
--recursive
--on-existing {overwrite,skip,error}
--continue-on-error
--no-progress
```

Run `pixel-playground resize --help` for the complete command reference.

## Roadmap

### 0.2 — image geometry

- aspect-ratio-aware `fit`, `cover`, `pad`, and letterbox modes
- crop and padding helpers
- transformation metadata for mapping coordinates back to the source image

### 0.3 — dataset utilities

- dataset audit and statistics
- corrupted-file and image-shape validation
- exact and perceptual duplicate detection
- deterministic train/validation/test splitting

### 0.4 — annotations

- bounding-box and mask primitives
- YOLO, COCO, and Pascal VOC readers/writers
- annotation-safe resize, crop, and conversion

### 0.5 — large-image workflows

- overlapping image tiling
- annotation-aware tile clipping
- inverse transforms for mapping model predictions to original coordinates

The longer-term goal is a small, composable toolkit for image + annotation geometry without requiring a database, server, or heavyweight dataset platform.

## Contributing

The project is in an early stage, so API feedback and focused feature proposals are especially useful. Please open an issue before large changes so the scope stays coherent.

## License

MIT
