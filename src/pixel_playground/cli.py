"""Command-line interface for Pixel Playground."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pixel_playground import __version__
from pixel_playground.image.resize import resize_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel-playground",
        description="Lightweight computer-vision image and dataset utilities.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    resize_parser = subparsers.add_parser("resize", help="resize images in a directory")
    resize_parser.add_argument("input", help="input directory")
    resize_parser.add_argument("output", help="output directory")
    resize_parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        required=True,
        help="target size as width height",
    )
    resize_parser.add_argument(
        "--backend",
        choices=("auto", "opencv", "pillow"),
        default="auto",
        help="resize backend (default: auto)",
    )
    resize_parser.add_argument(
        "--interpolation",
        choices=("auto", "nearest", "linear", "cubic", "lanczos", "area"),
        default="auto",
        help="interpolation method (default: auto)",
    )
    resize_parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        help="image extension to include; repeat for multiple (default: jpg, jpeg, png)",
    )
    resize_parser.add_argument(
        "--recursive",
        action="store_true",
        help="search subdirectories and preserve their relative paths",
    )
    resize_parser.add_argument(
        "--on-existing",
        choices=("overwrite", "skip", "error"),
        default="overwrite",
        help="what to do when an output file exists (default: overwrite)",
    )
    resize_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue processing if an image cannot be read or written",
    )
    resize_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the progress bar",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "resize":
        result = resize_directory(
            args.input,
            args.output,
            tuple(args.size),
            backend=args.backend,
            interpolation=args.interpolation,
            extensions=args.extensions or ("jpg", "jpeg", "png"),
            recursive=args.recursive,
            on_existing=args.on_existing,
            continue_on_error=args.continue_on_error,
            show_progress=not args.no_progress,
        )
        print(
            f"Done: {result.processed} processed, {result.skipped} skipped, "
            f"{result.failed_count} failed ({result.discovered} discovered)."
        )
        if result.failed:
            for path, reason in result.failed:
                print(f"FAILED {path}: {reason}")
            return 1
        return 0

    raise AssertionError(f"unhandled command: {args.command}")
