from pathlib import Path

from PIL import Image

from pixel_playground.cli import main


def test_cli_extension_argument_is_respected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (12, 8)).save(input_dir / "selected.png")
    Image.new("RGB", (12, 8)).save(input_dir / "not-selected.jpg")

    exit_code = main(
        [
            "resize",
            str(input_dir),
            str(output_dir),
            "--size",
            "6",
            "4",
            "--backend",
            "pillow",
            "--extension",
            "png",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "selected.png").exists()
    assert not (output_dir / "not-selected.jpg").exists()
