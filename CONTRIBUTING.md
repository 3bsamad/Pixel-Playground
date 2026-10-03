# Contributing

Thanks for considering a contribution to Pixel Playground.

## Development setup

```bash
python -m pip install -e ".[dev,opencv]"
pytest
ruff check .
```

## Scope

Pixel Playground aims to stay lightweight and focused on practical computer-vision image and dataset tooling. For substantial features or API changes, please open an issue first so the design can be discussed before implementation.

## Pull requests

- Keep changes focused.
- Add or update tests for behavior changes.
- Keep public APIs typed and documented.
- Run the test suite and linter before submitting.
- Avoid adding heavy mandatory dependencies when an optional extra is sufficient.
