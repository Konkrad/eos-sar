# Contributing

## Project structure

- [`src/eos/sar/`](src/eos/sar/) — sensor-agnostic SAR processing: the sensor model
  (`model.py`), orbit handling (`orbit.py`), coordinates and ROIs (`coordinates.py`,
  `roi.py`), registration and resampling (`regist.py`), deburst/mosaicking,
  radiometric terrain correction (`rtc.py`), orthorectification (`ortho.py`),
  interferometry (`coherence.py`, `goldstein_filter.py`, `unwrapping.py`) and the
  Cython-accelerated raster simulator (`simulator.pyx`).
- [`src/eos/products/`](src/eos/products/) — one subpackage per sensor
  (`sentinel1/`, `cosmoskymed/`, `terrasarx/`, `capella/`, `nisar/`, `snap/`), each
  adapting product metadata parsing, calibration and image readers to the generic
  `eos.sar` APIs.
- [`src/eos/dem.py`](src/eos/dem.py) and [`src/eos/cache.py`](src/eos/cache.py) — DEM
  sourcing (`DEMStitcherSource`, `srtm4`) and caching utilities.
- [`src/teosar/`](src/teosar/) — Persistent Scatterer Interferometry and time-series
  tooling, installed via the `teosar`/`teosar-light` extras.
- [`usage/`](usage/) — example scripts and the tutorial notebook (see the
  [README](README.md#usage)).
- [`tests/`](tests/) — the test suite, including sample product data under
  `tests/data/`.
- [`docs/`](docs/) — additional documentation.

## Tests

To run the tests, we use `pytest`:

    uv run --all-extras pytest -n auto -v -m "not cdse" .
    uv run --env-file .env --all-extras pytest -v -m "cdse" .

Ideally, you would put your CDSE credentials in the .env file (see the
[README](README.md#usage)), so that the tests that read data from CDSE can run. Otherwise, the tests will be skipped. Note that the tests reading from CDSE are run separately in the commands above, on a single worker, to avoid issues related to rate limiting. Also, those tests are marked as "flaky", i.e., they are retried if/when they fail (due to rate limiting).

## Setting up a development environment

To install the package in editable mode, you can run:

    uv sync

or

    pip install -e . --group dev

## Code formatting

The CI validates the code against pep8 rules and formatting, as configured in `pyproject.toml`.

You can check your code locally before commiting using pre-commit or using:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Avoid making commits that only format the code; instead, amend commits or rebase the changes against the relevant commit.

You can also use the pre-commit.

```
source .venv/bin/activate # the .venv needs to be activated
uvx pre-commit install # you can do this once
git add file.py
git commit -m "message here" # pre-commit runs, might fail, no commit
# In case the pre-commit failed because of formatting
# --> retry
# In case the pre-commit failed because of typing (mypy)
# --> fix problems then retry
git add file.py
git commit -m "message here" # should work now
```

## Making a release

1. generate the changelog: `uv run --no-project --with git-cliff git cliff --unreleased` and update `CHANGELOG.md` manually
2. update the version in `pyproject.toml` (try to respect semantic versioning)
3. run `uv lock` to update uv.lock
4. commit (message="x.y.z") and tag the commit (tag="x.y.z")
5. push with the tag (`git push --tags`)

## Tips for external contributors

Make sure to have pyproj data: `pyproj sync -v --file us_nga_egm96_15`
