# eos-sar

`eos-sar` provides generic SAR (Synthetic Aperture Radar) processing algorithms: sensor
modelling, geolocation, registration/resampling, radiometric calibration, terrain
correction, orthorectification, interferometry and related tools, together with readers
for several sensors' products.

Currently supported sensors:

- Sentinel-1 SLC in IW mode
- For the stripmap acquisition mode, some support for:
  - Cosmo-SkyMed
  - TerraSAR-X
  - Capella
- NISAR (initial support, work in progress)

## Installation

```bash
pip install eos-sar
```

Some processing pipelines assume that a library providing access to a DEM (Digital
Elevation Model) source is available. The recommended source is
`eos.dem.DEMStitcherSource` (`pip install dem-stitcher`), which uses GLO30 and GLO90.
You can also use [srtm4](https://github.com/centreborelli/srtm4) as an SRTM source,
installed with its `crop` extra:

```bash
pip install "srtm4[crop]"
```

If you wish to use another DEM source, inherit from the `eos.dem.DEMSource` template and
provide functions for cropping/querying a DEM.

`eos-sar` also ships optional extras:

- `teosar-light` for the `teosar/tsinsar.py` module, which creates a time series of
  coregistered Sentinel-1 crops with flattened phases, ready for interferometry.
- `teosar`, which adds support for Persistent Scatterer Interferometry (e.g.
  `teosar/ferreti_2001.py`). This part of the codebase is more research-oriented and not
  as heavily tested. It depends on `pyopencl`, which
  [needs an OpenCL driver](https://documen.tician.de/pyopencl/misc.html#enabling-access-to-cpus-and-gpus-via-py-opencl)
  (for CPU or GPU) to run.

```bash
pip install "eos-sar[teosar-light]"
```

## Usage

A typical pipeline reads a product's metadata and pixels, builds a sensor model for
geolocation, then calibrates/corrects and optionally orthorectifies a crop. For example,
reading and calibrating a region of interest from a Sentinel-1 GRD product (abridged —
see [`usage/grd.py`](usage/grd.py) for the full, runnable script including orbit and
atmospheric-correction setup):

```python
import numpy as np
import eos.dem
import eos.products.sentinel1 as sentinel1
from eos.sar.roi import Roi

# `product` wraps access to a Sentinel-1 GRD SAFE product (local path, S3, ...)
# `orbit` and `corrector` are built from the product's state vectors, see usage/grd.py
xml = product.get_xml_annotation("vv")
meta = sentinel1.metadata.extract_grd_metadata(xml)
proj_model = sentinel1.proj_model.grd_model_from_meta(meta, orbit, corrector)

roi = Roi(origin_x, origin_y, width, height)
reader = product.get_image_reader("vv")
raster = eos.sar.io.read_window(reader, roi, get_complex=False, out_dtype=np.float32)
```

For a complete, runnable walkthrough, see the [`usage/`](usage/) folder:

- [`usage/tutorial.ipynb`](usage/tutorial.ipynb) walks through forming a Sentinel-1
  interferogram (and more) over an earthquake event, covering the sensor model,
  calibration, registration/resampling/debursting, line-of-sight computation,
  interferogram formation and orthorectification.
- [`usage/grd.py`](usage/grd.py), [`usage/csk.py`](usage/csk.py),
  [`usage/tsx.py`](usage/tsx.py), [`usage/capella.py`](usage/capella.py) and
  [`usage/s1-slc-crop-simulation.py`](usage/s1-slc-crop-simulation.py) are standalone
  scripts for each supported sensor/product type.

The tutorial and several scripts read data from [CDSE](https://dataspace.copernicus.eu/),
which requires a CDSE account and [AWS secrets](https://eodata-s3keysmanager.dataspace.copernicus.eu/).
Set them as environment variables, e.g. via a `.env` file:

```
CDSE_ACCESS_KEY_ID = <value>
CDSE_SECRET_ACCESS_KEY = <value>
CDSE_USERNAME = <value>
CDSE_PASSWORD = <value>
```

Then run, for instance:

```bash
uv run --env-file .env --with jupyter --with matplotlib jupyter lab   # usage/tutorial.ipynb
uv run --env-file .env usage/grd.py
```

See [`docs/`](docs/) for further notes, such as
[border masking for GRD products](docs/border_masking_grd.md).

## Contributing

### Project structure

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
- [`usage/`](usage/) — example scripts and the tutorial notebook (see Usage above).
- [`tests/`](tests/) — the test suite, including sample product data under
  `tests/data/`.
- [`docs/`](docs/) — additional documentation.

### Tests

To run the tests, we use `pytest`:

    uv run --all-extras pytest -n auto -v -m "not cdse" .
    uv run --env-file .env --all-extras pytest -v -m "cdse" .

Ideally, you would put your CDSE credentials in the .env file (see section above), so that the tests that read data from CDSE can run. Otherwise, the tests will be skipped. Note that the tests reading from CDSE are run separately in the commands above, on a single worker, to avoid issues related to rate limiting. Also, those tests are marked as "flaky", i.e., they are retried if/when they fail (due to rate limiting).

### Setting up a development environment

To install the package in editable mode, you can run:

    uv sync

or

    pip install -e . --group dev

### Code formatting

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

### Making a release

1. generate the changelog: `uv run --no-project --with git-cliff git cliff --unreleased` and update `CHANGELOG.md` manually
2. update the version in `pyproject.toml` (try to respect semantic versioning)
3. run `uv lock` to update uv.lock
4. commit (message="x.y.z") and tag the commit (tag="x.y.z")
5. push with the tag (`git push --tags`)

### Tips for external contributors

Make sure to have pyproj data: `pyproj sync -v --file us_nga_egm96_15`
