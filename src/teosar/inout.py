import json
import os
import warnings
from typing import Optional

import numpy as np
import rasterio
import requests.exceptions

import eos.products.sentinel1 as s1
from eos.sar import io
from eos.sar.orbit import StateVector
from eos.sar.roi import Roi


def get_inputs_for_date(
    product_ids, swath, product_provider, statevectors: Optional[list[StateVector]], pol
):
    """
    Fetch Sentinel-1 products for one date and assemble them into a `Sentinel1Assembler`.

    Products that fail to fetch (HTTP error) are skipped with a printed
    message rather than raising.

    Parameters
    ----------
    product_ids : Iterable[str]
        Ids of the products to fetch for this date.
    swath : str
        Swath to process, one of "iw1", "iw2", "iw3", or "all".
    product_provider : Callable[[str], Any]
        Callable used to fetch a product by id.
    statevectors : list[StateVector], optional
        Orbit state vectors to use for assembling the products.
    pol : str
        Polarization to assemble.

    Returns
    -------
    products : list
        Successfully fetched products.
    asm : eos.products.sentinel1.assembler.Sentinel1Assembler
        Assembler built from the fetched products.
    """
    products = []
    for pid in product_ids:
        try:
            products.append(product_provider(pid))
        except requests.exceptions.HTTPError as e:
            print("skip", pid, e)

    swaths = ("iw1", "iw2", "iw3") if swath == "all" else (swath.lower(),)
    asm = s1.assembler.Sentinel1Assembler.from_products(
        products, pol, statevectors, swaths=swaths
    )

    return products, asm


def formatter(outdir, date, extension=".tif"):
    """Build a `{outdir}/{date}{extension}` path."""
    return os.path.join(outdir, f"{date}{extension}")


class DirectoryBuilder:
    """Builds standardized output file paths for a single-image (non-overlap) processing run.

    Creates (unless `makedirs=False`) and exposes subdirectories of
    `dstdir` for metadata, DEM, images, flattened phase, topographic
    phase, and orthorectifier outputs.
    """

    def __init__(
        self,
        dstdir,
        meta_dir="meta",
        dem_dir="dem",
        imgs_dir="imgs",
        flat_dir="flat",
        topo_dir="topo",
        ortho_dir="orthorectifier",
        makedirs=True,
    ):
        """
        Parameters
        ----------
        dstdir : str
            Root output directory.
        meta_dir, dem_dir, imgs_dir, flat_dir, topo_dir, ortho_dir : str, optional
            Names of the subdirectories created under `dstdir`.
        makedirs : bool, optional
            If True (default), create the subdirectories on disk.
        """
        self.dstdir = dstdir

        self.meta_dir = os.path.join(self.dstdir, meta_dir)
        self.dem_dir = os.path.join(self.dstdir, dem_dir)
        self.imgs_dir = os.path.join(self.dstdir, imgs_dir)
        self.flat_dir = os.path.join(self.dstdir, flat_dir)
        self.topo_dir = os.path.join(self.dstdir, topo_dir)
        self.ortho_dir = os.path.join(self.dstdir, ortho_dir)

        if makedirs:
            out_dirs = [
                self.meta_dir,
                self.dem_dir,
                self.imgs_dir,
                self.flat_dir,
                self.topo_dir,
                self.ortho_dir,
            ]

            for out_dir in out_dirs:
                os.makedirs(out_dir, exist_ok=True)

    def get_meta_path(self, date):
        """Path to the JSON metadata file for `date`."""
        return formatter(self.meta_dir, date, ".json")

    def get_geo_dem_path(self):
        """Path to the geocoded DEM GeoTIFF."""
        return os.path.join(self.dem_dir, "geo_dem.tif")

    def get_radar_dem_path(self):
        """Path to the radar-coded DEM GeoTIFF."""
        return os.path.join(self.dem_dir, "radar_dem.tif")

    def get_img_path(self, date):
        """Path to the SLC image GeoTIFF for `date`."""
        return formatter(self.imgs_dir, date, ".tif")

    def get_flat_path(self, date):
        """Path to the flattened (simulated flat-earth) phase GeoTIFF for `date`."""
        return formatter(self.flat_dir, date, ".tif")

    def get_topo_path(self, date):
        """Path to the simulated topographic phase GeoTIFF for `date`."""
        return formatter(self.topo_dir, date, ".tif")

    def get_proc_path(self):
        """Path to the JSON processing parameters file."""
        return os.path.join(self.dstdir, "proc.json")

    def get_svg_path(self):
        """Path to the SVG file used to visualize the AOI/scene footprint."""
        return os.path.join(self.dstdir, "loc.svg")

    def get_pickle_path(self, date):
        """Path to the pickled metadata file for `date`."""
        return formatter(self.meta_dir, date, ".pickle")

    def get_lut_path(self):
        """Path to the orthorectification lookup table GeoTIFF."""
        return os.path.join(self.ortho_dir, "lut.tif")

    def get_ortho_path(self):
        """Path to the JSON orthorectifier parameters file."""
        return os.path.join(self.ortho_dir, "orthorectifier.json")

    def get_proj_model_path(self):
        """Path to the JSON sensor (projection) model file."""
        return os.path.join(self.ortho_dir, "proj_model.json")


class OvlDirectoryBuilder(DirectoryBuilder):
    """Builds standardized output file paths for a burst-overlap (PSI-style) processing run.

    Extends `DirectoryBuilder` with paths keyed by burst overlap id
    (`Osid`)/intersection (`Bsint`) for interferograms and per-burst-overlap
    images, on top of the base per-date images/simulations.
    """

    def __init__(
        self,
        dstdir,
        meta_dir="meta",
        dem_dir="dem",
        imgs_dir="imgs",
        flat_dir="flat",
        topo_dir="topo",
        ifgs_dir="ifgs",
        ifgs_esd_dir="ifgs_esd",
        ifg_meta="ifgs_meta",
        makedirs=True,
    ):
        """
        Parameters
        ----------
        dstdir : str
            Root output directory.
        meta_dir, dem_dir, imgs_dir, flat_dir, topo_dir : str, optional
            Names of the subdirectories created under `dstdir` (see
            `DirectoryBuilder`).
        ifgs_dir, ifgs_esd_dir, ifg_meta : str, optional
            Names of the subdirectories created under `dstdir` for
            interferograms, ESD-corrected interferograms, and interferogram
            metadata.
        makedirs : bool, optional
            If True (default), create the subdirectories on disk.
        """
        super().__init__(
            dstdir, meta_dir, dem_dir, imgs_dir, flat_dir, topo_dir, makedirs
        )
        self.ifgs_dir = os.path.join(self.dstdir, ifgs_dir)
        self.ifgs_esd_dir = os.path.join(self.dstdir, ifgs_esd_dir)
        self.ifg_meta = os.path.join(self.dstdir, ifg_meta)
        self.makedirs = makedirs

        if self.makedirs:
            for out_dir in [self.ifgs_esd_dir, self.ifgs_dir, self.ifg_meta]:
                os.makedirs(out_dir, exist_ok=True)

    def ovl_array_formatter(self, outdir, osid, date, extension=".tif"):
        """Build the path (creating its directory if needed) for one overlap/date array."""
        bsint = osid.bsint
        extension = ".tif"
        out_img_dir = os.path.join(outdir, str(bsint), f"{date}")

        if self.makedirs:
            os.makedirs(out_img_dir, exist_ok=True)

        fname = f"{str(osid)}{extension}"

        return os.path.join(out_img_dir, fname)

    def ifg_formatter(self, outdir, osid, date1, date2, extension=".tif"):
        """Build the path (creating its directory if needed) for one interferogram."""
        bsint = osid.bsint
        extension = ".tif"
        out_img_dir = os.path.join(outdir, f"{date1}_{date2}", str(bsint))

        if self.makedirs:
            os.makedirs(out_img_dir, exist_ok=True)

        fname = f"{str(osid)}{extension}"

        return os.path.join(out_img_dir, fname)

    def ovl_simulation_formatter(self, outdir, bsint, date, extension=".tif"):
        """Build the path (creating its directory if needed) for one burst-intersection simulation array."""
        extension = ".tif"
        out_img_dir = os.path.join(outdir, str(bsint))

        if self.makedirs:
            os.makedirs(out_img_dir, exist_ok=True)

        fname = f"{date}{extension}"

        return os.path.join(out_img_dir, fname)

    def get_meta_path(self, date):
        """Path to the JSON metadata file for `date`."""
        return formatter(self.meta_dir, date, ".json")

    def get_geo_dem_path(self):
        """Path to the geocoded DEM GeoTIFF."""
        return os.path.join(self.dem_dir, "geo_dem.tif")

    def get_radar_dem_path(self, bsint):  # type: ignore[override]
        """Path to the radar-coded DEM GeoTIFF for burst intersection `bsint`."""
        out_dir = os.path.join(self.dem_dir, "radar_dem")
        if self.makedirs:
            os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, str(bsint) + ".tif")

    def get_img_path(self, osid, date):  # type: ignore[override]
        """Path to the SLC overlap image GeoTIFF for `osid`/`date`."""
        return self.ovl_array_formatter(self.imgs_dir, osid, date, ".tif")

    def get_flat_path(self, bsint, date):  # type: ignore[override]
        """Path to the flattened (simulated flat-earth) phase GeoTIFF for `bsint`/`date`."""
        return self.ovl_simulation_formatter(self.flat_dir, bsint, date, ".tif")

    def get_topo_path(self, bsint, date):  # type: ignore[override]
        """Path to the simulated topographic phase GeoTIFF for `bsint`/`date`."""
        return self.ovl_simulation_formatter(self.topo_dir, bsint, date, ".tif")

    def get_proc_path(self):
        """Path to the JSON processing parameters file."""
        return os.path.join(self.dstdir, "proc.json")

    def get_ifg_path(self, osid, date1, date2, esd=False):
        """Path to the interferogram GeoTIFF for `osid`/`date1`/`date2`.

        Parameters
        ----------
        osid : Osid
            Overlap spatial id.
        date1, date2 : Any
            Dates of the interferogram pair.
        esd : bool, optional
            If True, use the ESD-corrected interferogram directory instead
            of the regular one. Defaults to False.
        """
        if esd:
            out_dir = self.ifgs_esd_dir
        else:
            out_dir = self.ifgs_dir

        return self.ifg_formatter(out_dir, osid, date1, date2)

    def get_ifg_meta_path(self, date1, date2):
        """Path to the JSON interferogram metadata file for the `date1`/`date2` pair."""
        return os.path.join(self.ifg_meta, f"{date1}_{date2}.json")


def save_inputs_to_file(out_path, **kwargs):
    """Save `kwargs` as a JSON object to `out_path` (see `dict_to_json`)."""
    dict_to_json(kwargs, out_path)


def dict_to_json(out_dict, out_path):
    """Write `out_dict` as JSON to `out_path`."""
    with open(out_path, "w") as f:
        json.dump(out_dict, f)


def json_to_dict(json_path):
    """Read and return the JSON content of `json_path` as a dict."""
    with open(json_path, "r") as f:
        json_txt = json.load(f)
    return json_txt


def asm_to_json(asm, meta_out_path):
    """Write a `Sentinel1Assembler`'s dict representation as JSON to `meta_out_path`."""
    dict_to_json(asm.to_dict(), meta_out_path)


def json_to_asm(json_path):
    """Read a `Sentinel1Assembler` back from a JSON file's `"asm"` key."""
    return s1.assembler.Sentinel1Assembler.from_dict(json_to_dict(json_path)["asm"])


def imcoords_to_svg(im_coords, svg_path):
    """Write a polygon of image coordinates `im_coords` as a minimal SVG file to `svg_path`."""
    points = " ".join(f"{x},{y}" for x, y in im_coords)
    with open(svg_path, "w") as f:
        f.write(
            f"""
        <svg width="1" height="1">
        <polygon points="{points}" stroke="red" stroke-width="0.1"/>
        </svg>
        """
        )


def save_img(path, array, transform=None, crs=None):
    """
    Save array with rasterio, optionally storing a transform and crs.
    The array can have a single band and be of shape (h, w)
    The array can have multiple bands and be of shape (nbands, h, w).
    """
    # Get image size
    array_shape = array.shape

    len_shape = len(array_shape)
    assert len_shape in [2, 3], (
        "Only 2D arrays (single band) and 3D arrays (multi band) supported"
    )

    if len_shape == 2:  # Single band
        count = 1
        height, width = array_shape
    else:  # Multiple bands
        count, height, width = array_shape

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )
        profile = dict(
            count=count, width=width, height=height, dtype=array.dtype, nodata=np.nan
        )

        if crs is not None and transform is not None:
            profile["crs"] = crs
            profile["transform"] = transform
        elif (crs is None) ^ (transform is None):
            print("Both crs and transform should be provided, they will be ignored")

        with rasterio.open(path, "w", **profile) as f:
            if count == 1:  # Single band
                f.write(np.squeeze(array), 1)
            else:  # Multiple bands
                f.write(np.squeeze(array))


def read_img(path, roi=None, get_complex=False):
    """
    Read a raster file, optionally windowed to one or several ROIs.

    Parameters
    ----------
    path : str
        Path to the raster file.
    roi : Roi or list[Roi], optional
        If a single `Roi`, read that window (via `eos.sar.io.read_window`).
        If a list of `Roi`, read each window (via `eos.sar.io.read_windows`).
        If None (default), read the full raster.
    get_complex : bool, optional
        Whether to read the data as complex. Defaults to False.

    Returns
    -------
    ndarray or list[ndarray]
        The read array, or a list of arrays if `roi` is a list.
    """
    with rasterio.open(path, "r") as reader:
        if roi is None:
            roi = Roi(0, 0, reader.width, reader.height)
        if isinstance(roi, Roi):
            return io.read_window(reader, roi, get_complex)
        elif isinstance(roi, list):
            return io.read_windows(reader, roi, get_complex)


class DirectoryReader:
    """Reads images/simulations from the paths produced by a `DirectoryBuilder`."""

    def __init__(self, dir_builder):
        """
        Parameters
        ----------
        dir_builder : DirectoryBuilder
            Builder providing the paths to read from.
        """
        self.dir_builder = dir_builder

    def _read(self, path, get_complex, roi=None):
        reader = rasterio.open(path, "r")
        if roi is None:
            return reader.read().squeeze()
        else:
            if isinstance(roi, Roi):
                return io.read_window(reader, roi, get_complex)
            elif isinstance(roi, list):
                return io.read_windows(reader, roi, get_complex)
            else:
                print("unrecognized type")

    def read_imgs(self, dates, roi=None):
        """Read the SLC image (as complex) for each date in `dates`, optionally windowed to `roi`."""
        return [
            self._read(im, True, roi)
            for im in map(self.dir_builder.get_img_path, dates)
        ]

    def _read_simulation(self, dates, path_provider, roi=None):
        ims = []

        for im_path in map(path_provider, dates):
            if os.path.exists(im_path):
                im = self._read(im_path, False, roi)
            else:
                im = None
            ims.append(im)
        return ims

    def read_flat_phase(self, dates, roi=None):
        """Read the simulated flat-earth phase for each date in `dates` (None if missing on disk)."""
        return self._read_simulation(dates, self.dir_builder.get_flat_path, roi)

    def read_topo_phase(self, dates, roi=None):
        """Read the simulated topographic phase for each date in `dates` (None if missing on disk)."""
        return self._read_simulation(dates, self.dir_builder.get_topo_path, roi)


class OvlDirectoryReader(DirectoryReader):
    """Reads burst-overlap images/simulations from the paths produced by an `OvlDirectoryBuilder`."""

    def read_imgs(self, osid, dates, roi=None, get_complex=True):  # type: ignore[override]
        """Read the SLC overlap image for `osid` at each date in `dates`, optionally windowed to `roi`."""
        return [
            self._read(self.dir_builder.get_img_path(osid, date), get_complex, roi)
            for date in dates
        ]

    def read_flat_phase(self, bsint, dates, roi=None):  # type: ignore[override]
        """Read the simulated flat-earth phase for `bsint` at each date in `dates` (None if missing on disk)."""

        def path_provider(date):
            return self.dir_builder.get_flat_path(bsint, date)

        return self._read_simulation(dates, path_provider, roi)

    def read_topo_phase(self, bsint, dates, roi=None):  # type: ignore[override]
        """Read the simulated topographic phase for `bsint` at each date in `dates` (None if missing on disk)."""

        def path_provider(date):
            return self.dir_builder.get_topo_path(bsint, date)

        return self._read_simulation(dates, path_provider, roi)

    def read_radarcoded_dem(self, bsint, roi=None):
        """Read the radar-coded DEM for `bsint`, optionally windowed to `roi`."""
        return self._read(
            self.dir_builder.get_radar_dem_path(bsint), get_complex=False, roi=roi
        )


def get_mlooked_gcps(gcps, filter_size):
    """
    Rescale ground control points' image coordinates for a multi-looked (filtered) image.

    Divides each GCP's row/col by `filter_size` to account for a
    multi-looking filter applied to the image.

    Parameters
    ----------
    gcps : Iterable
        Ground control points, each with `row`, `col`, `x`, `y`, `z`
        attributes.
    filter_size : tuple[float, float]
        (row, col) multi-looking factor.

    Returns
    -------
    list[rasterio.control.GroundControlPoint]
        GCPs with row/col rescaled by `filter_size`.
    """
    mlooked_gcps = []
    for gcp in gcps:
        mlooked_gcps.append(
            rasterio.control.GroundControlPoint(
                gcp.row / filter_size[0], gcp.col / filter_size[1], gcp.x, gcp.y, gcp.z
            )
        )
    return mlooked_gcps


def geojson_dict(input_coordinates, orbit, startdate, enddate, dstdir=""):
    """
    Build a GeoJSON Feature dict describing a processed scene's footprint.

    Parameters
    ----------
    input_coordinates : list
        Polygon coordinates, as expected by GeoJSON's `"Polygon"` geometry
        `"coordinates"` field.
    orbit : Any
        Orbit identifier/info, stored as-is in the feature's properties.
    startdate, enddate : datetime.datetime
        Start/end dates of the processed scene, stored as ISO-formatted
        strings.
    dstdir : str, optional
        Output directory, stored in the feature's properties. Defaults to
        `""`.

    Returns
    -------
    dict
        A GeoJSON Feature dict with a Polygon geometry and the given
        properties.
    """
    geo_dict = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": input_coordinates},
        "properties": {
            "orbit": orbit,
            "start_date": startdate.isoformat(),
            "end_date": enddate.isoformat(),
            "dstdir": dstdir,
        },
    }
    return geo_dict
