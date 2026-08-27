import eos.products.sentinel1 as s1
from eos.products.sentinel1.overlap import Bsint, Osid
from eos.sar.roi import Roi


class OverlapRoiInfo:
    """ROIs describing Sentinel-1 burst overlap regions for a primary swath model.

    Bundles, per `Osid` (overlap spatial id), the regions of interest
    needed to read and resample burst overlaps for coregistration checks:
    the ROI within the burst, the ROI to write to, the output shape, and
    the ROI within the swath. Built from
    `eos.products.sentinel1.proj_model`'s `get_overlaps_roi` via
    `from_model`.
    """

    def __init__(
        self,
        all_osids,
        all_within_burst_rois,
        all_write_rois,
        all_out_shapes,
        all_within_swath_rois,
    ):
        self.all_osids = all_osids
        self.all_within_burst_rois = all_within_burst_rois
        self.all_write_rois = all_write_rois
        self.all_out_shapes = all_out_shapes
        self.all_within_swath_rois = all_within_swath_rois

    def get_swath_rois_per_bsint(self):
        """Return the within-swath ROI for each `Bsint`, memoized on first call.

        Returns
        -------
        dict[Bsint, Roi]
            Within-swath ROI for each burst spatial intersection. Raises an
            `AssertionError` if the osids of a `Bsint` disagree on their
            within-swath ROI.
        """
        if not hasattr(self, "swath_rois_per_bsint"):
            self.swath_rois_per_bsint: dict[Bsint, Roi] = {}
            for osid, roi in self.all_within_swath_rois.items():
                bsint = osid.bsint
                if bsint not in self.swath_rois_per_bsint.keys():
                    self.swath_rois_per_bsint[bsint] = roi
                else:
                    assert self.swath_rois_per_bsint[bsint] == roi
        return self.swath_rois_per_bsint

    @staticmethod
    def from_model(primary_swath_model):
        """Build an `OverlapRoiInfo` from a Sentinel-1 swath sensor model.

        Parameters
        ----------
        primary_swath_model
            Sentinel-1 swath sensor model exposing `get_overlaps_roi()`
            (see `eos.products.sentinel1.proj_model`).

        Returns
        -------
        OverlapRoiInfo
        """
        return OverlapRoiInfo(*primary_swath_model.get_overlaps_roi())

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation of this instance."""
        return dict(
            all_osids=[str(s) for s in self.all_osids],
            all_within_burst_rois=roidict_to_tupledict(self.all_within_burst_rois),
            all_write_rois=roidict_to_tupledict(self.all_write_rois),
            all_out_shapes={str(k): o for k, o in self.all_out_shapes.items()},
            all_within_swath_rois=roidict_to_tupledict(self.all_within_swath_rois),
        )

    @staticmethod
    def from_dict(info_dict):
        """Build an `OverlapRoiInfo` from a dict as produced by `to_dict`."""
        return OverlapRoiInfo(
            set([Osid.from_str(o) for o in info_dict["all_osids"]]),
            tupledict_to_roidict(info_dict["all_within_burst_rois"]),
            tupledict_to_roidict(info_dict["all_write_rois"]),
            {Osid.from_str(k): o for k, o in info_dict["all_out_shapes"].items()},
            tupledict_to_roidict(info_dict["all_within_swath_rois"]),
        )


class OverlapResampler:
    """Reads and resamples secondary-image burst overlap regions onto the primary grid.

    Uses the ROIs from an `OverlapRoiInfo` to warp, read, and resample
    (deramp/reramp) the overlap regions of a secondary Sentinel-1 image
    for a set of overlap spatial ids (`Osid`).
    """

    def __init__(
        self,
        ovl_roi_info: OverlapRoiInfo,
        primary_cutter: s1.acquisition.PrimarySentinel1AcquisitionCutter,
    ):
        """
        Parameters
        ----------
        ovl_roi_info : OverlapRoiInfo
            Overlap ROIs for the primary swath.
        primary_cutter : eos.products.sentinel1.acquisition.PrimarySentinel1AcquisitionCutter
            Cutter for the primary acquisition, used to get burst shapes.
        """
        self.ovl_roi_info = ovl_roi_info
        self.primary_cutter = primary_cutter

    def resample(
        self,
        osids,
        burst_resampling_matrices,
        secondary_resampler_provider,
        secondary_cutter,
        image_readers,
        get_complex=True,
        reramp=True,
    ):
        """Warp, read, and resample the overlap regions of a secondary image.

        Parameters
        ----------
        osids : Iterable[Osid]
            Overlap spatial ids to process; intersected with the ids known
            to `self.ovl_roi_info`.
        burst_resampling_matrices : dict
            Resampling matrix for each bsid involved in `osids`.
        secondary_resampler_provider : Callable
            Factory called as `(bsid, shape, matrix)` to build a resampler
            for each burst.
        secondary_cutter
            Cutter for the secondary acquisition.
        image_readers
            Readers for the secondary image data, passed through to
            `eos.products.sentinel1.overlap.warp_rois_read_resample_ovl`.
        get_complex : bool, optional
            Whether to read the data as complex. Defaults to True.
        reramp : bool, optional
            Whether to reramp the resampled overlaps. Defaults to True.

        Returns
        -------
        all_resampled_ovls
            Resampled overlap arrays, per osid.
        all_read_rois_correc
            Corrected read ROIs used, per osid.
        all_resamplers
            Resamplers used, per bsid.
        """
        osids_intersection = self.ovl_roi_info.all_osids.intersection(osids)

        bsids_for_osids = set([o.bsid() for o in osids_intersection])
        # instantiate resamplers
        resamplers = {
            bsid: secondary_resampler_provider(
                bsid,
                self.primary_cutter.get_burst_outer_roi_in_tiff(bsid).get_shape(),
                burst_resampling_matrices[bsid],
            )
            for bsid in bsids_for_osids
        }

        (
            all_resampled_ovls,
            all_read_rois_correc,
            all_resamplers,
        ) = s1.overlap.warp_rois_read_resample_ovl(
            osids_intersection,
            resamplers,
            self.ovl_roi_info.all_within_burst_rois,
            secondary_cutter,
            image_readers,
            self.ovl_roi_info.all_write_rois,
            self.ovl_roi_info.all_out_shapes,
            get_complex=get_complex,
            margin=5,
            reramp=reramp,
        )

        return all_resampled_ovls, all_read_rois_correc, all_resamplers


def roidict_to_tupledict(roi_dict):
    """Convert a `{Osid: Roi}` dict to a JSON-serializable `{str: tuple}` dict."""
    return {str(k): r.to_roi() for k, r in roi_dict.items()}


def tupledict_to_roidict(tuple_dict):
    """Convert a `{str: tuple}` dict (as produced by `roidict_to_tupledict`) back to `{Osid: Roi}`."""
    return {Osid.from_str(k): Roi.from_roi_tuple(r) for k, r in tuple_dict.items()}
