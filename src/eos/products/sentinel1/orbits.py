"""Retrieve precise/restituted Sentinel-1 orbit state vectors from local .EOF files.

Deprecated: use `eos.products.sentinel1.orbit_catalog` instead.
"""

import datetime
import glob
import io
import os
import warnings
from typing import Any, Optional, Sequence, Union

from lxml import etree

from eos.sar.orbit import StateVector

from .metadata import (
    Sentinel1BurstMetadata,
    Sentinel1GRDMetadata,
    isostring_to_timestamp,
)


def _string_to_timestamp(s):
    """Convert a string representing a date and time to a float number."""
    return (
        datetime.datetime.strptime(s, "%Y%m%dT%H%M%S")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def _parse_start_end_date_from_orbit_file(s):
    """
    Extract start and end dates for an orbit file filename.

    Args:
      s (str): filename string, formatted as \
      S1A_OPER_AUX_POEORB_OPOD_20161102T122427_V20161012T225943_20161014T005943.EOF

    Return:
      start, end (str): two dates as string (20161012T225943 and 20161014T005943 in the example)
    """
    start = s.split("_")[6][1:]
    end = s.split("_")[7].split(".")[0]
    return start, end


def select_orbit_files_from_filelist(files, date, missionid):
    """
    Select the orbit files of `missionid` that cover `date` with margin.

    Parameters
    ----------
    files : list of str
        Candidate orbit filenames (.EOF), formatted as
        S1A_OPER_AUX_POEORB_OPOD_20161102T122427_V20161012T225943_20161014T005943.EOF.
    date : str
        Date to cover, formatted as "%Y%m%dT%H%M%S".
    missionid : str
        Mission id (e.g. "S1A"), matched case-insensitively against the
        start of each filename.

    Returns
    -------
    list of str
        Matching filenames, sorted, whose validity interval covers `date`
        with a safety buffer of a few state vectors on each side.

    Raises
    ------
    FileNotFoundError
        If no orbit file covers `date` for `missionid`.
    """
    date = _string_to_timestamp(date)
    missionid = missionid.lower()

    candidates = []
    for file in files:
        filename = os.path.basename(file)

        if filename[: len(missionid)].lower() != missionid:
            continue

        s, e = _parse_start_end_date_from_orbit_file(filename)
        s = _string_to_timestamp(s)
        e = _string_to_timestamp(e)

        # time buffer of 10 state vectors with 10 seconds per state vector before the date
        buffer_pre = 10 * 10
        # time buffer of 20 state vectors with 10 seconds per state vector after the date, since the date often indicates the beginning of the product
        buffer_post = 10 * 10

        if s + buffer_pre < date and e - buffer_post > date:
            candidates.append(file)

    if candidates:
        return sorted(candidates)

    raise FileNotFoundError(
        f"could not find an orbit file for date={date} mission={missionid}"
    )


def retrieve_new_statevectors_to_slc_burst(
    xml_content, burst: Sentinel1BurstMetadata
) -> list[StateVector]:
    """Retrieve orbit state vectors from an orbit file xml, around a burst.

    Parameters
    ----------
    xml_content : str, bytes, or io.BytesIO
        Content of the orbit (.EOF) xml file.
    burst : Sentinel1BurstMetadata
        Burst metadata whose (approximate) time window is used to select
        the relevant state vectors.

    Returns
    -------
    list of StateVector
        State vectors from the orbit file, in a window centered on the burst.
    """
    return retrieve_new_statevectors_to_slc_bursts(xml_content, [burst])


def retrieve_new_statevectors_to_slc_bursts(
    xml_content: Union[str, bytes, io.BytesIO], bursts: Sequence[Sentinel1BurstMetadata]
) -> list[StateVector]:
    """Retrieve orbit state vectors from an orbit file xml, around a list of bursts.

    Parameters
    ----------
    xml_content : str, bytes, or io.BytesIO
        Content of the orbit (.EOF) xml file.
    bursts : Sequence[Sentinel1BurstMetadata]
        Bursts metadata (from a single acquisition/datatake) whose combined
        (approximate) time window is used to select the relevant state vectors.

    Returns
    -------
    list of StateVector
        State vectors from the orbit file, in a window centered on the bursts.
    """
    return get_new_list_of_statevectors(xml_content, [b.state_vectors for b in bursts])


def retrieve_new_statevectors_to_grd_meta(
    xml_content: Union[str, bytes, io.BytesIO], meta: Sentinel1GRDMetadata
) -> list[StateVector]:
    """Retrieve orbit state vectors from an orbit file xml, around a GRD product.

    Parameters
    ----------
    xml_content : str, bytes, or io.BytesIO
        Content of the orbit (.EOF) xml file.
    meta : Sentinel1GRDMetadata
        GRD product metadata whose (approximate) time window is used to
        select the relevant state vectors.

    Returns
    -------
    list of StateVector
        State vectors from the orbit file, in a window centered on the product.
    """
    return get_new_list_of_statevectors(xml_content, (meta.state_vectors,))


def retrieve_new_statevectors_to_dict_meta(
    xml_content: Union[str, bytes, io.BytesIO], meta: dict[str, Any]
) -> list[StateVector]:
    """Retrieve orbit state vectors from an orbit file xml, around a metadata dict.

    Parameters
    ----------
    xml_content : str, bytes, or io.BytesIO
        Content of the orbit (.EOF) xml file.
    meta : dict
        Metadata dict containing a "state_vectors" key, whose (approximate)
        time window is used to select the relevant state vectors.

    Returns
    -------
    list of StateVector
        State vectors from the orbit file, in a window centered on `meta`.
    """
    return get_new_list_of_statevectors(xml_content, (meta["state_vectors"],))


def get_new_list_of_statevectors(
    xml_content: Union[str, bytes, io.BytesIO],
    statevectors_list: Sequence[Sequence[StateVector]],
) -> list[StateVector]:
    """
    Extract, from an orbit file xml, the state vectors around a time window.

    The time window used is a 3-minute window centered on the midpoint
    between the earliest and latest state vector times found across
    `statevectors_list`.

    Parameters
    ----------
    xml_content : str, bytes, or io.BytesIO
        Content of the orbit (.EOF) xml file.
    statevectors_list : Sequence[Sequence[StateVector]]
        Groups of (already known, e.g. restituted) state vectors, used only
        to determine the time window of interest.

    Returns
    -------
    list of StateVector
        State vectors parsed from the orbit file within the time window.
    """
    # compute the approximative middle time of the burst/product
    # we will extract all orbit data over a window of 3 minutes centered around this middle
    start = min([state_vectors[0].time for state_vectors in statevectors_list])
    end = max([state_vectors[-1].time for state_vectors in statevectors_list])
    mid = (start + end) / 2

    newsvs: list[StateVector] = []

    if isinstance(xml_content, str):
        xml_content = io.BytesIO(xml_content.encode("utf-8"))

    context = etree.iterparse(xml_content, events=("end",), tag="OSV")
    for _, element in context:
        date = isostring_to_timestamp(element.findtext("UTC")[4:])

        if date < mid - 90:
            continue
        if date > mid + 90:
            break

        x = float(element.findtext("X"))
        y = float(element.findtext("Y"))
        z = float(element.findtext("Z"))
        vx = float(element.findtext("VX"))
        vy = float(element.findtext("VY"))
        vz = float(element.findtext("VZ"))
        newsvs.append(
            StateVector(
                time=date,
                position=(x, y, z),
                velocity=(vx, vy, vz),
            )
        )

    return newsvs


def search_valid_orbit_files_from_local_folder(path, product_info, type):
    """
    Find the most recent local orbit file of `type` covering `product_info`.

    Parameters
    ----------
    path : str
        Path to a folder containing .EOF orbit files.
    product_info : tuple
        (date, missionid), where date is formatted as "%Y%m%dT%H%M%S" and
        missionid is e.g. "S1A".
    type : str
        Orbit type, "poe" (precise) or "res" (restituted).

    Returns
    -------
    str or None
        Path to the most recent matching orbit file, or None if none is found.
    """
    date, missionid = product_info

    files = glob.glob(
        f"{path}/{missionid.upper()}_OPER_AUX_{type.upper()}ORB_OPOD_*.EOF"
    )
    try:
        files = select_orbit_files_from_filelist(files, date, missionid)
    except FileNotFoundError:
        return None

    return files[-1]


def _retrieve_statevectors_from_source(
    product_info, burst, *, force_type, source
) -> tuple[list[StateVector], str]:
    if isinstance(product_info, tuple):
        # ('20210216T151206', 'S1A')
        date, missionid = product_info
    else:
        # 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
        assert isinstance(product_info, str)
        missionid = product_info[:3]
        date = product_info[17:32]

    def try_for_orbit_type(type) -> Optional[tuple[list[StateVector], str]]:
        xml = source(date, missionid, type)
        if not xml:
            return None

        orbtype = f"orb{type}"
        if isinstance(burst, Sentinel1BurstMetadata):
            statevectors = retrieve_new_statevectors_to_slc_burst(xml, burst)
        elif isinstance(burst, list) and isinstance(burst[0], Sentinel1BurstMetadata):
            statevectors = retrieve_new_statevectors_to_slc_bursts(xml, burst)
        elif isinstance(burst, Sentinel1GRDMetadata):
            statevectors = retrieve_new_statevectors_to_grd_meta(xml, burst)
        elif isinstance(burst, dict):
            statevectors = retrieve_new_statevectors_to_dict_meta(xml, burst)
        else:
            assert False, burst

        return statevectors, orbtype

    if force_type:
        ret = try_for_orbit_type(force_type.replace("orb", ""))
    else:
        ret = try_for_orbit_type("poe") or try_for_orbit_type("res")
    if ret is not None:
        return ret

    raise FileNotFoundError(
        f"could not find an orbit file for date={date} mission={missionid}"
    )


def retrieve_statevectors_using_local_folder(
    path, product_info, burst, b, force_type=None
) -> tuple[list[StateVector], str]:
    """Retrieve the orbit statevectors of the given bursts using a local folder.

    Args
        path: filesystem path to a folder containing .EOF files
        product_info: can be either a S1 SLC product_id (str) or a tuple containing the missionid (str) and the date (str)
        burst: can be either a single burst metadata (Sentinel1BurstMetadata or Sentinel1GRDMetadata) or a list of burst metadata (list[Sentinel1BurstMetadata]). It should be metadata corresponding to a single acquisition/datatake.
        force_type (str, optional): request a specific type of orbit file (can be 'orbres' or 'orbpoe')

    Returns
        The two return values can be used for the *Metadata.with_new_state_vectors() method:
        str: the type of orbit found ('orbres' or 'orbpre')
        list: list of StateVector retrieved

    Raises
        FileNotFoundError: if no orbit file is found for the product_info
    """
    warnings.warn(
        "the sentinel1.orbits module is deprecated, use sentinel1.orbit_catalog instead.",
        DeprecationWarning,
    )

    def source(date, missionid, type):
        file = search_valid_orbit_files_from_local_folder(path, (date, missionid), type)
        if not file:
            return None

        return open(file, "rb")

    return _retrieve_statevectors_from_source(
        product_info, burst, force_type=force_type, source=source
    )
