"""Persistent Scatterer Interferometry (PSI) and time-series InSAR tooling
(e.g. Ferretti et al. 2001-style PSI), built on top of coregistered SAR
crop stacks produced by ``teosar.tsinsar``.

This part of the codebase is more research-oriented and not as heavily
tested as ``eos``.
"""

from teosar import center_surround as center_surround
from teosar import inout as inout
from teosar import neighbors as neighbors
from teosar import overlap_utils as overlap_utils
from teosar import periodogram as periodogram
from teosar import prepare_overlap_db as prepare_overlap_db
from teosar import psc as psc
from teosar import psutils as psutils
from teosar import tsinsar as tsinsar
from teosar import utils as utils
from teosar import workflow as workflow

try:
    from teosar import ferreti_2001 as ferreti_2001
    from teosar import periodogram_cl as periodogram_cl
    from teosar import periodogram_par as periodogram_par
except ModuleNotFoundError:
    # if tensorflow or pyopencl are not found, it's probably because the user used `teosar-light` extra
    pass
