"""Catalog search for Sentinel-1 SLC and GRD products (e.g. via CDSE)."""

import abc
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Literal, Union

import requests
import shapely
from typing_extensions import override

import eos.cache

logger = logging.Logger(__name__)

ProductPolarization = Literal["SV", "DV", "SH", "DH"]


@dataclass(frozen=True)
class Sentinel1CatalogQuery:
    """Criteria describing the Sentinel-1 products to search for."""

    geometry: shapely.Geometry
    """Area of Interest, products should *intersect* with it."""
    start_date: datetime.datetime
    """Earliest acquisition start date/time to consider."""
    end_date: datetime.datetime
    """Latest acquisition start date/time to consider."""
    relative_orbit_number: int
    """Relative orbit number products must be acquired on."""
    polarization: list[ProductPolarization]
    """Accepted polarization channel combinations."""


@dataclass(frozen=True)
class Sentinel1CatalogResult:
    """Result of a catalog search."""

    product_ids: list[str]
    """All matching product ids."""
    product_ids_per_date: dict[str, list[str]]
    """Matching product ids grouped by the date (YYYYMMDD) of the first
    product of their datatake, keyed on that date."""


@dataclass(frozen=True)
class Sentinel1SLCCatalogBackend(abc.ABC):
    """Base class for catalog backends returning Sentinel-1 IW SLC products."""

    @abc.abstractmethod
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        """
        Get the list of Sentinel-1 IW SLC product satisfying the provided query.
        """


@dataclass(frozen=True)
class Sentinel1GRDCatalogBackend(abc.ABC):
    """Base class for catalog backends returning Sentinel-1 IW GRD products."""

    @abc.abstractmethod
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        """
        Get the list of Sentinel-1 IW GRD product satisfying the provided query.
        """


def _search_from_backend(
    backend: Union[Sentinel1SLCCatalogBackend, Sentinel1GRDCatalogBackend],
    query: Sentinel1CatalogQuery,
    cache: eos.cache.Cache = eos.cache.no_cache(),
) -> Sentinel1CatalogResult:
    def pid2datatake(product_id: str) -> str:
        # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
        # mix the mission id and the datatake id
        return product_id.split("_")[0] + "_" + product_id.split("_")[8]

    def pid2date(product_id: str) -> str:
        # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
        return product_id.split("_")[5][:8]

    if (items := cache.get(query, list[str])) is None:
        items = backend.search(query)
        if query.end_date < datetime.datetime.now():
            cache.put(query, items)

    by_datatake: dict[str, list[str]] = {}
    for pid in items:
        by_datatake.setdefault(pid2datatake(pid), []).append(pid)

    # date of first product: list of product ids of the same datatake
    product_ids_per_date = {
        pid2date(sorted(by_datatake[datatake])[0]): sorted(by_datatake[datatake])
        for datatake in sorted(by_datatake.keys())
    }

    return Sentinel1CatalogResult(
        product_ids=items, product_ids_per_date=product_ids_per_date
    )


def search_slc(
    backend: Sentinel1SLCCatalogBackend,
    query: Sentinel1CatalogQuery,
    cache: eos.cache.Cache = eos.cache.no_cache(),
) -> Sentinel1CatalogResult:
    """Search Sentinel-1 IW SLC products matching `query` via `backend`.

    Parameters
    ----------
    backend : Sentinel1SLCCatalogBackend
        Catalog backend to run the search against.
    query : Sentinel1CatalogQuery
        Search criteria.
    cache : eos.cache.Cache, optional
        Cache used to avoid repeating identical searches. The default is to
        not cache.

    Returns
    -------
    Sentinel1CatalogResult
        Matching product ids, grouped by datatake date.
    """
    return _search_from_backend(backend, query, cache)


def search_grd(
    backend: Sentinel1GRDCatalogBackend,
    query: Sentinel1CatalogQuery,
    cache: eos.cache.Cache = eos.cache.no_cache(),
) -> Sentinel1CatalogResult:
    """Search Sentinel-1 IW GRD products matching `query` via `backend`.

    Parameters
    ----------
    backend : Sentinel1GRDCatalogBackend
        Catalog backend to run the search against.
    query : Sentinel1CatalogQuery
        Search criteria.
    cache : eos.cache.Cache, optional
        Cache used to avoid repeating identical searches. The default is to
        not cache.

    Returns
    -------
    Sentinel1CatalogResult
        Matching product ids, grouped by datatake date.
    """
    return _search_from_backend(backend, query, cache)


def _cdse_list_items(request: str) -> list[str]:
    limit = 1000
    request = f"{request}&$top={limit}"
    response = requests.get(request)
    response.raise_for_status()
    items = response.json()

    try:
        items = items["value"]
    except KeyError as e:
        raise Exception(f"OData parsing error? : {items}") from e

    assert len(items) < limit, (
        "maximum odata 'number of results' reached, please ask for the implementation of pagination"
    )

    pids = [
        item["Name"].replace(".SAFE", "")
        for item in items
        if item["EvictionDate"]
        in ("", "9999-12-31T23:59:59.999Z", "9999-12-31T23:59:59.999999Z")
        and item["Online"]
    ]
    return pids


def _pol_query_str(polarizations: list[ProductPolarization]) -> str:
    # %26 is the url encoding of &
    pol_dict = {"SV": "VV", "DV": "VV%26VH", "SH": "HH", "DH": "HH%26HV"}

    def pol_to_str(pol: str) -> str:
        return f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'polarisationChannels' and att/OData.CSC.StringAttribute/Value eq '{pol_dict[pol]}')"

    # we need to do a nested or statement
    full_str = " or ".join([pol_to_str(pol) for pol in polarizations])
    return f"({full_str})"


def query_to_request_str(
    query: Sentinel1CatalogQuery,
    product_type: Literal["IW_SLC__1S", "IW_GRDH_1S", "IW_GRDH_1S-COG"],
) -> str:
    """Build a CDSE OData request URL for `query` restricted to `product_type`.

    Parameters
    ----------
    query : Sentinel1CatalogQuery
        Search criteria.
    product_type : {"IW_SLC__1S", "IW_GRDH_1S", "IW_GRDH_1S-COG"}
        CDSE product type to filter on.

    Returns
    -------
    str
        Full CDSE OData Products request URL.
    """
    # TODO: we might want to look at neighbouring orbits
    # because of its loose definition around the equator
    # It can be achieved with an or statement similarly to polarizations
    request = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
        f"Collection/Name eq 'SENTINEL-1' "
        f"and ContentDate/Start gt {query.start_date.isoformat()} "
        f"and ContentDate/Start lt {query.end_date.isoformat()} "
        f"and Data.CSC.Intersects(area=geography'SRID=4326;{query.geometry.wkt}') "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'processingLevel' and att/OData.CSC.StringAttribute/Value eq 'LEVEL1') "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq {query.relative_orbit_number}) "
        f"and {_pol_query_str(query.polarization)}"
        "&$expand=Attributes&$orderby=ContentDate/Start asc"
    )
    return request


@dataclass(frozen=True)
class CDSESentinel1SLCCatalogBackend(Sentinel1SLCCatalogBackend):
    """Sentinel-1 IW SLC catalog backend querying CDSE's OData API."""

    def get_cdse_item(self, product_id: str) -> dict[str, Any]:
        """Get the raw CDSE OData item (with attributes) for `product_id`.

        Parameters
        ----------
        product_id : str
            Sentinel-1 product id (without the ".SAFE" extension).

        Returns
        -------
        dict
            The CDSE OData item for the product.
        """
        response = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name%20eq%20%27{product_id}.SAFE%27&$expand=Attributes"
        ).json()
        try:
            return response["value"][0]
        except KeyError as e:
            raise Exception(f"OData parsing error? : {response}") from e

    @override
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        """Get the list of Sentinel-1 IW SLC product ids matching `query`, from CDSE."""
        request = query_to_request_str(query, "IW_SLC__1S")
        return _cdse_list_items(request)


@dataclass(frozen=True)
class CDSESentinel1GRDCatalogBackend(Sentinel1GRDCatalogBackend):
    """Sentinel-1 IW GRD catalog backend querying CDSE's OData API."""

    use_cog_products: bool = False
    """
    This parameter defines the behavior of the `search` method and switches
    between the 'IW_GRDH_1S' and the 'IW_GRDH_1S-COG' collection.
    The `get_cdse_item` method would still work for a non COG product ID
    and vice-versa regardless of the value of this flag.
    """

    def get_cdse_item(self, product_id: str) -> dict[str, Any]:
        """Get the raw CDSE OData item (with attributes) for `product_id`.

        Parameters
        ----------
        product_id : str
            Sentinel-1 product id (without the ".SAFE" extension).

        Returns
        -------
        dict
            The CDSE OData item for the product.
        """
        response = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name%20eq%20%27{product_id}.SAFE%27&$expand=Attributes"
        ).json()
        try:
            return response["value"][0]
        except KeyError as e:
            raise Exception(f"OData parsing error? : {response}") from e

    @override
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        """Get the list of Sentinel-1 IW GRD product ids matching `query`, from CDSE."""
        request = (
            query_to_request_str(query, "IW_GRDH_1S-COG")
            if self.use_cog_products
            else query_to_request_str(query, "IW_GRDH_1S")
        )
        return _cdse_list_items(request)
