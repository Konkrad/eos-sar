"""Generic caching module.

Warning: the current implementation does not support concurrent read or write operations.
A Cache object should not be be used inside a multithread/multiprocess context.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import hashlib
import json
import os
import shelve
import typing
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Hashable, Optional, Type, TypeVar
from weakref import WeakValueDictionary

from typing_extensions import override

T = TypeVar("T")


def on_disk(path: str) -> Cache:
    """Create a :class:`Cache` backed by an on-disk ``shelve`` database.

    Parameters
    ----------
    path : str
        Path to the on-disk database (passed to ``shelve.open``, expanded
        with ``os.path.expanduser``). Reused if a database is already open
        at that path.

    Returns
    -------
    Cache
        Cache instance backed by an :class:`OnDiskCacheBackend`.
    """
    return Cache(OnDiskCacheBackend.open(path))


def no_cache() -> Cache:
    """Return a :class:`Cache` that stores nothing and always misses.

    Returns
    -------
    Cache
        A shared no-op cache instance.
    """
    return _NoCache


def json_default(o: Any) -> Any:
    """JSON ``default`` callback used to serialize otherwise non-serializable objects.

    Tries, in order, ``dataclasses.asdict``, ``o.__dict__``,
    ``o.__geo_interface__``, and an ISO-format string for
    ``datetime.datetime``. Intended for use as the ``default`` argument of
    ``json.dumps``, e.g. in :func:`hash_anything`.

    Parameters
    ----------
    o : Any
        Object that ``json.dumps`` could not serialize directly.

    Returns
    -------
    Any
        A JSON-serializable representation of ``o``.

    Raises
    ------
    TypeError
        If none of the fallback strategies apply.
    """
    try:
        return dataclasses.asdict(o)
    except TypeError:
        pass
    try:
        return o.__dict__
    except AttributeError:
        pass
    try:
        return o.__geo_interface__
    except AttributeError:
        pass
    if isinstance(o, datetime.datetime):
        return o.isoformat(timespec="microseconds")
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def hash_anything(obj) -> str:
    """Compute a stable SHA3-512 hex digest of a JSON-serializable object.

    ``obj`` is serialized with ``json.dumps`` (sorted keys, no extra
    whitespace, using :func:`json_default` as a fallback for non-native
    types) so that equal objects always hash to the same digest.
    See https://death.andgravity.com/stable-hashing#json .

    Parameters
    ----------
    obj : Any
        Object to hash. Must be JSON-serializable (directly, or via
        :func:`json_default`).

    Returns
    -------
    str
        Hexadecimal SHA3-512 digest of the canonical JSON encoding of
        ``obj``.
    """
    # see https://death.andgravity.com/stable-hashing#json
    return hashlib.sha3_512(
        json.dumps(
            obj,
            sort_keys=True,
            default=json_default,
            ensure_ascii=False,
            indent=None,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class CacheBackend(abc.ABC):
    """Storage interface used by :class:`Cache` to persist and retrieve values."""

    def put(self, key: str, object: Any) -> None:
        """Store ``object`` under ``key``."""
        ...

    def get(self, key: str) -> Optional[Any]:
        """Return the value stored under ``key``, or None if absent."""
        ...


@dataclass(frozen=True)
class NoCacheBackend(CacheBackend):
    """A :class:`CacheBackend` that stores nothing and always returns None."""

    @override
    def put(self, key: str, object: Any) -> None:
        pass

    @override
    def get(self, key: str) -> Optional[Any]:
        pass


@dataclass(frozen=True)
class OnDiskCacheBackend(CacheBackend):
    """A :class:`CacheBackend` persisting values to an on-disk ``shelve`` database."""

    # a dbm database can only be opened once at a time, so we share the existing instance
    _refs: ClassVar[WeakValueDictionary[str, shelve.Shelf]] = WeakValueDictionary()
    db: shelve.Shelf
    # TODO: locking? dbm does not support concurrent read/write

    @classmethod
    def open(cls, path: str) -> OnDiskCacheBackend:
        """Open (or reuse) the ``shelve`` database at ``path``.

        Since a dbm database can only be opened once at a time, an already
        open database for the same expanded path is reused instead of
        being reopened.

        Parameters
        ----------
        path : str
            Path to the database, expanded with ``os.path.expanduser``.

        Returns
        -------
        OnDiskCacheBackend
            Backend wrapping the (possibly shared) open database.
        """
        path = os.path.expanduser(path)
        if path in cls._refs:
            db = cls._refs[path]
        else:
            db = shelve.open(path)
            cls._refs[path] = db
        return OnDiskCacheBackend(db=db)

    @override
    def put(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` in the shelve database."""
        self.db[key] = value

    @override
    def get(self, key: str) -> Optional[Any]:
        """Return the value stored under ``key``, or None if absent."""
        try:
            return self.db[key]
        except KeyError:
            return None


@dataclass(frozen=True)
class Cache:
    """Key/value cache built on top of a :class:`CacheBackend`.

    The key can be any JSON-serializable (or dataclass-like) object; it is
    hashed with :func:`hash_anything` before being passed to the backend.
    Create instances with :func:`on_disk` or :func:`no_cache`.
    """

    backend: CacheBackend

    def put(self, key: Hashable, value: T) -> T:
        """Store ``value`` under ``key`` and return it unchanged.

        A no-op when this cache is the singleton returned by :func:`no_cache`.

        Parameters
        ----------
        key : Hashable
            JSON-serializable key identifying the value.
        value : T
            Value to store.

        Returns
        -------
        T
            ``value``, unchanged, for convenient chaining.
        """
        if self is _NoCache:
            return value
        self.backend.put(hash_anything(key), value)
        return value

    def get(self, key: Hashable, t: Type[T]) -> Optional[T]:
        """Retrieve the value stored under ``key``, if any.

        Always returns None when this cache is the singleton returned by
        :func:`no_cache`.

        Parameters
        ----------
        key : Hashable
            JSON-serializable key identifying the value.
        t : Type[T]
            Expected type of the stored value, checked against the cached
            value (generic types are checked against their origin, e.g.
            ``list`` for ``list[int]``).

        Returns
        -------
        Optional[T]
            The cached value, or None if not present.

        Raises
        ------
        TypeError
            If a cached value is found but is not an instance of ``t``
            (or its generic origin).
        """
        if self is _NoCache:
            return None
        a = self.backend.get(hash_anything(key))

        if a is not None:
            origin = typing.get_origin(t)
            # if T is 'list[int]', then check origin = list
            # for non generic types, origin is None
            if (origin is not None and not isinstance(a, origin)) or (
                origin is None and not isinstance(a, t)
            ):
                raise TypeError(
                    f"object (value: {a}, type: {type(a)} is not of type {t}."
                )

        return a

    def get_or_put(
        self, key: Hashable, t: Type[T], clb: Callable[..., T]
    ) -> Optional[T]:
        """Return the cached value for ``key``, computing and storing it if missing.

        Parameters
        ----------
        key : Hashable
            JSON-serializable key identifying the value.
        t : Type[T]
            Expected type of the stored value (see :meth:`get`).
        clb : Callable[..., T]
            Callback invoked with no arguments to compute the value when it
            is not already cached. Its result is not stored if it is None.

        Returns
        -------
        Optional[T]
            The cached (or freshly computed) value, or None if ``clb``
            returned None.
        """
        if (value := self.get(key, t)) is None:
            value = clb()
            if value is not None:
                self.put(key, value)
        return value


_NoCache = Cache(backend=NoCacheBackend())
