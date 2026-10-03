"""Port to the institutional repository: read-only item searches, item files, and their data.

The harvest and the PDF download use cases depend only on this module. The DSpace REST
adapter in :mod:`thematic_redundancy.corpus.dspace` implements :class:`ItemRepository` and
:class:`BitstreamRepository` over HTTP, and tests replace it with in-memory fakes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from pydantic import JsonValue

MetadataEntry = Mapping[str, JsonValue]
"""One value of a metadata field: ``value`` plus attributes such as ``language`` and ``place``."""

Metadata = Mapping[str, Sequence[MetadataEntry]]
"""Metadata of an item: the values of each field, such as ``dc.title``, in repository order."""


class RepositoryError(RuntimeError):
    """The repository could not be reached, refused a search, or answered unexpectedly."""


class ResourceUnavailableError(RepositoryError):
    """The repository answered HTTP 401, 403 or 404, which asking again would not change.

    401 and 403 mean that anonymous users may not read the resource, as with embargoed or
    restricted files; 404 means that it does not exist, or no longer does.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class IssuedYears:
    """Inclusive range of ``dc.date.issued`` years that a search keeps."""

    first: int
    last: int

    def __post_init__(self) -> None:
        if self.first > self.last:
            raise ValueError(f"first year ({self.first}) must not be after last year ({self.last})")


@dataclass(frozen=True, slots=True)
class RepositoryItem:
    """One archived item, as the repository describes it."""

    uuid: UUID
    handle: str | None
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class ScopeListing:
    """Every item that one search over a collection or a community returned."""

    items: tuple[RepositoryItem, ...]
    reported_total: int
    """Number of matching items that the repository reported for the search."""


@dataclass(frozen=True, slots=True)
class SearchDescription:
    """Where the searches go and which query parameters they share, for provenance."""

    search_url: str
    query_parameters: Mapping[str, str]
    """Parameters that every search sends, without the scope and the page, which vary."""


class ItemRepository(Protocol):
    """Read-only search for the items under one collection or community of a repository."""

    def describe_search(self, years: IssuedYears) -> SearchDescription:
        """Return the endpoint and the shared query parameters of searches over ``years``."""
        ...

    def list_items(self, scope: UUID, years: IssuedYears) -> ScopeListing:
        """Return every item under ``scope`` issued within ``years``, in a stable order."""
        ...

    def count_items(self, scope: UUID, years: IssuedYears) -> int:
        """Return how many items under ``scope`` were issued within ``years``."""
        ...


@dataclass(frozen=True, slots=True)
class Checksum:
    """Digest that the repository keeps for a file."""

    algorithm: str
    """Algorithm name as the repository spells it, such as ``MD5``."""
    value: str
    """Hexadecimal digest."""


@dataclass(frozen=True, slots=True)
class Bitstream:
    """One file of an item, as the repository lists it."""

    uuid: UUID
    name: str
    """Original file name. It may hold author names, so it is never stored."""
    size_bytes: int
    checksum: Checksum


@dataclass(frozen=True, slots=True)
class Bundle:
    """Named group of an item's files, such as ``ORIGINAL``, ``TEXT`` or ``THUMBNAIL``."""

    name: str
    bitstreams: tuple[Bitstream, ...]
    """Files of the bundle, in repository order."""


class ByteSink(Protocol):
    """Destination of a streamed download, such as a file that hashes what it receives."""

    def write(self, data: bytes, /) -> object:
        """Take the next bytes of the file."""
        ...

    def restart(self) -> None:
        """Forget the bytes received so far, because the download starts over."""
        ...


class BitstreamRepository(Protocol):
    """Read-only access to the files of the repository's items."""

    def list_bundles(self, item: UUID) -> tuple[Bundle, ...]:
        """Return the bundles of ``item``, each with every one of its files.

        Raises:
            ResourceUnavailableError: if the repository refuses the item or does not know it.
            RepositoryError: if the repository fails or answers unexpectedly.
        """
        ...

    def download(self, bitstream: UUID, sink: ByteSink) -> None:
        """Stream the content of ``bitstream`` into ``sink``, from its first byte to its last.

        A download that starts over calls :meth:`ByteSink.restart` first. Errors raised by
        ``sink`` end the download and reach the caller unchanged.

        Raises:
            ResourceUnavailableError: if the repository refuses the file or does not know it.
            RepositoryError: if the repository fails or answers unexpectedly.
        """
        ...
