"""Port to the institutional repository: read-only item searches and the data they return.

The harvest use case depends only on this module. The DSpace REST adapter in
:mod:`thematic_redundancy.corpus.dspace` implements :class:`ItemRepository` over HTTP, and
tests replace it with an in-memory fake.
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
