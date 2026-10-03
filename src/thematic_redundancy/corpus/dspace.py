"""DSpace 7 REST adapter for the repository ports of :mod:`thematic_redundancy.corpus.repository`.

As an :class:`~thematic_redundancy.corpus.repository.ItemRepository`, it searches
``<base_url>/server/api/discover/search/objects`` for the items under one collection or
community whose ``dc.date.issued`` year falls in a range, sorted by that date, and reads the
pages until the reported total is covered. As a
:class:`~thematic_redundancy.corpus.repository.BitstreamRepository`, it reads an item's
bundles and files in one request (``core/items/<uuid>?embed=bundles/bitstreams``) and streams
a file from ``core/bitstreams/<uuid>/content``.

Anonymous read access is enough, so it sends no credentials. It stays polite to the server:

- one request at a time, each starting at least ``request_interval_seconds`` after the
  previous one began, redirects included;
- a User-Agent that names the research project, and a timeout on every request;
- a bounded number of retries after HTTP 429, HTTP 5xx, a timeout or a network error, with
  exponential backoff that honours a ``Retry-After`` of up to one minute. A download that
  fails midway starts over from its first byte;
- no retry after HTTP 401, 403 or 404, which raise
  :class:`~thematic_redundancy.corpus.repository.ResourceUnavailableError`;
- redirects only within the repository's own scheme, host and port.

Each response is checked before use: its shape, the scope and year filter that the server
says it applied, the page number, paging that stays the same from page to page, the item
that a listing describes, and embedded lists that hold every element. Error messages quote
neither metadata values nor file names: a malformed response is described by the location
and kind of each problem, and a refused redirect by its scheme, host and port.
"""

import json
import time
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, Self
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, ValidationError

from thematic_redundancy.corpus.repository import (
    Bitstream,
    Bundle,
    ByteSink,
    Checksum,
    IssuedYears,
    RepositoryError,
    RepositoryItem,
    ResourceUnavailableError,
    ScopeListing,
    SearchDescription,
)
from thematic_redundancy.shared.config import RepositoryConfig

API_PATH = "server/api"
SEARCH_PATH = "discover/search/objects"
ITEMS_PATH = "core/items"
BITSTREAMS_PATH = "core/bitstreams"

FILES_EMBED = "bundles/bitstreams"
"""Projection that embeds an item's bundles, and the files of each, in the item itself."""

USER_AGENT = "thematic-redundancy-research/0.1 (academic thesis research)"
"""Names the project, and no person, so that the repository's operators can identify it."""

SORT = "dc.date.issued,ASC"
ISSUED_YEAR_FILTER = "dateIssued"
"""Discovery filter on the year of ``dc.date.issued``."""

REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

MIN_REQUEST_INTERVAL_SECONDS = 1.0
"""Shortest pace allowed, however the adapter is built: at most one request per second."""

BACKOFF_BASE_SECONDS = 2.0
"""Wait before the first retry; each later retry waits twice as long as the one before."""

MAX_RETRY_AFTER_SECONDS = 60.0
"""Longest ``Retry-After`` honoured; a server that asks for a longer wait ends the request."""

MAX_PAGES_PER_SEARCH = 100
"""Most pages that one search may take; more suggests a wrong scope or a filter not applied."""

MAX_REDIRECTS = 3
"""Most redirects that one request follows; each must stay on the repository's host."""

DOWNLOAD_CHUNK_BYTES = 64 * 1024

_JSON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
_CONTENT_HEADERS = {"User-Agent": USER_AGENT, "Accept": "*/*"}
_FINAL_STATUSES = frozenset({401, 403, 404})
"""Refusals that asking again would not change: no access, or no such resource."""
_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
_SHOWN_VALIDATION_ERRORS = 3


def search_url(base_url: str) -> str:
    """Return the search endpoint under ``base_url``, which may end with a slash or not."""
    return f"{api_url(base_url)}/{SEARCH_PATH}"


def api_url(base_url: str) -> str:
    """Return the REST API root under ``base_url``, which may end with a slash or not."""
    return f"{base_url.rstrip('/')}/{API_PATH}"


class _RetryableFailure(Exception):
    """A request failed in a way that may pass: HTTP 429 or 5xx, a timeout, a lost connection."""

    def __init__(self, description: str, retry_after: float | None = None) -> None:
        super().__init__(description)
        self.retry_after = retry_after


class _Response(BaseModel):
    """Read-only view of part of a response; it ignores unknown keys.

    Errors never quote the input, because a response carries DNI fields until the harvest
    drops them, and file names that may hold author names.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)


class _Page(_Response):
    number: Annotated[StrictInt, Field(ge=0)]
    total_pages: Annotated[StrictInt, Field(ge=0, alias="totalPages")]
    total_elements: Annotated[StrictInt, Field(ge=0, alias="totalElements")]


class _Item(_Response):
    type: Literal["item"]
    uuid: UUID
    handle: str | None
    metadata: dict[str, list[dict[str, JsonValue]]]


class _ObjectEmbedded(_Response):
    indexable_object: _Item = Field(alias="indexableObject")


class _SearchObject(_Response):
    embedded: _ObjectEmbedded = Field(alias="_embedded")


class _ResultEmbedded(_Response):
    objects: tuple[_SearchObject, ...] = ()


class _SearchResult(_Response):
    embedded: _ResultEmbedded = Field(alias="_embedded")
    page: _Page


class _ResponseEmbedded(_Response):
    search_result: _SearchResult = Field(alias="searchResult")


class _AppliedFilter(_Response):
    filter: str
    operator: str
    value: str


class _SearchResponse(_Response):
    scope: str | None
    applied_filters: tuple[_AppliedFilter, ...] = Field(alias="appliedFilters")
    embedded: _ResponseEmbedded = Field(alias="_embedded")

    @property
    def page(self) -> _Page:
        return self.embedded.search_result.page

    def items(self) -> list[RepositoryItem]:
        """Return the items of this page, in the order that the repository listed them."""
        items = []
        for listed in self.embedded.search_result.embedded.objects:
            item = listed.embedded.indexable_object
            items.append(RepositoryItem(uuid=item.uuid, handle=item.handle, metadata=item.metadata))
        return items


class _Checksum(_Response):
    algorithm: str = Field(alias="checkSumAlgorithm", min_length=1)
    value: str = Field(min_length=1)


class _Bitstream(_Response):
    type: Literal["bitstream"]
    uuid: UUID
    name: str
    size_bytes: Annotated[StrictInt, Field(ge=0, alias="sizeBytes")]
    checksum: _Checksum = Field(alias="checkSum")

    def as_bitstream(self) -> Bitstream:
        checksum = Checksum(algorithm=self.checksum.algorithm, value=self.checksum.value)
        return Bitstream(
            uuid=self.uuid, name=self.name, size_bytes=self.size_bytes, checksum=checksum
        )


class _BitstreamsEmbedded(_Response):
    bitstreams: tuple[_Bitstream, ...] = ()


class _BitstreamList(_Response):
    embedded: _BitstreamsEmbedded = Field(alias="_embedded")
    page: _Page


class _BundleEmbedded(_Response):
    bitstreams: _BitstreamList


class _Bundle(_Response):
    type: Literal["bundle"]
    name: str
    embedded: _BundleEmbedded = Field(alias="_embedded")


class _BundlesEmbedded(_Response):
    bundles: tuple[_Bundle, ...] = ()


class _BundleList(_Response):
    embedded: _BundlesEmbedded = Field(alias="_embedded")
    page: _Page


class _ItemEmbedded(_Response):
    bundles: _BundleList


class _ItemFiles(_Response):
    """An item with its bundles, and the files of each, embedded."""

    type: Literal["item"]
    uuid: UUID
    embedded: _ItemEmbedded = Field(alias="_embedded")

    def bundles(self) -> tuple[Bundle, ...]:
        """Return the bundles and their files, refusing any embedded list cut short.

        An embedded list holds one page of its elements; a longer list would need paging
        that this listing does not do.
        """
        listed = self.embedded.bundles
        shown = len(listed.embedded.bundles)
        if shown != listed.page.total_elements:
            raise RepositoryError(
                f"the response for item {self.uuid} embeds {shown} of its "
                f"{listed.page.total_elements} bundles; one embedded page cannot hold them all"
            )
        bundles = []
        for bundle in listed.embedded.bundles:
            files = bundle.embedded.bitstreams
            shown = len(files.embedded.bitstreams)
            if shown != files.page.total_elements:
                raise RepositoryError(
                    f"the response for item {self.uuid} embeds {shown} of the "
                    f"{files.page.total_elements} files of bundle {bundle.name}; one embedded "
                    "page cannot hold them all"
                )
            bitstreams = tuple(file.as_bitstream() for file in files.embedded.bitstreams)
            bundles.append(Bundle(name=bundle.name, bitstreams=bitstreams))
        return tuple(bundles)


class DSpaceRestRepository:
    """:class:`ItemRepository` and :class:`BitstreamRepository` over the REST API of a
    DSpace 7 repository, sharing one pace and one retry policy."""

    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
        *,
        page_size: int = 100,
        request_interval_seconds: float = 1.0,
        max_retries: int = 3,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Send requests through ``client``; ``clock`` and ``sleeper`` replace real time in
        tests.

        Raises:
            ValueError: if ``page_size`` is below 1, ``max_retries`` below 0, or
                ``request_interval_seconds`` below :data:`MIN_REQUEST_INTERVAL_SECONDS`.
        """
        if page_size < 1:
            raise ValueError(f"page_size must be at least 1, got {page_size}")
        if max_retries < 0:
            raise ValueError(f"max_retries must not be negative, got {max_retries}")
        if request_interval_seconds < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError(
                f"request_interval_seconds must be at least {MIN_REQUEST_INTERVAL_SECONDS:g}, "
                f"got {request_interval_seconds:g}"
            )
        self._client = client
        self._api_url = api_url(base_url)
        self._search_url = search_url(base_url)
        self._origin = _origin_of(httpx.URL(base_url))
        self._page_size = page_size
        self._request_interval = request_interval_seconds
        self._max_retries = max_retries
        self._clock = clock
        self._sleep = sleeper
        self._last_request_at: float | None = None
        self.request_count = 0
        """Requests sent so far, retries included."""

    @classmethod
    def from_config(
        cls,
        client: httpx.Client,
        config: RepositoryConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> Self:
        """Build the adapter for the repository, page size, pace and retries of ``config``."""
        return cls(
            client,
            str(config.base_url),
            page_size=config.page_size,
            request_interval_seconds=config.request_interval_seconds,
            max_retries=config.max_retries,
            clock=clock,
            sleeper=sleeper,
        )

    def describe_search(self, years: IssuedYears) -> SearchDescription:
        return SearchDescription(
            search_url=self._search_url, query_parameters=self._parameters(years, self._page_size)
        )

    def list_items(self, scope: UUID, years: IssuedYears) -> ScopeListing:
        first = self._search(scope, years, page=0, size=self._page_size)
        paging = first.page
        if paging.total_pages > MAX_PAGES_PER_SEARCH:
            raise RepositoryError(
                f"the search under scope {scope} needs {paging.total_pages} pages, more than "
                f"the {MAX_PAGES_PER_SEARCH} allowed; check the scope and the year filter, "
                "or raise repository.page_size"
            )
        items = first.items()
        for number in range(1, paging.total_pages):
            response = self._search(scope, years, page=number, size=self._page_size)
            _require_same_paging(response.page, paging, scope)
            items.extend(response.items())
        return ScopeListing(items=tuple(items), reported_total=paging.total_elements)

    def count_items(self, scope: UUID, years: IssuedYears) -> int:
        return self._search(scope, years, page=0, size=1).page.total_elements

    def list_bundles(self, item: UUID) -> tuple[Bundle, ...]:
        # The slash stays unescaped in the query, as the DSpace web client sends it.
        url = f"{self._api_url}/{ITEMS_PATH}/{item}?embed={FILES_EMBED}"
        payload = self._get_json(url)
        try:
            listed = _ItemFiles.model_validate(payload)
        except ValidationError as error:
            raise RepositoryError(
                f"unexpected item response from {url}: {_summarize(error)}"
            ) from error
        if listed.uuid != item:
            raise RepositoryError(
                f"asked for item {item}, but the repository answered with another item "
                f"({listed.uuid})"
            )
        return listed.bundles()

    def download(self, bitstream: UUID, sink: ByteSink) -> None:
        def stream_into_sink(response: httpx.Response) -> None:
            sink.restart()
            for chunk in response.iter_bytes(DOWNLOAD_CHUNK_BYTES):
                sink.write(chunk)

        url = f"{self._api_url}/{BITSTREAMS_PATH}/{bitstream}/content"
        self._get(url, None, _CONTENT_HEADERS, stream_into_sink)

    def _parameters(self, years: IssuedYears, size: int) -> dict[str, str]:
        """Return the query parameters that every search over ``years`` sends."""
        return {
            "dsoType": "ITEM",
            "size": str(size),
            "sort": SORT,
            "f.dateIssued": f"{_year_range(years)},equals",
        }

    def _search(self, scope: UUID, years: IssuedYears, *, page: int, size: int) -> _SearchResponse:
        """Fetch one page of the search under ``scope`` and check that it answers the query."""
        parameters = {"scope": str(scope), **self._parameters(years, size), "page": str(page)}
        payload = self._get_json(self._search_url, parameters)
        try:
            response = _SearchResponse.model_validate(payload)
        except ValidationError as error:
            raise RepositoryError(
                f"unexpected search response from {self._search_url} for scope {scope}: "
                f"{_summarize(error)}"
            ) from error
        if response.scope != str(scope):
            raise RepositoryError(
                f"the search under scope {scope} came back for scope {response.scope}; "
                "the repository ignored the scope"
            )
        expected = _year_range(years)
        if not any(
            applied.filter == ISSUED_YEAR_FILTER
            and applied.operator == "equals"
            and applied.value == expected
            for applied in response.applied_filters
        ):
            raise RepositoryError(
                f"the search under scope {scope} came back without the issue-year filter "
                f"{expected}; the repository ignored the filter"
            )
        if response.page.number != page:
            raise RepositoryError(
                f"asked for page {page} of the search under scope {scope}, but the repository "
                f"answered with page {response.page.number}"
            )
        return response

    def _get_json(self, url: str, parameters: Mapping[str, str] | None = None) -> Any:
        """GET ``url`` with ``parameters`` and return the decoded JSON body."""
        body = self._get(url, parameters, _JSON_HEADERS, httpx.Response.read)
        try:
            return json.loads(body)
        except ValueError as error:  # Malformed JSON, or a body that is not UTF-8 text.
            raise RepositoryError(
                f"GET {_with_parameters(url, parameters)} returned a body that is not JSON"
            ) from error

    def _get[T](
        self,
        url: str,
        parameters: Mapping[str, str] | None,
        headers: Mapping[str, str],
        consume: Callable[[httpx.Response], T],
    ) -> T:
        """GET ``url`` and return what ``consume`` makes of the HTTP 200 response.

        A timeout, a network error, HTTP 429 or HTTP 5xx is retried up to ``max_retries``
        times, and so is such a failure while ``consume`` reads the body; ``consume`` then
        runs again on the new response. HTTP 401, 403 and 404 raise
        :class:`ResourceUnavailableError`. Every other failure, and any other error that
        ``consume`` raises, ends the request at once.
        """
        for attempt in range(self._max_retries):
            try:
                return self._send(url, parameters, headers, consume)
            except _RetryableFailure as failure:
                self._sleep(self._backoff(url, attempt, failure))
        try:
            return self._send(url, parameters, headers, consume)
        except _RetryableFailure as failure:
            attempts = self._max_retries + 1
            plural = "s" if attempts > 1 else ""
            raise RepositoryError(
                f"GET {url} gave up after {attempts} attempt{plural}: {failure}"
            ) from failure

    def _send[T](
        self,
        url: str,
        parameters: Mapping[str, str] | None,
        headers: Mapping[str, str],
        consume: Callable[[httpx.Response], T],
    ) -> T:
        """Send one GET, follow its redirects within the repository, and consume HTTP 200.

        Each redirect is a request of its own, paced and counted like any other. Messages
        name the requested URL only, never a redirect target, whose path may hold a file name.

        Raises:
            _RetryableFailure: for HTTP 429 or 5xx, a timeout or a network error.
        """
        requested = _with_parameters(url, parameters)
        target = requested
        for _ in range(MAX_REDIRECTS + 1):
            self._wait_for_turn()
            self.request_count += 1
            try:
                with self._client.stream(
                    "GET", target, headers=headers, timeout=REQUEST_TIMEOUT
                ) as response:
                    status = response.status_code
                    if status == httpx.codes.OK:
                        return consume(response)
                    if response.has_redirect_location:
                        target = self._redirect_target(requested, response)
                        continue
                    if status in _FINAL_STATUSES:
                        raise ResourceUnavailableError(
                            f"GET {requested} returned HTTP {status}", status_code=status
                        )
                    if _is_transient(status):
                        raise _RetryableFailure(f"HTTP {status}", _retry_after_seconds(response))
                    raise RepositoryError(f"GET {requested} returned HTTP {status}")
            except _TRANSIENT_ERRORS as error:
                raise _RetryableFailure(f"{type(error).__name__}: {error}") from error
            except httpx.HTTPError as error:
                raise RepositoryError(
                    f"GET {requested} failed: {type(error).__name__}: {error}"
                ) from error
        raise RepositoryError(f"GET {requested} was redirected more than {MAX_REDIRECTS} times")

    def _redirect_target(self, requested: httpx.URL, response: httpx.Response) -> httpx.URL:
        """Return where ``response`` redirects to, refusing any place off the repository."""
        try:
            target = response.url.join(response.headers["Location"])
        except httpx.InvalidURL as error:
            raise RepositoryError(f"GET {requested} was redirected to an invalid URL") from error
        if _origin_of(target) != self._origin:
            raise RepositoryError(
                f"GET {requested} was redirected to {_describe_origin(_origin_of(target))}, "
                f"outside {_describe_origin(self._origin)}; requests stay on the repository"
            )
        return target

    def _wait_for_turn(self) -> None:
        """Sleep until ``request_interval_seconds`` have passed since the last request began."""
        if self._last_request_at is not None:
            wait = self._request_interval - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _backoff(self, url: str, attempt: int, failure: _RetryableFailure) -> float:
        """Return the wait before retry ``attempt + 1``, at least what ``Retry-After`` asks."""
        wait = BACKOFF_BASE_SECONDS * 2**attempt
        if failure.retry_after is None:
            return wait
        if failure.retry_after > MAX_RETRY_AFTER_SECONDS:
            raise RepositoryError(
                f"GET {url} got {failure} with Retry-After {failure.retry_after:g} s, "
                f"longer than the {MAX_RETRY_AFTER_SECONDS:g} s this client waits; "
                "try again later"
            )
        return max(wait, failure.retry_after)


def _with_parameters(url: str, parameters: Mapping[str, str] | None) -> httpx.URL:
    """Return ``url`` with ``parameters`` as its query; without them, keep its own query."""
    return httpx.URL(url) if parameters is None else httpx.URL(url, params=parameters)


_Origin = tuple[str, str, int | None]
"""Scheme, host and explicit port of a URL; a default port is ``None``."""


def _origin_of(url: httpx.URL) -> _Origin:
    return (url.scheme, url.host, url.port)


def _describe_origin(origin: _Origin) -> str:
    scheme, host, port = origin
    return f"{scheme}://{host}" if port is None else f"{scheme}://{host}:{port}"


def _year_range(years: IssuedYears) -> str:
    return f"[{years.first} TO {years.last}]"


def _is_transient(status_code: int) -> bool:
    return status_code == httpx.codes.TOO_MANY_REQUESTS or 500 <= status_code <= 599


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the wait that ``Retry-After`` asks for, or ``None`` when there is none.

    Only the delta-seconds form is read; an HTTP date leaves the usual backoff in place.
    """
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if value.isascii() and value.isdigit() else None


def _require_same_paging(paging: _Page, first: _Page, scope: UUID) -> None:
    """Refuse a later page whose totals differ from those of page 0."""
    if (paging.total_elements, paging.total_pages) != (first.total_elements, first.total_pages):
        raise RepositoryError(
            f"page {paging.number} of the search under scope {scope} reports "
            f"{paging.total_elements} items on {paging.total_pages} pages, but page 0 reported "
            f"{first.total_elements} items on {first.total_pages} pages; the repository "
            "changed during the harvest, so run it again"
        )


def _summarize(error: ValidationError) -> str:
    """Describe a validation error on one line, by location and problem, without input."""
    details = [
        f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
        for detail in error.errors(include_url=False, include_input=False)
    ]
    shown = "; ".join(details[:_SHOWN_VALIDATION_ERRORS])
    hidden = len(details) - _SHOWN_VALIDATION_ERRORS
    return f"{shown}; and {hidden} more" if hidden > 0 else shown
