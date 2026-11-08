"""OpenAlex REST client for the Topics list (T25, D17).

:class:`OpenAlexTopicsClient` reads ``<base_url>/topics`` with page-number paging: the first
request asks for ``page=1``, and the ``meta.count`` it reports fixes how many pages follow, one
per ``per-page`` topics, so a full last page closes the listing without an empty request. Every
request asks for ``per-page`` topics. The topic records are kept as the API serves them, in
order; only their ``id`` is read here. Cursor paging is not used: on 2026-11-05 OpenAlex
answered the anonymous request for a cursor's second page with HTTP 401.

It sends no personal data: the query holds ``per-page`` and ``page`` only, with no email,
``mailto`` parameter or API key; the headers hold the User-Agent that names the project tool,
:data:`USER_AGENT`, and ``Accept: application/json``; and any cookie that a response sets is
dropped before the next request. It stays polite to the server, as the repository harvest does
(:mod:`thematic_redundancy.corpus.dspace`):

- one request at a time, each starting at least ``request_interval_seconds`` after the previous
  one began;
- a timeout of ``timeout_seconds`` on each step of every request;
- at most ``max_retries`` retries after HTTP 429, HTTP 5xx, a timeout or a network error, the
  first after ``backoff_seconds`` and each later one after twice the wait before it, or after
  the ``Retry-After`` that the server asks for when that is longer, up to one minute;
- no retry after HTTP 401 or 403. Both raise :class:`AccessRefusedError`, and so does HTTP 429
  once the retries are spent: the API may then want a key or a lower rate, and sending a
  credential needs the user's authorization (U8);
- no redirect followed.

Each page is checked before use: a JSON object with ``meta.count``, the same ``meta.count`` as
the first page, and at most ``per-page`` results, each an object with a text ``id``. The
download (:mod:`thematic_redundancy.reference.openalex_topics`) then checks that the records
number ``meta.count`` and that every ``id`` is unique, which also shows that the server honoured
``page``: one that ignored it would serve the first page again.
"""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from thematic_redundancy.corpus.dspace import USER_AGENT
from thematic_redundancy.shared.config import OpenAlexConfig

TOPICS_PATH = "topics"
FIRST_PAGE = 1
"""Number of the first page; OpenAlex numbers pages from 1."""

MAX_PER_PAGE = 200
"""Most results that OpenAlex serves per page."""

MIN_REQUEST_INTERVAL_SECONDS = 1.0
"""Shortest pace allowed, however the client is built: at most one request per second."""

MAX_RETRY_AFTER_SECONDS = 60.0
"""Longest ``Retry-After`` honoured; a server that asks for a longer wait ends the download."""

_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
_REFUSED_STATUSES = frozenset({401, 403})
"""Refusals that asking again would not change: the API wants a credential or denies access."""
_TOO_MANY_REQUESTS = 429
_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
_SHOWN_VALIDATION_ERRORS = 3


class OpenAlexError(RuntimeError):
    """The API failed, or answered with something other than the pages of the Topics list."""


class AccessRefusedError(OpenAlexError):
    """The API refused the anonymous request: HTTP 401 or 403, or HTTP 429 after the retries."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class TopicsListing:
    """Every topic record that the listing gave, in order, and how it went."""

    records: tuple[dict[str, Any], ...]
    """The topic records exactly as the API served them."""
    meta_count: int
    """The ``meta.count`` of the first page: how many topics the API said it holds."""
    pages: int
    """Pages read: the ones that ``meta.count`` needs at ``per-page``, and at least one."""


def topics_url(base_url: str) -> str:
    """Return the Topics endpoint under ``base_url``, which may end with a slash or not."""
    return f"{base_url.rstrip('/')}/{TOPICS_PATH}"


class _RetryableFailure(Exception):
    """A request failed in a way that may pass: HTTP 429 or 5xx, a timeout, a lost connection."""

    def __init__(
        self, description: str, *, status_code: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(description)
        self.status_code = status_code
        self.retry_after = retry_after


class _Response(BaseModel):
    """Read-only view of part of a response; it ignores unknown keys and never quotes them."""

    model_config = ConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)


class _Meta(_Response):
    count: Annotated[StrictInt, Field(ge=0)]


class _Topic(_Response):
    id: Annotated[str, Field(min_length=1)]


class _Page(_Response):
    meta: _Meta
    results: tuple[_Topic, ...]


class OpenAlexTopicsClient:
    """Reads the OpenAlex Topics list through ``client``, with one pace and one retry policy."""

    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
        *,
        per_page: int = MAX_PER_PAGE,
        request_interval_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Send requests through ``client``; ``clock`` and ``sleeper`` replace real time in
        tests.

        Raises:
            ValueError: if ``per_page`` is not 1 to :data:`MAX_PER_PAGE`, ``max_retries`` is
                negative, ``request_interval_seconds`` is below
                :data:`MIN_REQUEST_INTERVAL_SECONDS`, or a time is not positive.
        """
        if not 1 <= per_page <= MAX_PER_PAGE:
            raise ValueError(f"per_page must be 1 to {MAX_PER_PAGE}, got {per_page}")
        if max_retries < 0:
            raise ValueError(f"max_retries must not be negative, got {max_retries}")
        if request_interval_seconds < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError(
                f"request_interval_seconds must be at least {MIN_REQUEST_INTERVAL_SECONDS:g}, "
                f"got {request_interval_seconds:g}"
            )
        if timeout_seconds <= 0 or backoff_seconds <= 0:
            raise ValueError(
                f"timeout_seconds and backoff_seconds must be positive, got {timeout_seconds:g} "
                f"and {backoff_seconds:g}"
            )
        self._client = client
        self._url = topics_url(base_url)
        self._per_page = per_page
        self._request_interval = request_interval_seconds
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_retries = max_retries
        self._backoff_base = backoff_seconds
        self._clock = clock
        self._sleep = sleeper
        self._last_request_at: float | None = None
        self.request_count = 0
        """Requests sent so far, retries included."""

    @classmethod
    def from_config(
        cls,
        client: httpx.Client,
        config: OpenAlexConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> Self:
        """Build the client for the API, page size, pace, timeout and retries of ``config``."""
        return cls(
            client,
            str(config.base_url),
            per_page=config.per_page,
            request_interval_seconds=config.request_interval_seconds,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            backoff_seconds=config.backoff_seconds,
            clock=clock,
            sleeper=sleeper,
        )

    @property
    def url(self) -> str:
        """The Topics endpoint that every request goes to."""
        return self._url

    def first_parameters(self) -> dict[str, str]:
        """Return the query of the first request; each later one asks for the next ``page``."""
        return self._parameters(FIRST_PAGE)

    def list_topics(self) -> TopicsListing:
        """Read every page of the Topics list and return its records, in order.

        The first page's ``meta.count`` fixes the pages read: ``ceil(meta.count / per-page)``,
        and at least one.

        Raises:
            AccessRefusedError: if the API answers HTTP 401 or 403, or HTTP 429 after the
                retries.
            OpenAlexError: if a request fails after its retries, or a page is not what the
                module description says.
        """
        payload, first = self._read_page(FIRST_PAGE)
        meta_count = first.meta.count
        last_page = max(FIRST_PAGE, math.ceil(meta_count / self._per_page))
        records = self._results(FIRST_PAGE, payload, first)
        for number in range(FIRST_PAGE + 1, last_page + 1):
            payload, listed = self._read_page(number)
            if listed.meta.count != meta_count:
                raise OpenAlexError(
                    f"page {number} of {self._url} reports meta.count {listed.meta.count}, but "
                    f"page 1 reported {meta_count}; the list changed during the download, so "
                    "run it again"
                )
            records.extend(self._results(number, payload, listed))
        return TopicsListing(records=tuple(records), meta_count=meta_count, pages=last_page)

    def _results(self, number: int, payload: dict[str, Any], listed: _Page) -> list[Any]:
        """Return the records of page ``number`` once it holds no more than ``per-page``."""
        if len(listed.results) > self._per_page:
            raise OpenAlexError(
                f"page {number} of {self._url} holds {len(listed.results)} topics, more than "
                f"the {self._per_page} asked for per page"
            )
        return list(payload["results"])

    def _parameters(self, number: int) -> dict[str, str]:
        return {"per-page": str(self._per_page), "page": str(number)}

    def _read_page(self, number: int) -> tuple[dict[str, Any], _Page]:
        """Fetch page ``number``; return its payload and its checked view."""
        requested = httpx.URL(self._url, params=self._parameters(number))
        payload = self._get_json(requested)
        try:
            listed = _Page.model_validate(payload)
        except ValidationError as error:
            raise OpenAlexError(
                f"unexpected response from GET {requested}: {_summarize(error)}"
            ) from error
        return payload, listed

    def _get_json(self, requested: httpx.URL) -> Any:
        """GET ``requested`` and return its decoded JSON body, retrying transient failures."""
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                body = self._send(requested)
                break
            except _RetryableFailure as failure:
                if attempt == attempts - 1:
                    raise self._given_up(requested, failure, attempts) from failure
                self._sleep(self._backoff(requested, attempt, failure))
        try:
            return json.loads(body)
        except ValueError as error:  # Malformed JSON, or a body that is not UTF-8 text.
            raise OpenAlexError(f"GET {requested} returned a body that is not JSON") from error

    def _send(self, requested: httpx.URL) -> bytes:
        """Send one GET and return the body of its HTTP 200 response.

        Raises:
            _RetryableFailure: for HTTP 429 or 5xx, a timeout or a network error.
        """
        self._wait_for_turn()
        self.request_count += 1
        self._client.cookies.clear()  # No cookie that an earlier response set goes back.
        try:
            with self._client.stream(
                "GET", requested, headers=_HEADERS, timeout=self._timeout
            ) as response:
                status = response.status_code
                if status == httpx.codes.OK:
                    return response.read()
                if status in _REFUSED_STATUSES:
                    raise AccessRefusedError(_refusal(requested, status), status_code=status)
                if status == _TOO_MANY_REQUESTS or 500 <= status <= 599:
                    raise _RetryableFailure(
                        f"HTTP {status}",
                        status_code=status,
                        retry_after=_retry_after_seconds(response),
                    )
                redirect = "; this client follows no redirect" if response.is_redirect else ""
                raise OpenAlexError(f"GET {requested} returned HTTP {status}{redirect}")
        except _TRANSIENT_ERRORS as error:
            raise _RetryableFailure(f"{type(error).__name__}: {error}") from error
        except httpx.HTTPError as error:
            raise OpenAlexError(
                f"GET {requested} failed: {type(error).__name__}: {error}"
            ) from error

    def _given_up(
        self, requested: httpx.URL, failure: _RetryableFailure, attempts: int
    ) -> OpenAlexError:
        """Return the error that ends the download once ``failure`` outlived the retries."""
        plural = "s" if attempts > 1 else ""
        if failure.status_code == _TOO_MANY_REQUESTS:
            return AccessRefusedError(
                _refusal(requested, _TOO_MANY_REQUESTS, f" after {attempts} attempt{plural}"),
                status_code=_TOO_MANY_REQUESTS,
            )
        return OpenAlexError(f"GET {requested} gave up after {attempts} attempt{plural}: {failure}")

    def _wait_for_turn(self) -> None:
        """Sleep until ``request_interval_seconds`` have passed since the last request began."""
        if self._last_request_at is not None:
            wait = self._request_interval - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _backoff(self, requested: httpx.URL, attempt: int, failure: _RetryableFailure) -> float:
        """Return the wait before retry ``attempt + 1``, at least what ``Retry-After`` asks."""
        wait = self._backoff_base * 2**attempt
        if failure.retry_after is None:
            return wait
        if failure.retry_after > MAX_RETRY_AFTER_SECONDS:
            longer = (
                f" with Retry-After {failure.retry_after:g} s, longer than the "
                f"{MAX_RETRY_AFTER_SECONDS:g} s this client waits"
            )
            if failure.status_code == _TOO_MANY_REQUESTS:
                raise AccessRefusedError(
                    _refusal(requested, _TOO_MANY_REQUESTS, longer),
                    status_code=_TOO_MANY_REQUESTS,
                )
            raise OpenAlexError(f"GET {requested} got {failure}{longer}; try again later")
        return max(wait, failure.retry_after)


def _refusal(requested: httpx.URL, status: int, detail: str = "") -> str:
    """Describe a refusal by its status, and why the download stops there."""
    return (
        f"GET {requested} returned HTTP {status}{detail}: OpenAlex refused the anonymous request. "
        "The download sends no email, API key or cookie, and using a credential or another "
        "rate needs the user's authorization (U8), so it stops here"
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the wait that ``Retry-After`` asks for, or ``None`` when there is none.

    Only the delta-seconds form is read; an HTTP date leaves the usual backoff in place.
    """
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if value.isascii() and value.isdigit() else None


def _summarize(error: ValidationError) -> str:
    """Describe a validation error on one line, by location and problem, without input."""
    details = [
        f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
        for detail in error.errors(include_url=False, include_input=False)
    ]
    shown = "; ".join(details[:_SHOWN_VALIDATION_ERRORS])
    hidden = len(details) - _SHOWN_VALIDATION_ERRORS
    return f"{shown}; and {hidden} more" if hidden > 0 else shown
