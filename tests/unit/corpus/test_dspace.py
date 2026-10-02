"""Behavior of the DSpace REST adapter: URLs, paging, pacing, retries, and response checks.

``httpx.MockTransport`` replays the synthetic search pages in ``tests/fixtures/dspace``. They
copy the structure of DSpace 7.6 search responses (keys, nesting and paging fields) with
fake values. A fake clock stands in for time, so no test opens a connection or sleeps.
"""

import itertools
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import HttpUrl

from thematic_redundancy.corpus.dspace import (
    MAX_PAGES_PER_SEARCH,
    USER_AGENT,
    DSpaceRestRepository,
)
from thematic_redundancy.corpus.repository import IssuedYears, RepositoryError, SearchDescription
from thematic_redundancy.shared.config import RepositoryConfig

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "dspace"
BASE_URL = "https://repo.example.edu"
SEARCH_URL = "https://repo.example.edu/server/api/discover/search/objects"
COLLECTION = UUID("00000000-0000-4000-8000-0000000000c1")
"""Scope that the synthetic pages answer for."""
FACULTY = UUID("00000000-0000-4000-8000-0000000000f0")
YEARS = IssuedYears(first=2021, last=2026)
LATENCY = 0.25
"""Seconds of fake time that the fake server takes to answer a request."""

Reply = httpx.Response | dict[str, Any] | type[httpx.TransportError]
"""A response, a JSON payload served with HTTP 200, or a transport error class to raise."""


def fixture(name: str) -> dict[str, Any]:
    """Return a fresh copy of one synthetic response, ready to be modified."""
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def page(number: int) -> dict[str, Any]:
    """Return page ``number`` of the synthetic listing: 5 items, 2 per page."""
    return fixture(f"search_page_{number}.json")


def empty_page() -> dict[str, Any]:
    return fixture("search_empty.json")


def result_of(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the ``searchResult`` part of a response, which holds the items and the paging."""
    return payload["_embedded"]["searchResult"]


def first_item(payload: dict[str, Any]) -> dict[str, Any]:
    return result_of(payload)["_embedded"]["objects"][0]["_embedded"]["indexableObject"]


class FakeTime:
    """Monotonic clock plus a sleeper that only moves the clock forward."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeServer:
    """MockTransport handler that answers with ``replies`` in turn and records each request."""

    def __init__(self, time: FakeTime, replies: list[Reply]) -> None:
        self.time = time
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []
        self.started_at: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.started_at.append(self.time.now)
        self.time.now += LATENCY
        reply = self.replies.pop(0)
        if isinstance(reply, type):
            raise reply("simulated transport failure", request=request)
        if isinstance(reply, dict):
            return httpx.Response(200, json=reply)
        return reply

    def sent(self, parameter: str) -> list[str]:
        """Return the value of a query parameter in each request, in order."""
        return [request.url.params[parameter] for request in self.requests]


Connect = Callable[..., tuple[DSpaceRestRepository, FakeServer]]


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def connect(fake_time: FakeTime) -> Iterator[Connect]:
    """Build an adapter over a fake server that gives ``replies``; settings go to the adapter."""
    clients: list[httpx.Client] = []

    def build(
        replies: list[Reply], *, base_url: str = BASE_URL, **settings: Any
    ) -> tuple[DSpaceRestRepository, FakeServer]:
        server = FakeServer(fake_time, replies)
        client = httpx.Client(transport=httpx.MockTransport(server))
        clients.append(client)
        repository = DSpaceRestRepository(
            client, base_url, clock=fake_time.clock, sleeper=fake_time.sleep, **settings
        )
        return repository, server

    yield build
    for client in clients:
        client.close()


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://repo.example.edu", SEARCH_URL),
        ("https://repo.example.edu/", SEARCH_URL),
        (str(HttpUrl("https://repo.example.edu")), SEARCH_URL),
        (
            "https://repo.example.edu/dspace/",
            "https://repo.example.edu/dspace/server/api/discover/search/objects",
        ),
    ],
    ids=["bare", "trailing-slash", "pydantic-normalized", "path-prefix"],
)
def test_searches_go_to_the_api_under_the_base_url_with_single_slashes(
    connect: Connect, base_url: str, expected: str
) -> None:
    repository, server = connect([empty_page()], base_url=base_url)

    repository.list_items(COLLECTION, YEARS)

    assert str(server.requests[0].url).split("?")[0] == expected
    assert repository.describe_search(YEARS).search_url == expected


def test_a_search_asks_for_the_items_of_the_scope_issued_in_the_years_by_date(
    connect: Connect,
) -> None:
    repository, server = connect([empty_page()])

    repository.list_items(COLLECTION, YEARS)

    [request] = server.requests
    assert dict(request.url.params) == {
        "scope": str(COLLECTION),
        "dsoType": "ITEM",
        "size": "100",
        "page": "0",
        "sort": "dc.date.issued,ASC",
        "f.dateIssued": "[2021 TO 2026],equals",
    }
    raw_query = request.url.query.decode("ascii")
    assert "f.dateIssued=%5B2021+TO+2026%5D%2Cequals" in raw_query
    assert not {"[", "]", " "} & set(raw_query)


def test_list_items_follows_every_page_in_order(connect: Connect) -> None:
    repository, server = connect([page(0), page(1), page(2)])

    listing = repository.list_items(COLLECTION, YEARS)

    assert server.sent("page") == ["0", "1", "2"]
    assert listing.reported_total == 5
    assert [str(item.uuid)[-4:] for item in listing.items] == [
        "a001",
        "a002",
        "a003",
        "a004",
        "a005",
    ]
    served = first_item(page(0))
    assert (listing.items[0].handle, listing.items[0].metadata) == (
        served["handle"],
        served["metadata"],
    )


def test_an_empty_search_returns_no_items_after_one_request(connect: Connect) -> None:
    repository, server = connect([empty_page()])

    listing = repository.list_items(COLLECTION, YEARS)

    assert (listing.items, listing.reported_total, len(server.requests)) == ((), 0, 1)


@pytest.mark.parametrize("interval", [1.0, 2.5])
def test_requests_start_at_least_the_request_interval_apart(
    connect: Connect, fake_time: FakeTime, interval: float
) -> None:
    repository, server = connect([page(0), page(1), page(2)], request_interval_seconds=interval)
    start = fake_time.now

    repository.list_items(COLLECTION, YEARS)

    assert server.started_at[0] == start  # The first request does not wait.
    gaps = [later - earlier for earlier, later in itertools.pairwise(server.started_at)]
    assert gaps == pytest.approx([interval, interval])
    assert fake_time.sleeps == pytest.approx([interval - LATENCY, interval - LATENCY])


def test_the_pace_also_holds_between_separate_searches(connect: Connect) -> None:
    faculty_page = page(0)
    faculty_page["scope"] = str(FACULTY)
    repository, server = connect([empty_page(), faculty_page])

    repository.list_items(COLLECTION, YEARS)
    repository.count_items(FACULTY, YEARS)

    assert server.started_at[1] - server.started_at[0] == pytest.approx(1.0)


def test_requests_identify_the_research_project_and_ask_for_json(connect: Connect) -> None:
    repository, server = connect([empty_page()])

    repository.list_items(COLLECTION, YEARS)

    headers = server.requests[0].headers
    assert headers["User-Agent"] == USER_AGENT
    assert USER_AGENT == "thematic-redundancy-research/0.1 (academic thesis research)"
    assert headers["Accept"] == "application/json"
    assert not {"authorization", "cookie"} & {name.lower() for name in headers}


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_statuses_are_retried_with_exponential_backoff(
    connect: Connect, fake_time: FakeTime, status: int
) -> None:
    repository, server = connect([httpx.Response(status), httpx.Response(status), empty_page()])

    listing = repository.list_items(COLLECTION, YEARS)

    assert listing.reported_total == 0
    assert len(server.requests) == 3
    assert fake_time.sleeps == [2.0, 4.0]


@pytest.mark.parametrize(
    ("retry_after", "expected_wait"),
    [("7", 7.0), ("60", 60.0), ("1", 2.0), ("Fri, 02 Oct 2026 15:00:00 GMT", 2.0)],
    ids=["longer-than-backoff", "at-the-limit", "shorter-than-backoff", "http-date"],
)
def test_the_wait_before_a_retry_honours_retry_after_seconds(
    connect: Connect, fake_time: FakeTime, retry_after: str, expected_wait: float
) -> None:
    repository, _ = connect(
        [httpx.Response(429, headers={"Retry-After": retry_after}), empty_page()]
    )

    repository.list_items(COLLECTION, YEARS)

    assert fake_time.sleeps == [expected_wait]


def test_a_retry_after_beyond_the_limit_gives_up_without_waiting(
    connect: Connect, fake_time: FakeTime
) -> None:
    repository, server = connect([httpx.Response(503, headers={"Retry-After": "3600"})])

    with pytest.raises(RepositoryError, match="3600"):
        repository.list_items(COLLECTION, YEARS)

    assert (len(server.requests), fake_time.sleeps) == (1, [])


@pytest.mark.parametrize("max_retries", [0, 1, 3])
def test_a_search_gives_up_after_the_last_retry(
    connect: Connect, fake_time: FakeTime, max_retries: int
) -> None:
    attempts = max_retries + 1
    repository, server = connect([httpx.Response(503)] * attempts, max_retries=max_retries)

    with pytest.raises(RepositoryError, match=rf"after {attempts} attempts?: HTTP 503"):
        repository.list_items(COLLECTION, YEARS)

    assert len(server.requests) == repository.request_count == attempts
    assert fake_time.sleeps == [2.0, 4.0, 8.0][:max_retries]


def test_timeouts_and_network_errors_are_retried(connect: Connect, fake_time: FakeTime) -> None:
    failures = [
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    ]
    repository, _ = connect([*failures, empty_page()], max_retries=len(failures))

    listing = repository.list_items(COLLECTION, YEARS)

    assert listing.reported_total == 0
    assert fake_time.sleeps == [2.0, 4.0, 8.0, 16.0]


def test_a_network_failure_on_every_attempt_names_the_url_and_the_error(connect: Connect) -> None:
    repository, _ = connect([httpx.ReadTimeout, httpx.ReadTimeout], max_retries=1)

    with pytest.raises(RepositoryError, match="ReadTimeout") as caught:
        repository.list_items(COLLECTION, YEARS)

    assert SEARCH_URL in str(caught.value)


@pytest.mark.parametrize("status", [400, 403, 404])
def test_client_errors_are_not_retried(connect: Connect, fake_time: FakeTime, status: int) -> None:
    repository, server = connect([httpx.Response(status)])

    with pytest.raises(RepositoryError, match=f"HTTP {status}"):
        repository.list_items(COLLECTION, YEARS)

    assert (len(server.requests), fake_time.sleeps) == (1, [])


def without_embedded_results(payload: dict[str, Any]) -> None:
    del payload["_embedded"]


def with_a_collection_among_the_items(payload: dict[str, Any]) -> None:
    first_item(payload)["type"] = "collection"


def with_an_item_without_uuid(payload: dict[str, Any]) -> None:
    del first_item(payload)["uuid"]


def with_a_dni_that_is_not_a_list(payload: dict[str, Any]) -> None:
    first_item(payload)["metadata"]["renati.author.dni"] = "SYNTHETIC-DNI-AUTHOR-001"


def with_a_negative_total(payload: dict[str, Any]) -> None:
    result_of(payload)["page"]["totalElements"] = -1


@pytest.mark.parametrize(
    "damage",
    [
        without_embedded_results,
        with_a_collection_among_the_items,
        with_an_item_without_uuid,
        with_a_dni_that_is_not_a_list,
        with_a_negative_total,
    ],
)
def test_a_response_of_an_unexpected_shape_is_rejected_without_retry(
    connect: Connect, damage: Callable[[dict[str, Any]], None]
) -> None:
    payload = page(0)
    damage(payload)
    repository, server = connect([payload])

    with pytest.raises(RepositoryError, match="unexpected") as caught:
        repository.list_items(COLLECTION, YEARS)

    assert len(server.requests) == 1
    assert "SYNTHETIC-DNI" not in str(caught.value)  # Messages never repeat response values.


def test_a_body_that_is_not_json_is_rejected_without_retry(connect: Connect) -> None:
    repository, server = connect([httpx.Response(200, text="<html>Maintenance</html>")])

    with pytest.raises(RepositoryError, match="JSON"):
        repository.list_items(COLLECTION, YEARS)

    assert len(server.requests) == 1


def for_another_scope(payload: dict[str, Any]) -> None:
    payload["scope"] = str(FACULTY)


def without_the_year_filter(payload: dict[str, Any]) -> None:
    payload["appliedFilters"] = []


def with_other_years(payload: dict[str, Any]) -> None:
    payload["appliedFilters"][0]["value"] = "[2020 TO 2026]"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (for_another_scope, "scope"),
        (without_the_year_filter, "filter"),
        (with_other_years, "filter"),
    ],
)
def test_a_response_that_ignored_the_scope_or_the_year_filter_is_rejected(
    connect: Connect, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    payload = page(0)
    change(payload)
    repository, _ = connect([payload])

    with pytest.raises(RepositoryError, match=message):
        repository.list_items(COLLECTION, YEARS)


def with_a_new_total(payload: dict[str, Any]) -> None:
    result_of(payload)["page"]["totalElements"] = 6


def with_more_pages(payload: dict[str, Any]) -> None:
    result_of(payload)["page"]["totalPages"] = 4


def with_the_first_page_number(payload: dict[str, Any]) -> None:
    result_of(payload)["page"]["number"] = 0


@pytest.mark.parametrize("change", [with_a_new_total, with_more_pages, with_the_first_page_number])
def test_a_listing_that_changes_while_paging_is_rejected(
    connect: Connect, change: Callable[[dict[str, Any]], None]
) -> None:
    second = page(1)
    change(second)
    repository, server = connect([page(0), second])

    with pytest.raises(RepositoryError, match="page 1"):
        repository.list_items(COLLECTION, YEARS)

    assert len(server.requests) == 2


def test_a_listing_that_needs_too_many_pages_is_refused(connect: Connect) -> None:
    payload = page(0)
    result_of(payload)["page"].update(
        totalPages=MAX_PAGES_PER_SEARCH + 1, totalElements=2 * (MAX_PAGES_PER_SEARCH + 1)
    )
    repository, server = connect([payload])

    with pytest.raises(RepositoryError, match=str(MAX_PAGES_PER_SEARCH + 1)):
        repository.list_items(COLLECTION, YEARS)

    assert len(server.requests) == 1


def test_count_items_reads_the_reported_total_from_a_one_item_page(connect: Connect) -> None:
    faculty_page = page(0)
    faculty_page["scope"] = str(FACULTY)
    repository, server = connect([faculty_page])

    assert repository.count_items(FACULTY, YEARS) == 5
    assert (server.sent("scope"), server.sent("size"), server.sent("page")) == (
        [str(FACULTY)],
        ["1"],
        ["0"],
    )


def test_describe_search_reports_the_endpoint_and_the_shared_parameters(connect: Connect) -> None:
    repository, server = connect([], page_size=50)

    assert repository.describe_search(YEARS) == SearchDescription(
        search_url=SEARCH_URL,
        query_parameters={
            "dsoType": "ITEM",
            "size": "50",
            "sort": "dc.date.issued,ASC",
            "f.dateIssued": "[2021 TO 2026],equals",
        },
    )
    assert server.requests == []


@pytest.mark.parametrize(
    ("setting", "value"),
    [("page_size", 0), ("max_retries", -1), ("request_interval_seconds", 0.5)],
)
def test_settings_that_would_break_paging_retries_or_the_pace_are_rejected(
    setting: str, value: float
) -> None:
    with httpx.Client() as client, pytest.raises(ValueError, match=setting):
        DSpaceRestRepository(client, BASE_URL, **{setting: value})


def test_from_config_applies_the_page_size_the_pace_and_the_retries(fake_time: FakeTime) -> None:
    settings = RepositoryConfig(
        base_url=HttpUrl(BASE_URL),
        faculty_community_uuid=FACULTY,
        page_size=25,
        request_interval_seconds=2.0,
        max_retries=1,
    )
    server = FakeServer(fake_time, [page(0), httpx.Response(503), httpx.Response(503)])

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        repository = DSpaceRestRepository.from_config(
            client, settings, clock=fake_time.clock, sleeper=fake_time.sleep
        )
        with pytest.raises(RepositoryError, match="after 2 attempts"):
            repository.list_items(COLLECTION, YEARS)

    assert server.sent("size") == ["25", "25", "25"]
    assert str(server.requests[0].url).startswith(f"{SEARCH_URL}?")
    # The second page waits out the 2 s pace; its retry waits the 2 s backoff, which covers it.
    assert fake_time.sleeps == [1.75, 2.0]
