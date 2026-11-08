"""Behavior of the OpenAlex Topics client: page-number paging, what each request sends, pacing,
retries, refusals and the checks of every page.

``httpx.MockTransport`` serves synthetic pages (:mod:`openalex_fakes`) and a fake clock stands
in for time, so no test opens a connection or sleeps.
"""

from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from openalex_fakes import (
    FakeServer,
    FakeTime,
    ServerFactory,
    numbered_pages,
    page,
    topic,
    topics,
)
from pydantic import HttpUrl

from thematic_redundancy.reference.openalex import (
    MAX_RETRY_AFTER_SECONDS,
    USER_AGENT,
    AccessRefusedError,
    OpenAlexError,
    OpenAlexTopicsClient,
)
from thematic_redundancy.shared.config import OpenAlexConfig

BASE_URL = "https://api.openalex.org"
TOPICS_URL = "https://api.openalex.org/topics"
PER_PAGE = 2

Connect = Callable[..., tuple[OpenAlexTopicsClient, FakeServer]]


@pytest.fixture
def connect(fake_time: FakeTime, server: ServerFactory) -> Iterator[Connect]:
    """Build a client over a fake server; ``settings`` go to the client."""
    clients: list[httpx.Client] = []

    def build(
        replies: Any = (), *, by_page: Any = (), **settings: Any
    ) -> tuple[OpenAlexTopicsClient, FakeServer]:
        fake = server(replies, by_page=by_page)
        client = httpx.Client(transport=httpx.MockTransport(fake))
        clients.append(client)
        settings.setdefault("per_page", PER_PAGE)
        source = OpenAlexTopicsClient(
            client, BASE_URL, clock=fake_time.clock, sleeper=fake_time.sleep, **settings
        )
        return source, fake

    yield build
    for client in clients:
        client.close()


def test_the_listing_reads_the_pages_that_the_first_meta_count_needs(connect: Connect) -> None:
    records = topics(5)
    source, fake = connect(by_page=numbered_pages(records, PER_PAGE))

    listing = source.list_topics()

    assert listing.records == tuple(records)
    assert (listing.meta_count, listing.pages) == (5, 3)
    assert fake.sent("page") == ["1", "2", "3"]
    assert fake.sent("per-page") == ["2"] * 3
    assert {str(request.url.copy_with(query=None)) for request in fake.requests} == {TOPICS_URL}
    assert source.request_count == 3


def test_a_full_last_page_ends_the_listing_without_asking_for_an_empty_one(
    connect: Connect,
) -> None:
    records = topics(4)
    source, fake = connect(by_page=numbered_pages(records, PER_PAGE))

    listing = source.list_topics()

    assert listing.records == tuple(records)
    assert listing.pages == 2
    assert fake.sent("page") == ["1", "2"]


def test_an_empty_list_takes_one_request(connect: Connect) -> None:
    source, fake = connect([page([], count=0)])

    listing = source.list_topics()

    assert (listing.records, listing.meta_count, listing.pages) == ((), 0, 1)
    assert len(fake.requests) == 1


def test_the_records_are_kept_as_the_api_serves_them(connect: Connect) -> None:
    served = {**topic(1), "unknown_new_field": {"nested": [1, 2.5, None, "ñ"]}}
    source, _ = connect([page([served], count=1)])

    (record,) = source.list_topics().records

    assert record == served
    assert list(record) == list(served)


def test_requests_carry_no_personal_data_and_drop_the_cookies_they_got(connect: Connect) -> None:
    pages = numbered_pages(topics(3), PER_PAGE)
    first = httpx.Response(200, json=pages[0], headers={"Set-Cookie": "visit=inventada; Path=/"})
    source, fake = connect([first], by_page=pages)

    source.list_topics()

    assert len(fake.requests) == 2
    for request in fake.requests:
        assert list(request.url.params) == ["per-page", "page"]
        assert request.headers["User-Agent"] == USER_AGENT
        assert request.headers["Accept"] == "application/json"
        for header in ("Authorization", "Cookie", "From", "Api-Key"):
            assert header not in request.headers
        assert "@" not in str(request.url)
        assert "mailto" not in str(request.url).lower()
        assert "api_key" not in str(request.url).lower()
    assert "@" not in USER_AGENT
    assert "thematic-redundancy" in USER_AGENT


def test_requests_start_at_least_the_interval_apart(connect: Connect, fake_time: FakeTime) -> None:
    source, fake = connect(
        by_page=numbered_pages(topics(5), PER_PAGE), request_interval_seconds=1.5
    )

    source.list_topics()

    starts = fake.started_at
    gaps = [later - earlier for earlier, later in zip(starts, starts[1:], strict=False)]
    assert gaps == [pytest.approx(1.5)] * 2
    assert fake_time.sleeps == [pytest.approx(1.25)] * 2  # The fake server took 0.25 s each.


def test_transient_failures_are_retried_with_a_doubling_backoff(
    connect: Connect, fake_time: FakeTime
) -> None:
    pages = numbered_pages(topics(3), PER_PAGE)
    replies = [httpx.Response(503), httpx.ConnectTimeout, httpx.Response(500)]
    source, fake = connect(replies, by_page=pages, backoff_seconds=3.0)

    listing = source.list_topics()

    assert len(listing.records) == 3
    assert fake.sent("page") == ["1", "1", "1", "1", "2"]
    assert fake_time.sleeps[:3] == [3.0, 6.0, 12.0]
    assert source.request_count == 5


def test_a_retry_after_longer_than_the_backoff_is_honoured(
    connect: Connect, fake_time: FakeTime
) -> None:
    busy = httpx.Response(429, headers={"Retry-After": "7"})
    source, _ = connect([busy], by_page=numbered_pages(topics(1), PER_PAGE))

    source.list_topics()

    assert fake_time.sleeps[0] == 7.0


def test_a_retry_after_beyond_a_minute_ends_the_download_naming_the_status(
    connect: Connect, fake_time: FakeTime
) -> None:
    wait = f"{MAX_RETRY_AFTER_SECONDS + 1:g}"
    source, fake = connect([httpx.Response(429, headers={"Retry-After": wait})])

    with pytest.raises(AccessRefusedError, match=r"HTTP 429.*Retry-After") as caught:
        source.list_topics()

    assert caught.value.status_code == 429
    assert len(fake.requests) == 1
    assert fake_time.sleeps == []


@pytest.mark.parametrize("status", [401, 403])
def test_an_access_refusal_is_not_retried_and_names_its_status(
    connect: Connect, status: int
) -> None:
    source, fake = connect([httpx.Response(status)])

    with pytest.raises(AccessRefusedError) as caught:
        source.list_topics()

    message = str(caught.value)
    assert caught.value.status_code == status
    assert f"HTTP {status}" in message
    assert "no email, API key or cookie" in message
    assert "authorization" in message
    assert len(fake.requests) == 1


def test_too_many_requests_ends_the_download_once_the_retries_are_spent(
    connect: Connect, fake_time: FakeTime
) -> None:
    source, fake = connect([httpx.Response(429)] * 3, max_retries=2)

    with pytest.raises(AccessRefusedError, match="HTTP 429 after 3 attempts") as caught:
        source.list_topics()

    assert caught.value.status_code == 429
    assert len(fake.requests) == 3
    assert fake_time.sleeps[-2:] == [2.0, 4.0]


def test_a_server_failure_ends_the_download_once_the_retries_are_spent(connect: Connect) -> None:
    source, fake = connect([httpx.Response(503)] * 2, max_retries=1)

    with pytest.raises(OpenAlexError, match="gave up after 2 attempts: HTTP 503") as caught:
        source.list_topics()

    assert not isinstance(caught.value, AccessRefusedError)
    assert len(fake.requests) == 2


@pytest.mark.parametrize("status", [301, 404, 400])
def test_other_statuses_end_the_download_at_once(connect: Connect, status: int) -> None:
    reply = httpx.Response(status, headers={"Location": "https://elsewhere.example.org/"})
    source, fake = connect([reply])

    with pytest.raises(OpenAlexError, match=f"HTTP {status}") as caught:
        source.list_topics()

    assert not isinstance(caught.value, AccessRefusedError)
    assert len(fake.requests) == 1


def without(payload: dict[str, Any], *path: str) -> dict[str, Any]:
    node = payload
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return payload


@pytest.mark.parametrize(
    ("replies", "message"),
    [
        ([httpx.Response(200, content=b"<html>busy</html>")], "not JSON"),
        ([httpx.Response(200, json=[topic(1)])], "unexpected response"),
        ([without(page([topic(1)], count=1), "meta")], "meta"),
        ([without(page([topic(1)], count=1), "meta", "count")], "meta.count"),
        ([without(page([topic(1)], count=1), "results")], "results"),
        ([page([{"display_name": "Sin id"}], count=1)], "results.0.id"),
        ([page([{**topic(1), "id": 7}], count=1)], "results.0.id"),
        ([page([topic(1), topic(2), topic(3)], count=3)], "more than the 2"),
        (
            [
                page([topic(1), topic(2)], count=4, number=1),
                page([topic(3)], count=5, number=2),
            ],
            "meta.count 5, but page 1 reported 4",
        ),
    ],
    ids=[
        "not-json",
        "not-an-object",
        "no-meta",
        "no-count",
        "no-results",
        "record-without-id",
        "id-not-text",
        "page-too-long",
        "count-changed",
    ],
)
def test_an_unexpected_page_is_refused(connect: Connect, replies: list[Any], message: str) -> None:
    source, _ = connect(replies)

    with pytest.raises(OpenAlexError, match=message):
        source.list_topics()


def test_the_first_request_is_described_for_the_manifest(connect: Connect) -> None:
    source, fake = connect()

    assert source.url == TOPICS_URL
    assert source.first_parameters() == {"per-page": "2", "page": "1"}
    assert fake.requests == []


def test_the_client_is_built_from_the_openalex_block(fake_time: FakeTime) -> None:
    config = OpenAlexConfig(
        base_url=HttpUrl("https://mirror.example.org/openalex/"),
        per_page=150,
        request_interval_seconds=2.0,
        timeout_seconds=12.0,
        max_retries=1,
        backoff_seconds=5.0,
    )
    seen: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(refuse)) as client:
        source = OpenAlexTopicsClient.from_config(
            client, config, clock=fake_time.clock, sleeper=fake_time.sleep
        )
        with pytest.raises(OpenAlexError, match="gave up after 2 attempts"):
            source.list_topics()

    assert source.url == "https://mirror.example.org/openalex/topics"
    assert [request.url.params["per-page"] for request in seen] == ["150", "150"]
    assert seen[0].extensions["timeout"] == {
        "connect": 12.0,
        "read": 12.0,
        "write": 12.0,
        "pool": 12.0,
    }
    assert fake_time.sleeps == [5.0]


@pytest.mark.parametrize(
    "settings",
    [
        {"per_page": 0},
        {"per_page": 201},
        {"request_interval_seconds": 0.5},
        {"max_retries": -1},
        {"timeout_seconds": 0.0},
        {"backoff_seconds": 0.0},
    ],
)
def test_settings_outside_the_contract_are_refused(settings: dict[str, Any]) -> None:
    with httpx.Client() as client, pytest.raises(ValueError):
        OpenAlexTopicsClient(client, BASE_URL, **settings)
