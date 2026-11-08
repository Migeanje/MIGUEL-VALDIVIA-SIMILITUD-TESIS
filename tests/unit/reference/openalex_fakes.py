"""Synthetic OpenAlex Topics pages and a fake server for the tests of the Topics download.

``FakeServer`` is a handler for ``httpx.MockTransport``: it serves pages in the shape of the
responses of OpenAlex's ``/topics`` in page-number paging (``meta`` with ``count``, ``page``
and ``per_page``, and ``results``), with invented values, and records every request.
``FakeTime`` stands in for the monotonic clock and the sleeper, so no test waits.
"""

from collections.abc import Callable, Sequence
from typing import Any

import httpx

LATENCY = 0.25
"""Seconds of fake time that the fake server takes to answer a request."""

Reply = httpx.Response | dict[str, Any] | type[BaseException] | BaseException
"""A response, a JSON payload served with HTTP 200, or an exception (class or instance) to
raise; a transport error class gets the request."""


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


def topic(number: int) -> dict[str, Any]:
    """Return a synthetic topic record, laid out as OpenAlex lays out its topics."""
    identifier = f"https://openalex.org/T9{number:04d}"
    return {
        "id": identifier,
        "display_name": f"Tema inventado {number:03d}: diseño de señales pequeñas",
        "description": "Descripción inventada para las pruebas, con acentos y eñes.",
        "keywords": ["palabra inventada", f"clave {number}"],
        "ids": {"openalex": identifier, "wikipedia": None},
        "subfield": {"id": "https://openalex.org/subfields/9901", "display_name": "Subcampo"},
        "field": {"id": "https://openalex.org/fields/99", "display_name": "Campo inventado"},
        "domain": {"id": "https://openalex.org/domains/9", "display_name": "Dominio inventado"},
        "siblings": [{"id": "https://openalex.org/T99999", "display_name": "Hermano"}],
        "works_count": 1000 + number,
        "cited_by_count": 25 * number,
        "score": 0.875,
        "updated_date": "2026-01-02T03:04:05.123456",
        "created_date": "2024-01-02",
    }


def topics(count: int) -> list[dict[str, Any]]:
    return [topic(number) for number in range(1, count + 1)]


def page(results: Sequence[dict[str, Any]], *, count: int, number: int = 1) -> dict[str, Any]:
    """Return page ``number`` of the Topics list, as OpenAlex serves it in page-number paging."""
    return {
        "meta": {
            "count": count,
            "db_response_time_ms": 12,
            "page": number,
            "per_page": 2,
            "groups_count": None,
        },
        "results": list(results),
        "group_by": [],
    }


def numbered_pages(
    records: Sequence[dict[str, Any]], per_page: int, *, count: int | None = None
) -> list[dict[str, Any]]:
    """Split ``records`` into the pages of the listing, page ``n`` answering ``page=n``. Every
    page reports ``count`` as its ``meta.count``, the number of records by default; an empty
    list is one empty page."""
    reported = len(records) if count is None else count
    chunks = [records[start : start + per_page] for start in range(0, len(records), per_page)]
    return [
        page(chunk, count=reported, number=number)
        for number, chunk in enumerate(chunks or [[]], start=1)
    ]


class FakeServer:
    """MockTransport handler that answers each request in turn and records it.

    ``replies`` are served in order. With ``by_page``, a request whose ``page`` names one of its
    pages (``1`` for the first) gets that page instead, once ``replies`` are spent.
    """

    def __init__(
        self,
        time: FakeTime,
        replies: Sequence[Reply] = (),
        *,
        by_page: Sequence[dict[str, Any]] = (),
    ) -> None:
        self.time = time
        self.replies = list(replies)
        self.pages = {str(number): reply for number, reply in enumerate(by_page, start=1)}
        self.requests: list[httpx.Request] = []
        self.started_at: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.started_at.append(self.time.now)
        self.time.now += LATENCY
        reply: Reply = (
            self.replies.pop(0) if self.replies else self.pages[request.url.params["page"]]
        )
        if isinstance(reply, type):
            if issubclass(reply, httpx.TransportError):
                raise reply("simulated transport failure", request=request)
            raise reply()
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, dict):
            return httpx.Response(200, json=reply)
        # A fresh copy, so that one response can be served more than once.
        return httpx.Response(reply.status_code, headers=reply.headers, content=reply.content)

    def sent(self, parameter: str) -> list[str | None]:
        """Return the value of a query parameter in each request, in order."""
        return [request.url.params.get(parameter) for request in self.requests]


ServerFactory = Callable[..., FakeServer]
"""Builds a :class:`FakeServer` on the shared fake clock, from ``replies`` and options."""
