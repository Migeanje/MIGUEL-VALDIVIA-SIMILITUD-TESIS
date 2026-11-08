"""Fixtures of the tests of the OpenAlex Topics download: no test reaches the network.

An autouse guard makes httpx's real transport fail, so a test that forgot its fake transport
cannot reach ``api.openalex.org``. The fakes live in :mod:`openalex_fakes`.
"""

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from openalex_fakes import FakeServer, FakeTime, Reply, ServerFactory


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every request through httpx's real transport fail before it leaves the machine."""

    def refuse(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a test tried to reach the network: {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def server(fake_time: FakeTime) -> ServerFactory:
    """Build a :class:`FakeServer` on the shared fake clock."""

    def build(replies: Sequence[Reply] = (), **options: Any) -> FakeServer:
        return FakeServer(fake_time, replies, **options)

    return build
