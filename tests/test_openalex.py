"""OpenAlex tool with HTTP mocked. The fixture is a real (trimmed) /works response."""

import json
from pathlib import Path

import httpx
import pytest

from litagent.schemas import SearchCriteria
from litagent.tools import openalex
from litagent.tools.openalex import OpenAlexError, build_params, search_openalex

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "openalex_works.json").read_text(encoding="utf-8"))


@pytest.fixture
def http(monkeypatch):
    """Queue responses (or exceptions) for httpx.get; records the params of each call and every sleep."""
    state = {"responses": [], "params": [], "sleeps": []}

    def fake_get(url, params=None, timeout=None):
        assert timeout == openalex.TIMEOUT_S
        state["params"].append(params)
        item = state["responses"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(openalex.httpx, "get", fake_get)
    monkeypatch.setattr(openalex.time, "sleep", state["sleeps"].append)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
    return state


def ok():
    return httpx.Response(200, json=FIXTURE)


def filters(params) -> set[str]:
    return set(params["filter"].split(","))


def test_default_filters():
    params = build_params("gnn drug target", SearchCriteria(topic="t", year_from=2021, paper_types=["primary", "review"]), 30, True)
    assert params["search"] == "gnn drug target"
    assert params["per-page"] == 30
    assert "abstract_inverted_index" in params["select"]
    assert filters(params) == {"type:article|preprint|review", "from_publication_date:2021-01-01", "is_oa:true"}


@pytest.mark.parametrize("criteria, expected", [
    ({"paper_types": ["review"]}, "type:review"),
    ({"paper_types": ["primary"]}, "type:article|preprint"),
    ({"paper_types": []}, "type:article|preprint|review"),
    ({"min_citations": 50}, "cited_by_count:>49"),
    ({"year_to": 2015}, "to_publication_date:2015-12-31"),
])
def test_criteria_become_filters(criteria, expected):
    params = build_params("q", SearchCriteria(topic="t", **criteria), 30, True)
    assert expected in filters(params)


def test_signed_in_drops_open_access_filter_and_page_size_is_capped():
    params = build_params("q", SearchCriteria(topic="t"), 500, open_access_only=False)
    assert "is_oa:true" not in filters(params)
    assert params["per-page"] == 200


def test_api_key_preferred_over_email(monkeypatch):
    monkeypatch.setenv("OPENALEX_EMAIL", "me@example.com")
    assert build_params("q", SearchCriteria(topic="t"), 30, True)["mailto"] == "me@example.com"
    monkeypatch.setenv("OPENALEX_API_KEY", "key123")
    params = build_params("q", SearchCriteria(topic="t"), 30, True)
    assert params["api_key"] == "key123" and "mailto" not in params


def test_parses_real_response(http):
    http["responses"] = [ok()]
    papers = search_openalex("gnn", SearchCriteria(topic="t"), 6)
    assert len(papers) == len(FIXTURE["results"]) == 6
    first = papers[0]
    assert first.id.startswith("W")
    assert first.title.startswith("MGraphDTA")
    assert first.doi == "https://doi.org/10.1039/d1sc05180f"
    assert first.abstract and len(first.abstract.split()) > 50
    assert first.authors and first.year == 2022 and first.citation_count == 348
    assert first.source == "openalex"


def test_daily_limit_fails_fast_without_waiting(http):
    http["responses"] = [httpx.Response(429, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "7200"})]
    with pytest.raises(OpenAlexError, match="daily limit"):
        search_openalex("gnn", SearchCriteria(topic="t"))
    assert len(http["params"]) == 1 and http["sleeps"] == []


def test_short_rate_limit_waits_and_retries(http):
    http["responses"] = [httpx.Response(429, headers={"retry-after": "1"}), ok()]
    assert len(search_openalex("gnn", SearchCriteria(topic="t"))) == 6
    assert http["sleeps"] == [1.0]


def test_server_and_network_errors_are_retried(http):
    http["responses"] = [httpx.Response(503), httpx.ConnectTimeout("timeout"), ok()]
    assert len(search_openalex("gnn", SearchCriteria(topic="t"))) == 6
    assert len(http["params"]) == 3


def test_gives_up_after_three_attempts(http):
    http["responses"] = [httpx.Response(500)] * 3
    with pytest.raises(OpenAlexError, match="after 3 attempts"):
        search_openalex("gnn", SearchCriteria(topic="t"))


def test_bad_request_is_not_retried(http):
    http["responses"] = [httpx.Response(400, text="Invalid filter")]
    with pytest.raises(OpenAlexError, match="HTTP 400"):
        search_openalex("gnn", SearchCriteria(topic="t"))
    assert len(http["params"]) == 1


def test_coverage_call_can_bound_timeout_and_disable_http_retries(monkeypatch):
    calls = []
    def fail(url, params, timeout):
        calls.append(timeout)
        raise httpx.ConnectTimeout("bounded coverage request")
    monkeypatch.setattr(openalex.httpx, "get", fail)
    monkeypatch.setattr(openalex.time, "sleep", lambda *a: pytest.fail("One-attempt coverage must not back off"))
    with pytest.raises(OpenAlexError, match="after 1 attempts"):
        search_openalex("positive study", SearchCriteria(topic="coverage"), timeout_s=7.5, attempts=1)
    assert calls == [7.5]
