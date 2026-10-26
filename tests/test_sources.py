"""arXiv and Semantic Scholar tools with HTTP mocked."""

import httpx
import pytest

from litagent.schemas import SearchCriteria
from litagent.tools import arxiv, semantic_scholar
from litagent.tools.arxiv import ArxivError, search_arxiv
from litagent.tools.semantic_scholar import SemanticScholarError, search_semantic_scholar

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2410.07830v1</id>
    <published>2024-10-10T11:33:25Z</published>
    <title>NusaMT-7B: Machine Translation for Low-Resource
      Indonesian Languages with Large Language Models</title>
    <summary>  Large Language Models (LLMs) have demonstrated exceptional promise.
    </summary>
    <author><name>William Tan</name></author>
    <author><name>Kevin Zhu</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2410.07830v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/solv-int/9901001v3</id>
    <published>1999-01-02T00:00:00Z</published>
    <title>Old style identifier</title>
    <summary>Text.</summary>
    <arxiv:doi>10.1000/old</arxiv:doi>
  </entry>
</feed>"""


@pytest.fixture
def http(monkeypatch):
    state = {"responses": [], "calls": [], "sleeps": []}
    def get(url, params=None, headers=None, timeout=None):
        state["calls"].append({"url": url, "params": params, "headers": headers})
        item = state["responses"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    for module in (arxiv, semantic_scholar):
        monkeypatch.setattr(module.httpx, "get", get)
        monkeypatch.setattr(module.time, "sleep", state["sleeps"].append)
    monkeypatch.setattr(arxiv, "_last_call", 0.0)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    return state


def test_arxiv_parses_feed_and_versions(http):
    http["responses"].append(httpx.Response(200, text=FEED))
    papers = search_arxiv("LoRA translation", SearchCriteria(topic="t", year_from=2022), limit=500)
    params = http["calls"][0]["params"]
    assert params["search_query"] == "(all:LoRA AND all:translation) AND submittedDate:[202201010000 TO 299912312359]"
    assert params["max_results"] == arxiv.MAX_RESULTS
    first, old = papers
    assert first.id == "arXiv:2410.07830" and first.year == 2024 and first.source == "arxiv"
    assert first.title == "NusaMT-7B: Machine Translation for Low-Resource Indonesian Languages with Large Language Models"
    assert first.abstract.startswith("Large Language Models") and first.authors == ["William Tan", "Kevin Zhu"]
    assert first.doi == "10.48550/arXiv.2410.07830" and first.pdf_url.endswith("2410.07830v1") and first.is_open_access
    assert old.id == "arXiv:solv-int/9901001" and old.doi == "10.1000/old"


def test_arxiv_skips_reviews_only_plans_and_retries_server_errors(http):
    assert search_arxiv("q", SearchCriteria(topic="t", paper_types=["review"])) == [] and not http["calls"]
    http["responses"] += [httpx.Response(503), httpx.ConnectError("down"), httpx.Response(503)]
    with pytest.raises(ArxivError, match="after 3 attempts"):
        search_arxiv("q", SearchCriteria(topic="t"))
    http["responses"].append(httpx.Response(400, text="bad query"))
    with pytest.raises(ArxivError, match="rejected"):
        search_arxiv("q", SearchCriteria(topic="t"))


def test_arxiv_spaces_requests_three_seconds_apart(http, monkeypatch):
    clock = iter([100.0, 100.0, 101.0, 101.0])
    monkeypatch.setattr(arxiv.time, "monotonic", lambda: next(clock))
    http["responses"] += [httpx.Response(200, text=FEED), httpx.Response(200, text=FEED)]
    search_arxiv("q", SearchCriteria(topic="t"))
    search_arxiv("q", SearchCriteria(topic="t"))
    assert http["sleeps"] == [pytest.approx(2.0)]


S2 = {"data": [
    {"paperId": "abc", "title": "LoRA  for MT", "abstract": "An abstract.", "year": 2024, "citationCount": 7,
     "authors": [{"name": "A B"}], "isOpenAccess": True, "openAccessPdf": {"url": "https://x/pdf"},
     "externalIds": {"DOI": "10.1/x"}, "publicationTypes": ["JournalArticle"]},
    {"paperId": "def", "title": "A survey of MT", "abstract": None, "year": 2023, "citationCount": 40,
     "authors": [], "isOpenAccess": False, "openAccessPdf": None, "externalIds": {"ArXiv": "2301.00001"},
     "publicationTypes": ["Review"]},
]}


def test_semantic_scholar_maps_fields_and_filters(http):
    http["responses"].append(httpx.Response(200, json=S2))
    papers = search_semantic_scholar("lora mt", SearchCriteria(topic="t", year_from=2021, min_citations=5), limit=500)
    params = http["calls"][0]["params"]
    assert params["year"] == "2021-" and params["minCitationCount"] == 5 and params["limit"] == 100
    assert "openAccessPdf" in params and http["calls"][0]["headers"] == {}
    lora, survey = papers
    assert lora.id == "S2:abc" and lora.title == "LoRA for MT" and lora.doi == "10.1/x" and lora.pdf_url == "https://x/pdf"
    assert survey.is_review and survey.doi == "10.48550/arXiv.2301.00001" and survey.abstract is None


def test_semantic_scholar_type_filter_and_api_key(http, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "k-123456")
    http["responses"].append(httpx.Response(200, json=S2))
    papers = search_semantic_scholar("q", SearchCriteria(topic="t", paper_types=["primary"]), open_access_only=False)
    assert [p.id for p in papers] == ["S2:abc"]
    assert http["calls"][0]["headers"] == {"x-api-key": "k-123456"} and "openAccessPdf" not in http["calls"][0]["params"]


def test_semantic_scholar_rate_limit(http):
    http["responses"] += [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json={"data": []})]
    assert search_semantic_scholar("q", SearchCriteria(topic="t")) == [] and http["sleeps"] == [1.0]
    http["responses"].append(httpx.Response(429, headers={"retry-after": "60"}))
    with pytest.raises(SemanticScholarError, match="no SEMANTIC_SCHOLAR_API_KEY"):
        search_semantic_scholar("q", SearchCriteria(topic="t"))
