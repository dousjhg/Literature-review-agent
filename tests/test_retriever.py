"""Retriever (real, non-sample path) with search and the LLM mocked."""

import json
from datetime import date

import pytest

from litagent.agents import retriever
from litagent.agents.retriever import (
    RetrievalError, citations_per_year, merge_ranked, no_results_message, rank, retrieve, summarise,
)
from litagent.config import Effort, Settings
from litagent.human_review import AutoApproveReviewer
from litagent.integrations.llm import LLMError, TokenBudgetExceeded
from litagent.orchestrator import run_pipeline
from litagent.schemas import Paper, SearchCriteria, SearchPlan
from litagent.tools.openalex import OpenAlexError
from litagent.trace import Trace

THIS_YEAR = date.today().year


def paper(pid, title=None, doi=None, year=THIS_YEAR, cites=0, abstract="Some abstract text.", review=False):
    return Paper(id=pid, title=title or f"A long enough title for paper {pid}", doi=doi, year=year,
                 citation_count=cites, abstract=abstract, is_review=review, source="openalex")


def make_plan(queries=("q1", "q2"), max_papers=30, effort=Effort.NORMAL, **criteria) -> SearchPlan:
    return SearchPlan(question="How do GNNs predict drug-target interactions?",
                      criteria=SearchCriteria(topic="GNNs for drug-target interaction", **criteria),
                      queries=list(queries), effort=effort, max_papers=max_papers, token_budget=50_000)


def real_settings(sources=("openalex",)) -> Settings:
    s = Settings()
    s.sample_mode = False
    s.retrieval_sources = sources  # tests replace only the sources they name; others would hit the network
    return s


def fake_search(monkeypatch, results: dict):
    """results: query -> list[Paper] or an exception to raise."""
    def search(query, criteria, limit=25, open_access_only=True):
        r = results[query]
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(retriever, "search_openalex", search)


# ----- merging and duplicates -----

def test_same_doi_from_two_queries_is_merged_and_ranks_higher():
    a1 = paper("W1", doi="https://doi.org/10.1/ABC")
    a2 = paper("W9", doi="10.1/abc")  # same DOI, different spelling
    b = paper("W2")
    merged = merge_ranked([[a1, b], [a2]])
    assert [p.id for p, _ in merged] == ["W1", "W2"]
    relevance = dict((p.id, s) for p, s in merged)
    assert relevance["W1"] > relevance["W2"]


def test_preprint_and_published_version_merge_by_title():
    preprint = paper("W1", title="GraphDTA: Predicting drug-target binding affinity", doi="10.1101/preprint")
    published = paper("W2", title="GraphDTA — predicting drug–target binding affinity!", doi="10.1093/journal")
    assert [p.id for p, _ in merge_ranked([[preprint], [published]])] == ["W1"]


def test_short_titles_do_not_merge_unrelated_papers():
    assert len(merge_ranked([[paper("W1", title="Editorial")], [paper("W2", title="Editorial")]])) == 2


# ----- ranking -----

def test_citations_per_year():
    assert citations_per_year(paper("W1", year=THIS_YEAR, cites=10)) == 10
    assert citations_per_year(paper("W2", year=THIS_YEAR - 3, cites=40)) == 10


@pytest.mark.parametrize("weight, expected", [(1.0, ["A", "B", "C"]), (0.7, ["A", "B", "C"]), (0.3, ["C", "B", "A"]), (0.0, ["C", "B", "A"])])
def test_rank_blends_relevance_and_citations(weight, expected):
    scored = [(paper("A", cites=0), 0.03), (paper("B", cites=50), 0.02), (paper("C", cites=100), 0.01)]
    assert [p.id for p in rank(scored, weight)] == expected


# ----- retrieve -----

def test_retrieve_merges_filters_ranks_and_cuts(monkeypatch, tmp_path):
    fake_search(monkeypatch, {
        "q1": [paper("W1", doi="10.1/a", cites=5), paper("W2", abstract=None), paper("W3", cites=1)],
        "q2": [paper("W4", cites=100), paper("W1", doi="10.1/a", cites=5)],
    })
    with Trace(str(tmp_path)) as trace:
        papers = retrieve(make_plan(max_papers=2), real_settings())
    ids = [p.id for p in papers]
    assert len(ids) == 2
    assert "W2" not in ids  # no abstract
    assert ids[0] == "W1"  # found by both queries
    summary = [e for e in trace.events if e["action"] == "merge, filter, rank"][0]
    assert "5 found by 2 searches (openalex)" in summary["why"] and "1 duplicates" in summary["why"] and "1 without abstract" in summary["why"]


def test_citations_per_year_minimum_filters(monkeypatch):
    fake_search(monkeypatch, {"q1": [paper("W1", cites=1), paper("W2", cites=20)]})
    papers = retrieve(make_plan(queries=["q1"], min_citations_per_year=5), real_settings())
    assert [p.id for p in papers] == ["W2"]


def test_one_failed_query_is_logged_and_skipped(monkeypatch, tmp_path):
    fake_search(monkeypatch, {"q1": OpenAlexError("OpenAlex failed after 3 attempts (HTTP 500)"), "q2": [paper("W1")]})
    with Trace(str(tmp_path)) as trace:
        papers = retrieve(make_plan(), real_settings())
    assert [p.id for p in papers] == ["W1"]
    assert any(e["decision"] == "failed" and e["result"] == "q1" for e in trace.events)


def test_every_query_failing_raises(monkeypatch):
    err = OpenAlexError("OpenAlex daily limit reached (resets in 2.0 h).")
    fake_search(monkeypatch, {"q1": err, "q2": err})
    with pytest.raises(RetrievalError, match="daily limit reached"):
        retrieve(make_plan(), real_settings())


def test_nothing_found_returns_empty_list(monkeypatch):
    fake_search(monkeypatch, {"q1": [], "q2": []})
    assert retrieve(make_plan(), real_settings()) == []


def test_full_text_effort_fetches_open_access_pdfs_and_falls_back_to_abstracts(monkeypatch, tmp_path):
    from litagent.tools.fulltext import FullTextError
    oa = paper("W1").model_copy(update={"is_open_access": True, "pdf_url": "https://x/1.pdf"})
    landing = paper("W2").model_copy(update={"is_open_access": True, "pdf_url": "https://x/landing"})
    closed = paper("W3").model_copy(update={"pdf_url": "https://x/3.pdf"})  # not open access: never downloaded
    fake_search(monkeypatch, {"q1": [oa, landing, closed]})
    fetched = []
    def fetch(url, **limits):
        fetched.append(url)
        if url.endswith("landing"):
            raise FullTextError("not a PDF (probably a landing page)")
        return "Full body text of the paper."
    monkeypatch.setattr(retriever, "fetch_full_text", fetch)
    with Trace(str(tmp_path)) as trace:
        papers = retrieve(make_plan(queries=["q1"], effort=Effort.HIGH), real_settings())
    assert sorted(fetched) == ["https://x/1.pdf", "https://x/landing"]
    assert {p.id: p.full_text for p in papers} == {"W1": "Full body text of the paper.", "W2": None, "W3": None}
    event = next(e for e in trace.events if e["action"] == "full text")
    assert event["decision"] == "1 of 3 papers" and "not a PDF" in event["why"]


def test_metadata_effort_never_downloads(monkeypatch):
    fake_search(monkeypatch, {"q1": [paper("W1").model_copy(update={"is_open_access": True, "pdf_url": "https://x"})]})
    monkeypatch.setattr(retriever, "fetch_full_text", lambda *a, **k: pytest.fail("Normal effort must not download PDFs"))
    assert retrieve(make_plan(queries=["q1"]), real_settings())[0].full_text is None


# ----- zero results -----

def test_no_results_message_names_the_filters():
    message = no_results_message(make_plan(year_from=2023, paper_types=["review"], min_citations=50))
    assert "published 2023–now" in message and "reviews only" in message and "at least 50 citations" in message
    assert "relax the filters" in message


def test_orchestrator_returns_no_results_status(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "retrieve", lambda plan, settings, signed_in=False: [])
    settings = Settings()
    settings.sample_mode = True
    settings.logs_dir = str(tmp_path)
    result = run_pipeline("graph neural networks for drug discovery", Effort.NORMAL, AutoApproveReviewer(), settings)
    assert result.status == "no_results"
    assert result.plan is not None and "couldn't find any papers" in result.message_to_user


# ----- summary -----

PAPERS = [paper("W1", year=2021, review=True), paper("W2", year=2024), paper("W3", year=2023)]


def test_summary_from_llm_gets_the_facts(monkeypatch):
    sent = []

    def fake_llm(system_prompt, user_message, output_model, **kwargs):
        sent.append(json.loads(user_message))
        return output_model(summary="Three papers on GNNs.")

    monkeypatch.setattr(retriever, "complete_json", fake_llm)
    result = summarise(PAPERS, make_plan(), real_settings())
    assert result.summary == "Three papers on GNNs."
    assert result.papers == PAPERS
    facts = sent[0]["facts"]
    assert facts == {"topic": "GNNs for drug-target interaction", "filters_chosen_by_user": [], "papers": 3,
                     "year_range": "2021–2024", "reviews": 1, "original_studies": 2}
    assert len(sent[0]["papers"]) == 3


def test_summary_falls_back_when_llm_fails(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise LLMError("model output failed validation twice")

    monkeypatch.setattr(retriever, "complete_json", broken)
    with Trace(str(tmp_path)) as trace:
        result = summarise(PAPERS, make_plan(), real_settings())
    assert result.summary == "Found 3 papers on 'GNNs for drug-target interaction', published 2021–2024: 1 reviews and 2 original studies."
    assert any(e["decision"] == "LLM summary failed" for e in trace.events)


def test_summary_does_not_swallow_token_budget(monkeypatch):
    def over_budget(*args, **kwargs):
        raise TokenBudgetExceeded("used up")

    monkeypatch.setattr(retriever, "complete_json", over_budget)
    with pytest.raises(TokenBudgetExceeded):
        summarise(PAPERS, make_plan(), real_settings())


def test_sample_mode_summary_makes_no_llm_call(monkeypatch):
    monkeypatch.setattr(retriever, "complete_json", lambda *a, **k: pytest.fail("LLM called in sample mode"))
    settings = Settings()
    settings.sample_mode = True
    assert "Found 3 papers" in summarise(PAPERS, make_plan(), settings).summary


def test_summary_lists_only_filters_the_user_chose(monkeypatch):
    sent = []
    monkeypatch.setattr(retriever, "complete_json",
                        lambda s, m, model, **k: sent.append(json.loads(m)) or model(summary="ok"))
    plan = make_plan(year_from=2021, paper_types=["review"], min_citations=50)
    plan.used_defaults = ["year_from=2021"]  # the year was a default, the rest the user's choice
    summarise(PAPERS, plan, real_settings())
    assert sent[0]["facts"]["filters_chosen_by_user"] == ["reviews only", "at least 50 citations"]


# ----- several sources -----

def fake_sources(monkeypatch, **by_source):
    """by_source: source name -> list[Paper] or exception, the same for every query."""
    calls = []
    for name, value in by_source.items():
        def search(query, criteria, limit=25, open_access_only=True, _name=name, _value=value):
            calls.append((_name, query, open_access_only))
            if isinstance(_value, Exception):
                raise _value
            return _value
        monkeypatch.setattr(retriever, retriever.SOURCES[name], search)
    return calls


def test_every_query_runs_on_every_source_and_versions_merge(monkeypatch):
    from litagent.tools.semantic_scholar import SemanticScholarError
    published = paper("W1", title="NusaMT-7B: Machine Translation for Low-Resource Indonesian Languages",
                      doi="https://doi.org/10.48550/arxiv.2410.07830", abstract=None, cites=12)
    preprint = Paper(id="arXiv:2410.07830v2", title="NusaMT-7B: Machine Translation for Low-Resource Languages",
                     abstract="We introduce NusaMT-7B.", source="arxiv", is_open_access=True)
    calls = fake_sources(monkeypatch, openalex=[published], arxiv=[preprint],
                         semantic_scholar=SemanticScholarError("rate limited"))
    papers = retrieve(make_plan(queries=["q1", "q2"]), real_settings(("openalex", "semantic_scholar", "arxiv")))
    assert len(calls) == 6 and {c[0] for c in calls} == {"openalex", "semantic_scholar", "arxiv"}
    # One paper: OpenAlex identity and citations, abstract filled from the arXiv version.
    assert [(p.id, p.abstract, p.citation_count) for p in papers] == [("W1", "We introduce NusaMT-7B.", 12)]


def test_stated_citation_minimum_drops_sources_without_counts(monkeypatch):
    fake_sources(monkeypatch, openalex=[paper("W1", cites=80)],
                 arxiv=[Paper(id="arXiv:1", title="A long enough preprint title here", abstract="x", source="arxiv")])
    papers = retrieve(make_plan(queries=["q1"], min_citations=50), real_settings(("openalex", "arxiv")))
    assert [p.id for p in papers] == ["W1"]


def test_unknown_source_is_a_configuration_error():
    with pytest.raises(ValueError, match="retrieval_sources"):
        retrieve(make_plan(), real_settings(("openalex", "scholar_google")))
