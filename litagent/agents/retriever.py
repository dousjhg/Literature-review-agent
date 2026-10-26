"""2. Retriever agent. Spec: docs/agents/retriever.md

Runs the Planner's queries against each configured scholarly API (OpenAlex, Semantic Scholar,
arXiv), merges and deduplicates the results across sources and paper versions,
drops papers without an abstract (later agents quote them as evidence), ranks by a blend of
relevance and citations per year, caps the number of papers by effort level, and builds the
summary shown at Human review 1.
"""

from __future__ import annotations

import json
import re
import time
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path

from openai import OpenAIError
from pydantic import BaseModel, Field

from litagent.config import EFFORT_PROFILES, Settings
from litagent.integrations.llm import LLMError, TokenBudgetExceeded, complete_json, load_prompt
from litagent.schemas import Paper, RetrievalSummary, SearchCriteria, SearchPlan
from litagent.tools.arxiv import ArxivError, search_arxiv
from litagent.tools.fulltext import FullTextError, fetch_full_text
from litagent.tools.openalex import OpenAlexError, search_openalex
from litagent.tools.semantic_scholar import SemanticScholarError, search_semantic_scholar
from litagent.trace import current_trace

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
RRF_K = 60  # standard reciprocal-rank-fusion constant: one query adds 1 / (RRF_K + rank) to a paper's relevance
SUMMARY_SAMPLE_SIZE = 30  # papers the LLM sees when it writes the summary
SOURCES = {"openalex": "search_openalex", "semantic_scholar": "search_semantic_scholar", "arxiv": "search_arxiv"}
SEARCH_ERRORS = (OpenAlexError, SemanticScholarError, ArxivError)
_ARXIV_DOI = re.compile(r"(?:^|/)10\.48550/arxiv\.(.+)$", re.IGNORECASE)


class RetrievalError(RuntimeError):
    """Every search failed (as opposed to searches that ran and found nothing)."""


class SummaryDraft(BaseModel):
    summary: str = Field(description="2-4 plain sentences for the user")


def retrieve(plan: SearchPlan, settings: Settings, signed_in: bool = False) -> list[Paper]:
    """Returns [] when the searches ran but found nothing; raises RetrievalError when every search failed."""
    if settings.sample_mode:
        return _sample_papers(plan)

    unknown = set(settings.retrieval_sources) - set(SOURCES)
    if unknown or not settings.retrieval_sources:
        raise ValueError(f"retrieval_sources must be a non-empty subset of {sorted(SOURCES)}; got {sorted(unknown)}")
    results: list[list[Paper]] = []
    failures: list[str] = []
    for query in plan.queries:
        for source in settings.retrieval_sources:
            search = globals()[SOURCES[source]]  # looked up per call so tests can replace a source
            try:
                found = search(query, plan.criteria, limit=plan.max_papers, open_access_only=not signed_in)
            except SEARCH_ERRORS as e:
                failures.append(f"{source}: {e}")
                _log(f"search {source}", "failed", query, "next search", str(e))
                continue
            results.append(found)
            _log(f"search {source}", f"{len(found)} papers", query, "next search")
    if not results:
        raise RetrievalError("Every search failed: " + " | ".join(dict.fromkeys(failures)))

    if signed_in:
        # TODO: institutional access. Check your university's licence terms first:
        # publisher licences usually forbid automated bulk downloading.
        pass

    found_total = sum(len(r) for r in results)
    merged = merge_ranked(results)
    with_abstract = [(p, s) for p, s in merged if p.abstract]
    # Sources without citation counts (arXiv) cannot show they meet a stated minimum
    kept = [(p, s) for p, s in with_abstract if _meets_citation_rate(p, plan.criteria.min_citations_per_year)
            and (not plan.criteria.min_citations or (p.citation_count or 0) >= plan.criteria.min_citations)]
    papers = rank(kept, settings.rank_relevance_weight)[: plan.max_papers]
    w = settings.rank_relevance_weight
    _log(
        "merge, filter, rank", f"{len(papers)} papers kept", f"limit {plan.max_papers}", "summarise",
        f"{found_total} found by {len(results)} searches ({', '.join(settings.retrieval_sources)}); removed "
        f"{found_total - len(merged)} duplicates, {len(merged) - len(with_abstract)} without abstract, "
        f"{len(with_abstract) - len(kept)} below the citation minimum; ranked {w:.0%} relevance / {1 - w:.0%} citations per year",
    )

    if EFFORT_PROFILES[plan.effort].use_full_text:
        papers = attach_full_text(papers, settings)
    return papers


def attach_full_text(papers: list[Paper], settings: Settings) -> list[Paper]:
    """High / Ultra: fill full_text from open-access PDFs; papers without one keep their abstract."""
    wanted = [p for p in papers if p.is_open_access and p.pdf_url and not p.full_text]
    texts: dict[str, str] = {}
    failures: dict[str, str] = {}

    def fetch(paper: Paper) -> None:
        try:
            texts[paper.id] = fetch_full_text(paper.pdf_url, timeout_s=settings.full_text_timeout_s,
                                              max_bytes=settings.full_text_max_bytes,
                                              max_pages=settings.full_text_max_pages,
                                              min_chars=settings.full_text_min_chars,
                                              max_chars=settings.full_text_max_chars)
        except FullTextError as e:
            failures[paper.id] = str(e)

    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=max(1, settings.full_text_workers))
    futures = [pool.submit(fetch, p) for p in wanted]
    _, late = wait(futures, timeout=settings.full_text_total_timeout_s)
    for future in late:
        future.cancel()
    pool.shutdown(wait=False, cancel_futures=True)  # a stuck download ends at its own timeout
    timed_out = [p.id for p, f in zip(wanted, futures) if f in late]
    reasons: dict[str, int] = {}
    for reason in failures.values():
        reasons[reason.split(" (")[0]] = reasons.get(reason.split(" (")[0], 0) + 1
    _log("full text", f"{len(texts)} of {len(papers)} papers", next_step="cluster",
         why=f"{len(wanted)} open-access PDFs tried in {time.monotonic() - started:.0f} s; "
             f"{len(papers) - len(wanted)} without an open-access PDF link, {len(failures)} failed {reasons or ''}, "
             f"{len(timed_out)} cut by the {settings.full_text_total_timeout_s:.0f} s limit; the rest use abstracts")
    return [p.model_copy(update={"full_text": texts[p.id]}) if p.id in texts else p for p in papers]


def merge_ranked(results: list[list[Paper]]) -> list[tuple[Paper, float]]:
    """Merge per-query result lists (each in relevance order) and remove duplicates.

    Duplicates share a DOI, an arXiv ID or a normalised title: the same paper found by two queries or
    sources, or a preprint and its published version. The first-seen record keeps its ID; fields it
    lacks (abstract, DOI, PDF link) are filled from its duplicates and the highest citation count is
    kept, so a source that omits abstracts does not drop a paper another source describes. Relevance
    is the reciprocal-rank-fusion score summed over every search that found the paper.
    """
    merged: list[list] = []  # [paper, relevance]
    index: dict[str, int] = {}
    for found in results:
        for position, paper in enumerate(found, start=1):
            keys = _dedupe_keys(paper)
            i = next((index[k] for k in keys if k in index), None)
            if i is None:
                i = len(merged)
                merged.append([paper, 0.0])
            else:
                merged[i][0] = _combine(merged[i][0], paper)
                keys += [k for k in _dedupe_keys(merged[i][0]) if k not in keys]
            for k in keys:
                index.setdefault(k, i)
            merged[i][1] += 1 / (RRF_K + position)
    return [(p, s) for p, s in merged]


def _combine(kept: Paper, other: Paper) -> Paper:
    """Fill the kept record's missing fields from a duplicate of the same paper."""
    update = {f: getattr(other, f) for f in ("abstract", "doi", "pdf_url", "year", "full_text")
              if not getattr(kept, f) and getattr(other, f)}
    if other.citation_count is not None and (kept.citation_count or 0) < other.citation_count:
        update["citation_count"] = other.citation_count
    if kept.is_review is None and other.is_review is not None:
        update["is_review"] = other.is_review
    if other.is_open_access and not kept.is_open_access:
        update["is_open_access"] = True
    if not kept.authors and other.authors:
        update["authors"] = other.authors
    return kept.model_copy(update=update) if update else kept


def rank(scored: list[tuple[Paper, float]], relevance_weight: float) -> list[Paper]:
    """Blend relevance and citations per year, each as a percentile (0-1) so the scales match.
    Ties keep the merged (relevance) order."""
    if not scored:
        return []
    relevance = _percentiles([s for _, s in scored])
    citations = _percentiles([citations_per_year(p) for p, _ in scored])
    blend = [relevance_weight * r + (1 - relevance_weight) * c for r, c in zip(relevance, citations)]
    order = sorted(range(len(scored)), key=lambda i: blend[i], reverse=True)
    return [scored[i][0] for i in order]


def citations_per_year(paper: Paper) -> float:
    """Citations divided by years since publication (counting the current year), so new papers aren't penalised."""
    this_year = date.today().year
    age = max(1, this_year - (paper.year or this_year) + 1)
    return (paper.citation_count or 0) / age


def _percentiles(values: list[float]) -> list[float]:
    """Each value's position in the sorted list as a fraction 0-1 (ties share the lower position)."""
    if len(values) == 1:
        return [1.0]
    ordered = sorted(values)
    return [bisect_left(ordered, v) / (len(values) - 1) for v in values]


def _meets_citation_rate(paper: Paper, minimum: float | None) -> bool:
    return minimum is None or citations_per_year(paper) >= minimum


def _dedupe_keys(paper: Paper) -> list[str]:
    keys = []
    if paper.doi:
        doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", paper.doi.strip().lower())
        keys.append("doi:" + doi)
        if match := _ARXIV_DOI.search(doi):
            keys.append("arxiv:" + re.sub(r"v\d+$", "", match.group(1)))
    if paper.id.lower().startswith("arxiv:"):
        keys.append("arxiv:" + re.sub(r"v\d+$", "", paper.id[6:].lower()))
    title = re.sub(r"[^a-z0-9]+", "", paper.title.lower())
    if len(title) > 20:  # very short titles ("Introduction", "Editorial") would merge unrelated papers
        keys.append("title:" + title)
    return keys or ["id:" + paper.id]


def summarise(papers: list[Paper], plan: SearchPlan, settings: Settings) -> RetrievalSummary:
    """Summary for Human review 1: an LLM paragraph, or a plain one written in code if the LLM fails."""
    facts = _facts(papers, plan)
    text = _plain_summary(facts)
    if papers and not settings.sample_mode:
        message = json.dumps({
            "question": plan.question,
            "facts": facts,
            "papers": [{"title": p.title, "year": p.year, "review": p.is_review} for p in papers[:SUMMARY_SAMPLE_SIZE]],
        }, ensure_ascii=False)
        try:
            text = complete_json(load_prompt("retrieval_summary"), message, SummaryDraft, settings=settings, agent_name="retriever").summary
        except TokenBudgetExceeded:
            raise
        except (LLMError, OpenAIError) as e:
            _log("summarise", "LLM summary failed", next_step="use plain summary", why=f"{type(e).__name__}: {e}")
    return RetrievalSummary(papers=papers, summary=text)


def no_results_message(plan: SearchPlan) -> str:
    """Shown when the searches found nothing: name the filters so the user knows what to loosen."""
    filters = _describe_filters(plan.criteria)
    text = f"I couldn't find any papers on '{plan.criteria.topic}'"
    if filters:
        text += " with these filters: " + ", ".join(filters)
    return text + ". Try broader or different terms" + (", or relax the filters." if filters else ".")


def _describe_filters(c: SearchCriteria, skip: frozenset[str] = frozenset()) -> list[str]:
    """Human-readable filters, leaving out the criteria named in `skip`."""
    filters = []
    if (c.year_from and "year_from" not in skip) or (c.year_to and "year_to" not in skip):
        start = c.year_from if "year_from" not in skip else None
        filters.append(f"published {start or '...'}–{c.year_to or 'now'}")
    if c.paper_types in (["review"], ["primary"]) and "paper_types" not in skip:
        filters.append("reviews only" if c.paper_types == ["review"] else "original studies only")
    if c.min_citations and "min_citations" not in skip:
        filters.append(f"at least {c.min_citations} citations")
    if c.min_citations_per_year and "min_citations_per_year" not in skip:
        filters.append(f"at least {c.min_citations_per_year} citations per year")
    return filters


def _facts(papers: list[Paper], plan: SearchPlan) -> dict:
    years = [p.year for p in papers if p.year]
    reviews = sum(1 for p in papers if p.is_review)
    defaulted = frozenset(d.split("=", 1)[0] for d in plan.used_defaults)
    return {
        "topic": plan.criteria.topic,
        "filters_chosen_by_user": _describe_filters(plan.criteria, skip=defaulted),
        "papers": len(papers),
        "year_range": f"{min(years)}–{max(years)}" if years else None,
        "reviews": reviews,
        "original_studies": len(papers) - reviews,
    }


def _plain_summary(facts: dict) -> str:
    years = f", published {facts['year_range']}" if facts["year_range"] else ""
    return (f"Found {facts['papers']} papers on '{facts['topic']}'{years}: "
            f"{facts['reviews']} reviews and {facts['original_studies']} original studies.")


def _log(action: str, decision: str, result=None, next_step: str = "", why: str = "") -> None:
    if trace := current_trace():
        trace.log("retriever", action, decision, result, next_step, why)


def _sample_papers(plan: SearchPlan) -> list[Paper]:
    """Sample mode: real OpenAlex papers saved under data/sample/, no network.
    Picks the file whose name shares the most words with the question (first file on a tie)."""
    files = sorted(SAMPLE_DIR.glob("*.json"))
    if not files:
        return _fake_papers(plan.max_papers)
    words = _words(plan.question)
    best = max(files, key=lambda f: len(words & _words(f.stem)))
    papers = [Paper.model_validate(d) for d in json.loads(best.read_text(encoding="utf-8"))]
    return papers[: plan.max_papers]


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 3}


def _fake_papers(n: int) -> list[Paper]:
    return [
        Paper(
            id=f"P{i}",
            title=f"Fake paper {i}",
            abstract=f"Abstract of fake paper {i}.",
            year=2022 + i % 4,
            citation_count=10 * i,
            is_open_access=True,
            source="fake",
        )
        for i in range(1, n + 1)
    ]
