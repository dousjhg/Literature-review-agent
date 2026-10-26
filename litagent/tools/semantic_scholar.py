"""Semantic Scholar tool. Docs: https://api.semanticscholar.org/api-docs/graph

name: search_semantic_scholar
inputs: query, SearchCriteria, limit, open_access_only
output: list[Paper] in Semantic Scholar relevance order (one request per query)
fail mode: network error, 5xx or 429 -> retry up to 3 times with backoff, then raise SemanticScholarError;
           other 4xx -> raise at once. Without SEMANTIC_SCHOLAR_API_KEY requests share a public rate
           limit and 429s are common; the Retriever logs the failure and continues with other sources.

Citation links (references / citations of a paper) are not used yet: the Connection finder claims a
citation or extension only from explicit text in the supplied sources.
"""

from __future__ import annotations

import os
import time

import httpx

from litagent.schemas import Paper, SearchCriteria

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
TIMEOUT_S = 30
MAX_LIMIT = 100
MAX_WAIT_S = 5
FIELDS = "title,abstract,year,authors,citationCount,isOpenAccess,openAccessPdf,externalIds,publicationTypes"


class SemanticScholarError(RuntimeError):
    pass


def search_semantic_scholar(query: str, criteria: SearchCriteria, limit: int = 25, open_access_only: bool = True,
                            *, timeout_s: float = TIMEOUT_S, attempts: int = 3) -> list[Paper]:
    data = _get_with_retry(build_params(query, criteria, limit, open_access_only), attempts, timeout_s)
    papers = [_to_paper(item) for item in data.get("data") or [] if item.get("paperId") and item.get("title")]
    types = set(criteria.paper_types)
    if types == {"review"}:
        papers = [p for p in papers if p.is_review]
    elif types == {"primary"}:
        papers = [p for p in papers if not p.is_review]
    return papers


def build_params(query: str, criteria: SearchCriteria, limit: int, open_access_only: bool) -> dict:
    params: dict = {"query": query, "limit": min(limit, MAX_LIMIT), "fields": FIELDS}
    if criteria.year_from or criteria.year_to:
        params["year"] = f"{criteria.year_from or ''}-{criteria.year_to or ''}"
    if criteria.min_citations:
        params["minCitationCount"] = criteria.min_citations
    if open_access_only:
        params["openAccessPdf"] = ""
    return params


def _get_with_retry(params: dict, attempts: int, timeout_s: float) -> dict:
    headers = {"x-api-key": key} if (key := os.getenv("SEMANTIC_SCHOLAR_API_KEY")) else {}
    problem = ""
    for i in range(attempts):
        try:
            r = httpx.get(BASE_URL, params=params, headers=headers, timeout=timeout_s)
        except httpx.HTTPError as e:
            problem = f"network error: {type(e).__name__}"
        else:
            if r.status_code == 429:
                problem = "rate limited" + ("" if headers else " (no SEMANTIC_SCHOLAR_API_KEY)")
                try:
                    wait = float(r.headers.get("retry-after") or 2**i)
                except ValueError:
                    wait = 2**i
                if wait > MAX_WAIT_S:
                    break
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                problem = f"HTTP {r.status_code}"
            elif r.is_error:
                raise SemanticScholarError(f"Semantic Scholar rejected the request: HTTP {r.status_code} {r.text[:200]}")
            else:
                return r.json()
        if i < attempts - 1:
            time.sleep(2**i)
    raise SemanticScholarError(f"Semantic Scholar failed after {attempts} attempts ({problem})")


def _to_paper(item: dict) -> Paper:
    ids = item.get("externalIds") or {}
    pdf = (item.get("openAccessPdf") or {}).get("url") or None
    return Paper(
        id=f"S2:{item['paperId']}",
        title=" ".join(item["title"].split()),
        abstract=item.get("abstract") or None,
        year=item.get("year"),
        authors=[a["name"] for a in item.get("authors") or [] if a.get("name")],
        citation_count=item.get("citationCount"),
        is_review="Review" in (item.get("publicationTypes") or []),
        is_open_access=bool(item.get("isOpenAccess") or pdf),
        pdf_url=pdf,
        doi=ids.get("DOI") or (f"10.48550/arXiv.{ids['ArXiv']}" if ids.get("ArXiv") else None),
        source="semantic_scholar",
    )
