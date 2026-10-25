"""OpenAlex search tool. Docs: https://docs.openalex.org

name: search_openalex
inputs: query, SearchCriteria, limit, open_access_only
output: list[Paper], in OpenAlex relevance order (one request per query)
fail mode: network error, 5xx or a short 429 -> retry up to 3 times with backoff, then raise OpenAlexError.
           Daily budget used up (429 with a long wait) or a 4xx -> raise OpenAlexError at once:
           retrying can't help, and a run must not wait hours.

Usage is metered per day: about 100 searches without a key, 10x that with a free OPENALEX_API_KEY.
"""

from __future__ import annotations

import os
import time

import httpx

from litagent.schemas import Paper, SearchCriteria

BASE_URL = "https://api.openalex.org/works"
TIMEOUT_S = 30
MAX_PER_PAGE = 200
MAX_WAIT_S = 5  # a 429 asking us to wait longer than this means the daily budget is used up
FIELDS = "id,doi,display_name,publication_year,type,cited_by_count,open_access,authorships,abstract_inverted_index"
# Leave out works that aren't papers (datasets, errata, paratext, ...)
TYPE_FILTER = {
    "both": "article|preprint|review",
    "review": "review",
    "primary": "article|preprint",
}


class OpenAlexError(RuntimeError):
    pass


def search_openalex(query: str, criteria: SearchCriteria, limit: int = 25, open_access_only: bool = True,
                   *, timeout_s: float = TIMEOUT_S, attempts: int = 3) -> list[Paper]:
    data = _get_with_retry(BASE_URL, build_params(query, criteria, limit, open_access_only), attempts, timeout_s)
    return [_to_paper(w) for w in data.get("results", [])]


def build_params(query: str, criteria: SearchCriteria, limit: int, open_access_only: bool) -> dict:
    types = set(criteria.paper_types)
    kind = "review" if types == {"review"} else "primary" if types == {"primary"} else "both"
    filters = [f"type:{TYPE_FILTER[kind]}"]
    if criteria.year_from:
        filters.append(f"from_publication_date:{criteria.year_from}-01-01")
    if criteria.year_to:
        filters.append(f"to_publication_date:{criteria.year_to}-12-31")
    if criteria.min_citations:
        filters.append(f"cited_by_count:>{criteria.min_citations - 1}")
    if open_access_only:
        filters.append("is_oa:true")

    params = {"search": query, "filter": ",".join(filters), "per-page": min(limit, MAX_PER_PAGE), "select": FIELDS}
    if key := os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = key
    elif email := os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = email
    return params


def _get_with_retry(url: str, params: dict, attempts: int = 3, timeout_s: float = TIMEOUT_S) -> dict:
    problem = ""
    for i in range(attempts):
        try:
            r = httpx.get(url, params=params, timeout=timeout_s)
        except httpx.HTTPError as e:
            problem = f"network error: {type(e).__name__}"
        else:
            if r.status_code == 429:
                wait = _seconds_to_wait(r)
                if wait > MAX_WAIT_S:
                    raise OpenAlexError(
                        f"OpenAlex daily limit reached (resets in {wait / 3600:.1f} h). "
                        "A free OPENALEX_API_KEY raises the limit 10x."
                    )
                problem = "rate limited"
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                problem = f"HTTP {r.status_code}"
            elif r.is_error:  # our request is wrong; retrying won't help
                raise OpenAlexError(f"OpenAlex rejected the request: HTTP {r.status_code} {r.text[:200]}")
            else:
                return r.json()
        if i < attempts - 1:
            time.sleep(2**i)
    raise OpenAlexError(f"OpenAlex failed after {attempts} attempts ({problem})")


def _seconds_to_wait(r: httpx.Response) -> float:
    if r.headers.get("x-ratelimit-remaining") == "0":
        return float(r.headers.get("x-ratelimit-reset") or 86_400)
    try:
        return float(r.headers.get("retry-after") or 1)
    except ValueError:
        return 1.0


def _to_paper(work: dict) -> Paper:
    oa = work.get("open_access") or {}
    return Paper(
        id=work["id"].rsplit("/", 1)[-1],
        title=work.get("display_name") or "",
        abstract=_rebuild_abstract(work.get("abstract_inverted_index")),
        year=work.get("publication_year"),
        authors=[a["author"]["display_name"] for a in work.get("authorships") or [] if a.get("author")],
        citation_count=work.get("cited_by_count"),
        is_review=work.get("type") == "review",
        is_open_access=bool(oa.get("is_oa")),
        pdf_url=oa.get("oa_url"),
        doi=work.get("doi"),
        source="openalex",
    )


def _rebuild_abstract(inverted_index: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; turn it back into text."""
    if not inverted_index:
        return None
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))
