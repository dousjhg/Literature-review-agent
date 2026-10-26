"""arXiv tool: preprints with free full-text PDFs. Docs: https://info.arxiv.org/help/api/

name: search_arxiv
inputs: query, SearchCriteria, limit
output: list[Paper] in arXiv relevance order (pdf_url always set, citation_count unknown)
fail mode: network error or 5xx -> retry up to 3 times with backoff, then raise ArxivError; 4xx -> raise at once.
           arXiv asks for at most one request every 3 seconds, so calls are spaced by MIN_INTERVAL_S.

arXiv has no review/primary type and no citation counts: a "reviews only" plan skips arXiv, and the
Retriever ranks arXiv papers with zero citations per year (relevance still counts).
"""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx

from litagent.schemas import Paper, SearchCriteria

BASE_URL = "https://export.arxiv.org/api/query"
TIMEOUT_S = 30
MAX_RESULTS = 200
MIN_INTERVAL_S = 3.0
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_lock = threading.Lock()
_last_call = 0.0


class ArxivError(RuntimeError):
    pass


def search_arxiv(query: str, criteria: SearchCriteria, limit: int = 25, open_access_only: bool = True,
                 *, timeout_s: float = TIMEOUT_S, attempts: int = 3) -> list[Paper]:
    if criteria.paper_types == ["review"]:
        return []  # arXiv cannot tell reviews apart; returning preprints would break the user's filter
    text = _get_with_retry(build_params(query, criteria, limit), attempts, timeout_s)
    return parse_feed(text)


def build_params(query: str, criteria: SearchCriteria, limit: int) -> dict:
    terms = " AND ".join(f"all:{_term(w)}" for w in query.split() if _term(w))
    if criteria.year_from or criteria.year_to:
        start, end = f"{criteria.year_from or 1991}01010000", f"{criteria.year_to or 2999}12312359"
        terms = f"({terms}) AND submittedDate:[{start} TO {end}]"
    return {"search_query": terms, "start": 0, "max_results": min(limit, MAX_RESULTS),
            "sortBy": "relevance", "sortOrder": "descending"}


def _term(word: str) -> str:
    return re.sub(r"[^\w\-]+", "", word)


def parse_feed(text: str) -> list[Paper]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ArxivError(f"arXiv returned invalid XML: {e}") from e
    papers = []
    for entry in root.findall("a:entry", NS):
        url = (entry.findtext("a:id", "", NS) or "").strip()
        match = re.search(r"arxiv\.org/abs/(\S+?)(?:v\d+)?$", url)
        if not match:
            continue  # the API reports errors as an entry without an abs link
        arxiv_id = match.group(1)
        published = entry.findtext("a:published", "", NS) or ""
        pdf = next((link.get("href") for link in entry.findall("a:link", NS) if link.get("title") == "pdf"), None)
        papers.append(Paper(
            id=f"arXiv:{arxiv_id}",
            title=_clean(entry.findtext("a:title", "", NS)),
            abstract=_clean(entry.findtext("a:summary", "", NS)) or None,
            year=int(published[:4]) if published[:4].isdigit() else None,
            authors=[_clean(a.findtext("a:name", "", NS)) for a in entry.findall("a:author", NS)],
            citation_count=None,
            is_review=None,
            is_open_access=True,
            pdf_url=pdf or f"https://arxiv.org/pdf/{arxiv_id}",
            doi=_clean(entry.findtext("arxiv:doi", "", NS)) or f"10.48550/arXiv.{arxiv_id}",
            source="arxiv",
        ))
    return papers


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _get_with_retry(params: dict, attempts: int, timeout_s: float) -> str:
    global _last_call
    problem = ""
    for i in range(attempts):
        with _lock:  # space requests as arXiv's terms ask, also across threads
            wait = _last_call + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        try:
            r = httpx.get(BASE_URL, params=params, timeout=timeout_s)
        except httpx.HTTPError as e:
            problem = f"network error: {type(e).__name__}"
        else:
            if r.status_code >= 500 or r.status_code == 429:
                problem = f"HTTP {r.status_code}"
            elif r.is_error:
                raise ArxivError(f"arXiv rejected the request: HTTP {r.status_code} {r.text[:200]}")
            else:
                return r.text
        if i < attempts - 1:
            time.sleep(2**i)
    raise ArxivError(f"arXiv failed after {attempts} attempts ({problem})")
