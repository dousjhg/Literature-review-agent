"""Full-text tool: download an open-access PDF and extract its text (High / Ultra effort).

name: fetch_full_text
inputs: PDF URL, size/page/char limits, timeout
output: cleaned plain text (line-break hyphenation joined, references section removed)
fail mode: network error, HTTP error, a non-PDF response (OpenAlex oa_url is sometimes a landing page),
           a file over the size limit, or too little extractable text (scanned PDF) -> FullTextError.
           One attempt, no retries: the Retriever falls back to the abstract and logs why.

Only called for open-access records; paywalled downloads need a licence check first.
"""

from __future__ import annotations

import io
import logging
import re

import httpx

REFERENCES_AFTER = 0.3  # share of the text before which a "References" heading is not the reference list
USER_AGENT = "litagent/0.1 (literature review agent; open-access full text)"


class FullTextError(RuntimeError):
    pass


def fetch_full_text(url: str, *, timeout_s: float, max_bytes: int, max_pages: int, min_chars: int,
                    max_chars: int) -> str:
    data = _download(url, timeout_s, max_bytes)
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise FullTextError("not a PDF (probably a landing page)")
    text = extract_text(data, max_pages)
    if len(text) < min_chars:
        raise FullTextError(f"only {len(text)} characters of extractable text (scanned or image PDF?)")
    return text[:max_chars]


def _download(url: str, timeout_s: float, max_bytes: int) -> bytes:
    try:
        with httpx.stream("GET", url, timeout=timeout_s, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"}) as r:
            if r.is_error:
                raise FullTextError(f"HTTP {r.status_code}")
            if int(r.headers.get("content-length") or 0) > max_bytes:
                raise FullTextError(f"PDF larger than {max_bytes // 1_000_000} MB")
            chunks, size = [], 0
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise FullTextError(f"PDF larger than {max_bytes // 1_000_000} MB")
                chunks.append(chunk)
            return b"".join(chunks)
    except httpx.HTTPError as e:
        raise FullTextError(f"download failed: {type(e).__name__}") from e


def extract_text(data: bytes, max_pages: int) -> str:
    """pdfminer's layout analysis keeps two-column papers in reading order (pypdf interleaved them)."""
    from pdfminer.high_level import extract_text as pdfminer_text
    from pdfminer.pdfparser import PDFSyntaxError

    logging.getLogger("pdfminer").setLevel(logging.ERROR)  # font warnings would flood the run log
    try:
        raw = pdfminer_text(io.BytesIO(data), maxpages=max_pages)
    except (PDFSyntaxError, ValueError, KeyError, TypeError, AssertionError) as e:
        raise FullTextError(f"unreadable PDF: {type(e).__name__}") from e
    return clean_text(raw)


def clean_text(raw: str) -> str:
    """Plain prose: join hyphenated line breaks and wrapped lines, drop the reference list."""
    text = raw.replace("­", "").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "trans-\nlation" -> "translation"
    # The first references heading after the opening third ends the body; appendices and checklists follow it.
    # (Measured: NeurIPS-style papers put references near 45% of the text, before long appendices.)
    for match in re.finditer(r"\n\s*(?:\d+\.?\s*)?(references|bibliography)\s*\n", text, re.IGNORECASE):
        if match.start() > len(text) * REFERENCES_AFTER:
            text = text[:match.start()]
            break
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)
