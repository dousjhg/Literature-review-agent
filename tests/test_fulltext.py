"""Full-text PDF tool with HTTP mocked, and question-relevant passage selection for long text."""

import httpx
import pytest

from litagent.cluster_labels import relevant_spans, source_spans
from litagent.tools import fulltext
from litagent.tools.fulltext import FullTextError, clean_text, fetch_full_text

LIMITS = dict(timeout_s=5, max_bytes=100_000, max_pages=5, min_chars=40, max_chars=10_000)


def pdf(lines: list[str]) -> bytes:
    """A minimal one-page PDF whose text pypdf can extract."""
    stream = "BT /F1 11 Tf 50 750 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               "/Resources << /Font << /F1 5 0 R >> >> >>",
               f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def served(monkeypatch):
    state = {}
    def stream(method, url, **kwargs):
        response = state["response"]
        if isinstance(response, Exception):
            raise response
        return httpx.Response(response[0], content=response[1], headers=response[2] if len(response) > 2 else None,
                              request=httpx.Request(method, url))
    class Context:
        def __init__(self, *a, **k):
            self.response = stream(*a, **k)
        def __enter__(self):
            return self.response
        def __exit__(self, *exc):
            return False
    monkeypatch.setattr(fulltext.httpx, "stream", Context)
    return state


def test_extracts_text_from_a_real_pdf(served):
    served["response"] = (200, pdf(["LoRA adapts attention layers for low-resource", "machine translation on FLORES."]))
    text = fetch_full_text("https://x/paper.pdf", **LIMITS)
    assert "LoRA adapts attention layers" in text and "FLORES" in text


@pytest.mark.parametrize("response, message", [
    ((200, b"<html>Landing page</html>"), "not a PDF"),
    ((404, b""), "HTTP 404"),
    ((200, b"%PDF" + b"x" * 200_000), "larger than"),
    ((200, b"%PDF-1.4", {"content-length": "999999999"}), "larger than"),
    (httpx.ConnectTimeout("slow"), "download failed"),
])
def test_unusable_responses_fail_with_a_reason(served, response, message):
    served["response"] = response
    with pytest.raises(FullTextError, match=message):
        fetch_full_text("https://x/p", **LIMITS)


def test_too_little_text_counts_as_a_scanned_pdf(served):
    served["response"] = (200, pdf(["Hi"]))
    with pytest.raises(FullTextError, match="scanned"):
        fetch_full_text("https://x/p", **LIMITS)


def test_clean_text_joins_hyphenation_and_drops_references():
    body = "Introduction\nWe study trans-\nlation of low-resource\nlanguages.\n\n" + "Method text. " * 20
    raw = body + "\nReferences\n[1] A. Author. Some paper. 2020."
    cleaned = clean_text(raw)
    assert "translation of low-resource languages." in cleaned and "[1] A. Author" not in cleaned
    early = "Abstract text.\nReferences\nmentioned early" + " body" * 50
    assert "mentioned early" in clean_text(early)  # a heading in the opening third is not the reference list
    appendix = "Body text. " * 40 + "\nReferences\n[1] Cited work.\n\nAppendix A. Checklist." + " more" * 60
    assert "Checklist" not in clean_text(appendix)  # references before long appendices still end the body


def test_long_text_keeps_question_relevant_verbatim_passages():
    front = "University of Somewhere. Department of Computing. Corresponding author email. " * 6
    middle = "We fine-tune with LoRA adapters for low-resource machine translation and report COMET. "
    text = front + "Unrelated background on optimisers. " * 20 + middle * 3 + "Acknowledgements and funding. " * 10
    spans = relevant_spans(text, "LoRA low-resource machine translation", budget=400, span_chars=200)
    assert sum(len(s["quote"]) for s in spans) <= 400 and all(s["quote"] in text for s in spans)
    assert any("LoRA adapters" in s["quote"] for s in spans)
    assert [s["start"] for s in spans] == sorted(s["start"] for s in spans)


def test_short_text_is_unchanged():
    text = "A short abstract about LoRA. It fits the budget."
    assert relevant_spans(text, "anything", budget=1000, span_chars=20) == source_spans(text, 20)


def test_cluster_naming_quotes_the_abstract_not_pdf_front_matter():
    import numpy as np
    from litagent.cluster_labels import label_contexts
    from litagent.config import Settings
    from litagent.schemas import Paper
    papers = [Paper(id=f"P{i}", title=f"Paper {i}", abstract=f"Abstract {i} about LoRA translation adapters.",
                    full_text=f"University {i}. Department. Email. " * 40, source="test") for i in (1, 2)]
    papers.append(Paper(id="P3", title="No abstract", full_text="Body text about adapters. " * 10, source="test"))
    contexts = label_contexts(papers, [p.full_text for p in papers], [[0, 1, 2]], np.ones((3, 3)), Settings())
    fields = {e["id"]: (e["source_field"], e["text"][:8]) for e in contexts[0]["papers"]}
    assert fields == {"P1": ("abstract", "Abstract"), "P2": ("abstract", "Abstract"), "P3": ("full_text", "Body tex")}
