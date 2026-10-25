"""Bounded, neighbor-aware naming; the language model cannot edit the partition."""

from __future__ import annotations

import json
import re

import numpy as np

from litagent.config import Settings
from litagent.integrations.llm import LLMError, complete_json, load_prompt
from litagent.schemas import ClusterAssessment, ClusterAssessmentBatch, ClusterAssessmentProposals, Evidence, Paper
from litagent.similarity import representative_indices
from litagent.trace import current_trace


def evidence_source(paper: Paper, selected_text: str) -> tuple[str, str]:
    """Quote a single original field, never a synthetic title/abstract concatenation."""
    if paper.full_text and selected_text == paper.full_text:
        return "full_text", paper.full_text
    if (paper.abstract or "").strip():
        return "abstract", paper.abstract
    return "title", paper.title


def source_spans(text: str, max_chars: int) -> list[dict]:
    """Contiguous source offsets, sentence boundaries where possible; no text normalization."""
    if max_chars < 1:
        raise ValueError("Source span size must be positive")
    spans = []
    start = 0
    while start < len(text):
        stop = min(start + max_chars, len(text))
        if stop < len(text):
            boundaries = list(re.finditer(r"[.!?](?=\s|[A-Z])", text[start:stop]))
            if boundaries:
                stop = start + boundaries[-1].end()
            else:
                space = text.rfind(" ", start, stop)
                if space > start:
                    stop = space + 1
        spans.append({"span_id": f"S{len(spans) + 1}", "start": start, "end": stop, "quote": text[start:stop]})
        start = stop
    return spans


def label_contexts(papers: list[Paper], texts: list[str], groups: list[list[int]],
                   similarity: np.ndarray, settings: Settings) -> list[dict]:
    """Neighbors are retrieved by their closest pair, with full cross-group range shown.

    Titles of every member are included; bounded MMR examples supply quotable source text.
    Context construction uses no generated labels, so batch boundaries cannot feed back.
    """
    if (settings.cluster_label_batch_size < 1 or settings.cluster_label_neighbors < 0
            or settings.cluster_neighbor_text_chars < 1 or settings.cluster_label_validation_retries < 0
            or settings.cluster_evidence_min_chars < 1
            or settings.cluster_evidence_span_chars < settings.cluster_evidence_min_chars):
        raise ValueError("Invalid cluster labeling limits")
    contexts = []
    for g, group in enumerate(groups):
        reps = representative_indices(similarity, group, settings.cluster_label_max_papers,
                                      settings.cluster_representative_relevance_weight)
        neighbors = []
        for h, other in enumerate(groups):
            if g == h:
                continue
            block = similarity[np.ix_(group, other)]
            a, b = np.unravel_index(np.argmax(block), block.shape)
            neighbor = papers[other[b]]
            neighbors.append({
                "cluster_id": f"C{h + 1}", "max_similarity": float(block.max()),
                "min_similarity": float(block.min()), "mean_similarity": float(block.mean()),
                "closest_pair": [papers[group[a]].id, neighbor.id],
                "members": [{"id": papers[i].id, "title": papers[i].title} for i in other],
                "example": {"id": neighbor.id, "title": neighbor.title,
                            "text": texts[other[b]][:settings.cluster_neighbor_text_chars]},
            })
        neighbors.sort(key=lambda n: (-n["max_similarity"], n["cluster_id"]))
        examples = []
        for i in reps:
            source_field, original = evidence_source(papers[i], texts[i])
            text = original[:settings.cluster_label_text_chars]
            examples.append({"id": papers[i].id, "title": papers[i].title, "text": text,
                             "has_body": source_field != "title", "source_field": source_field,
                             "spans": source_spans(text, settings.cluster_evidence_span_chars)})
        contexts.append({
            "cluster_id": f"C{g + 1}", "total_papers": len(group),
            "members": [{"id": papers[i].id, "title": papers[i].title} for i in group],
            "papers": examples,
            "neighbors": neighbors[:settings.cluster_label_neighbors],
        })
    return contexts


def resolve_assessments(proposals: ClusterAssessmentProposals, contexts: list[dict]) -> ClusterAssessmentBatch:
    sources = {c["cluster_id"]: {p["id"]: {s["span_id"]: s["quote"] for s in p["spans"]}
                                  for p in c["papers"]} for c in contexts}
    assessments = []
    for proposal in proposals.assessments:
        evidence = []
        for ref in proposal.evidence:
            try:
                quote = sources[proposal.cluster_id][ref.paper_id][ref.span_id]
            except KeyError as error:
                raise ValueError(f"Unknown source reference: {proposal.cluster_id}/{ref.paper_id}/{ref.span_id}") from error
            evidence.append(Evidence(paper_id=ref.paper_id, quote=quote))
        assessments.append(ClusterAssessment(**proposal.model_dump(exclude={"evidence"}), evidence=evidence))
    return ClusterAssessmentBatch(assessments=assessments)


def validate_assessments(batch: ClusterAssessmentBatch, contexts: list[dict], settings: Settings) -> None:
    expected = {c["cluster_id"]: c for c in contexts}
    ids = [a.cluster_id for a in batch.assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"Expected cluster_ids {sorted(expected)} exactly once; received {ids}. "
                         "Include title-only groups as insufficient_evidence; neighbors are not targets.")
    for assessment in batch.assessments:
        context = expected[assessment.cluster_id]
        sources = {p["id"]: p for p in context["papers"]}
        evidence_ids = set()
        for e in assessment.evidence:
            if e.paper_id not in sources:
                raise ValueError(f"{assessment.cluster_id}: evidence must cite this group's supplied examples, not neighbors")
            if len(e.quote.strip()) < settings.cluster_evidence_min_chars or e.quote not in sources[e.paper_id]["text"]:
                raise ValueError(f"{assessment.cluster_id}: quotes must be sufficiently long, exact substrings of supplied text")
            evidence_ids.add(e.paper_id)
        if assessment.coherence == "coherent":
            # A title-only pair or unquoted counterexample cannot be turned into a verified shared idea.
            if not all(p["has_body"] for p in sources.values()) or evidence_ids != set(sources):
                raise ValueError(f"{assessment.cluster_id}: coherent requires body text and a quote for EVERY shown example; otherwise abstain")
        elif assessment.coherence == "mixed" and len(evidence_ids) < min(2, len(sources)):
            raise ValueError(f"{assessment.cluster_id}: mixed requires quotes from at least two examples")


def contrastive_labels(papers: list[Paper], texts: list[str], groups: list[list[int]],
                       similarity: np.ndarray, settings: Settings, fix: str | None) -> dict[str, ClusterAssessment]:
    contexts = [c for c in label_contexts(papers, texts, groups, similarity, settings) if c["total_papers"] > 1]
    labels = {}
    for context in contexts:
        if not any(p["has_body"] for p in context["papers"]):
            assessment = ClusterAssessment(
                cluster_id=context["cluster_id"], label=f"Title-only records: {context['members'][0]['title']}",
                rationale="No abstract or full text is available to verify a shared research subtopic.",
                coherence="insufficient_evidence", evidence=[],
                contrast="Titles suggest a topic, but no reliable distinction from neighboring research can be established.")
            labels[assessment.cluster_id] = assessment
            if trace := current_trace():
                trace.log("clusterer", "assess cluster", assessment.coherence, result=assessment.model_dump(),
                          why="Title-only group; deterministic abstention without an LLM call")
    contexts = [c for c in contexts if c["cluster_id"] not in labels]
    for start in range(0, len(contexts), settings.cluster_label_batch_size):
        batch_context = contexts[start:start + settings.cluster_label_batch_size]
        # Source spans carry the text; avoid duplicating full text in the model context.
        public_context = [{**c, "papers": [{**{k: v for k, v in p.items() if k not in {"text", "spans"}},
                                            "spans": [{"span_id": s["span_id"], "quote": s["quote"]} for s in p["spans"]]}
                                           for p in c["papers"]]}
                          for c in batch_context]
        payload = {"task_version": "contrastive-cluster-labels-v2", "groups": public_context, "fix": fix,
                   "min_quote_chars": settings.cluster_evidence_min_chars,
                   "target_cluster_ids": [c["cluster_id"] for c in batch_context]}
        for attempt in range(settings.cluster_label_validation_retries + 1):
            proposals = complete_json(load_prompt("cluster_labeler_contrastive"), json.dumps(payload, ensure_ascii=False),
                                      ClusterAssessmentProposals, settings=settings, agent_name="clusterer")
            try:
                batch = resolve_assessments(proposals, batch_context)
                validate_assessments(batch, batch_context, settings)
            except ValueError as error:
                if trace := current_trace():
                    trace.log("clusterer", "validate cluster evidence", "rejected", why=str(error),
                              next_step="repair" if attempt < settings.cluster_label_validation_retries else "raise")
                if attempt == settings.cluster_label_validation_retries:
                    raise LLMError("Cluster label evidence failed validation after bounded repair") from error
                payload["validation_error"] = str(error)
                payload["rejected_proposal"] = proposals.model_dump()
                continue
            for assessment in batch.assessments:
                labels[assessment.cluster_id] = assessment
                if trace := current_trace():
                    # A nested dict preserves evidence in trace (top-level lists are summarized).
                    trace.log("clusterer", "assess cluster", assessment.coherence,
                              result=assessment.model_dump(), why="Exact quotes checked; semantic support still requires review")
            break
    return labels
