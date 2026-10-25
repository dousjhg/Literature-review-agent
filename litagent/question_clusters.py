"""Question relevance selects the main pool; content similarity organizes remaining papers.

Alternative clusters are numerical hints, never additional membership or scientific claims.
All scores use the original uncentered embeddings; question and document gates are distinct.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np

from litagent.cluster_labels import contrastive_labels, evidence_source, source_spans
from litagent.config import Settings
from litagent.integrations.llm import LLMError, TokenBudgetExceeded, complete_json, load_prompt
from litagent.schemas import (ClusterAlternative, ClusterLabel, Evidence, Paper, PaperClusterDetail,
                             QuestionCluster, QuestionRelevanceBatch)
from litagent.similarity import SimilarityMatrices, cosine_matrix, linkage_groups, partition_metrics
from litagent.trace import current_trace


def validate_settings(settings: Settings) -> None:
    for value in (settings.main_similarity_threshold, settings.side_similarity_threshold,
                  settings.related_similarity_threshold, settings.related_boundary_margin):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Question-map thresholds/margin must be finite values in [0, 1]")
    if (settings.main_relevance_batch_size < 1 or settings.main_relevance_text_chars < 1
            or settings.related_cluster_limit < 1 or settings.paper_keyword_limit < 1
            or settings.cluster_label_validation_retries < 0 or settings.cluster_evidence_min_chars < 1
            or settings.cluster_evidence_span_chars < settings.cluster_evidence_min_chars):
        raise ValueError("Question-map limits must be positive and repair count non-negative")
    if settings.side_linkage not in {"complete", "average"}:
        raise ValueError("side_linkage must be complete or average")


def assess_relevance(papers: list[Paper], texts: list[str], similarities: np.ndarray,
                     question: str, settings: Settings, fix: str | None) -> dict[str, PaperClusterDetail]:
    details, candidates = {}, []
    for i in sorted(range(len(papers)), key=lambda i: papers[i].id):
        paper = papers[i]
        field, source = evidence_source(paper, texts[i])
        body = source[:settings.main_relevance_text_chars]
        if field == "title":
            reason = "No abstract or full text; direct relevance cannot be verified from a title alone."
        elif similarities[i] < settings.main_similarity_threshold:
            reason = "Below the question-similarity candidate gate; direct relevance was not assessed."
        else:
            candidates.append({"id": paper.id, "title": paper.title, "source_field": field,
                               "spans": source_spans(body, settings.cluster_evidence_span_chars)})
            reason = "Awaiting source-based relevance assessment."
        details[paper.id] = PaperClusterDetail(paper_id=paper.id, question_similarity=float(similarities[i]),
                                               relevance="uncertain", reason=reason)
    for start in range(0, len(candidates), settings.main_relevance_batch_size):
        batch = candidates[start:start + settings.main_relevance_batch_size]
        details.update(_assess_batch(batch, details, question, settings, "question_relevance", fix))
        # Verify only the evidence selected for proposed main members. No producer reason,
        # score or prior verdict is supplied; a fresh Nightshift scope isolates the call.
        direct = [{**p, "spans": [s for s in p["spans"] if any(s["quote"] == e.quote for e in details[p["id"]].evidence)]}
                  for p in batch if details[p["id"]].relevance == "direct"]
        if direct:
            before = {p["id"] for p in direct}
            checked = _assess_batch(direct, details, question, settings, "question_relevance_verify", None)
            details.update(checked)
            if trace := current_trace():
                trace.log("clusterer", "verify main evidence", f"{sum(d.relevance == 'direct' for d in checked.values())} of {len(before)} accepted",
                          result={"downgraded_ids": sorted(pid for pid in before if checked[pid].relevance != "direct")},
                          why="Independent assessment of selected original-source spans against every question constraint")
    return details


def _assess_batch(batch: list[dict], details: dict[str, PaperClusterDetail], question: str,
                  settings: Settings, prompt: str, fix: str | None) -> dict[str, PaperClusterDetail]:
    source_refs = {p["id"]: {s["span_id"]: s["quote"] for s in p["spans"]} for p in batch}
    payload = {"task_version": prompt + "-v1", "question": question, "fix": fix,
               "target_paper_ids": list(source_refs), "min_quote_chars": settings.cluster_evidence_min_chars,
               "papers": [{**p, "spans": [{"span_id": s["span_id"], "quote": s["quote"]} for s in p["spans"]]} for p in batch]}
    for attempt in range(settings.cluster_label_validation_retries + 1):
        proposal = complete_json(load_prompt(prompt), json.dumps(payload, ensure_ascii=False),
                                 QuestionRelevanceBatch, settings=settings, agent_name="clusterer")
        try:
            ids = [p.paper_id for p in proposal.assessments]
            if len(ids) != len(set(ids)) or set(ids) != set(source_refs):
                raise ValueError(f"Return exactly the target paper IDs {list(source_refs)}; received {ids}")
            checked = {}
            for assessment in proposal.assessments:
                evidence = []
                for ref in assessment.evidence:
                    quote = source_refs[assessment.paper_id].get(ref.span_id)
                    if ref.paper_id != assessment.paper_id or quote is None or len(quote.strip()) < settings.cluster_evidence_min_chars:
                        raise ValueError(f"Invalid source reference for {assessment.paper_id}; cite its own supplied spans")
                    evidence.append(Evidence(paper_id=assessment.paper_id, quote=quote))
                if assessment.relevance != "uncertain" and not evidence:
                    raise ValueError(f"{assessment.paper_id}: direct/supporting requires source evidence; otherwise use uncertain")
                checked[assessment.paper_id] = details[assessment.paper_id].model_copy(update={
                    "relevance": assessment.relevance, "reason": assessment.reason, "evidence": evidence})
            return checked
        except ValueError as error:
            if trace := current_trace():
                trace.log("clusterer", "validate question relevance", "rejected", why=str(error))
            if attempt == settings.cluster_label_validation_retries:
                raise LLMError("Question relevance failed bounded source validation") from error
            payload.update(validation_error=str(error), rejected_proposal=proposal.model_dump())
    raise AssertionError("unreachable")


def paper_keywords(papers: list[Paper], limit: int) -> dict[str, list[str]]:
    """TF-IDF terms found verbatim in title/abstract, rather than generated keywords."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sources = [f"{p.title} {p.abstract or ''}" for p in papers]
    result = {p.id: [] for p in papers}
    if not papers:
        return result
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2),
                                max_df=0.95 if len(papers) > 1 else 1.0,
                                token_pattern=r"(?u)\b[^\W\d_][\w-]+\b")
    try:
        matrix = vectorizer.fit_transform(sources)
    except ValueError as error:
        if any(t in str(error).lower() for t in ("empty vocabulary", "no terms remain", "after pruning")):
            return result
        raise
    terms = vectorizer.get_feature_names_out()
    for i, paper in enumerate(papers):
        row = matrix.getrow(i)
        for j, _ in sorted(zip(row.indices, row.data), key=lambda item: (-item[1], terms[item[0]])):
            match = next((m for source in (paper.title, paper.abstract or "")
                          if (m := re.search(re.escape(terms[j]), source, re.IGNORECASE))), None)
            if match:
                result[paper.id].append(match.group())
                if len(result[paper.id]) == limit:
                    break
    return result


def cluster_id(member_ids: list[str]) -> str:
    digest = hashlib.sha256(json.dumps(sorted(member_ids), ensure_ascii=False).encode()).hexdigest()
    return "SIDE-" + digest[:16]


def attach_alternatives(clusters: list[QuestionCluster], papers: list[Paper], similarity: np.ndarray,
                        settings: Settings) -> list[QuestionCluster]:
    indices = {p.id: i for i, p in enumerate(papers)}
    for cluster in clusters:
        for detail in cluster.paper_details:
            i = indices[detail.paper_id]
            own = [indices[pid] for pid in cluster.paper_ids if pid != detail.paper_id]
            detail.assigned_similarity = float(similarity[i, own].mean()) if own else None
            detail.runner_up_cluster_id = detail.runner_up_similarity = detail.margin = None
            detail.near_boundary = False
            detail.also_related = []
            alternatives = []
            for other in clusters:
                if other.id == cluster.id:
                    continue
                values = similarity[i, [indices[pid] for pid in other.paper_ids]]
                mean, minimum = float(values.mean()), float(values.min())
                if other.kind == "main":
                    eligible = detail.relevance == "direct" and detail.question_similarity >= settings.main_similarity_threshold
                    constraint = "Main admission also requires the question gate and source-verified direct relevance."
                else:
                    content_score = minimum if settings.side_linkage == "complete" else mean
                    eligible = content_score >= settings.side_similarity_threshold and cluster.kind != "main"
                    constraint = (f"Side {settings.side_linkage} admission score {content_score:.4f}; "
                                  f"cutoff {settings.side_similarity_threshold:.4f}. Main membership takes priority.")
                alternatives.append(ClusterAlternative(cluster_id=other.id, similarity=mean, min_similarity=minimum,
                    meets_membership_criteria=eligible,
                    reason=f"Mean content similarity to {len(other.paper_ids)} final members: {mean:.4f}; minimum: {minimum:.4f}. "
                           + constraint + " A numerical hint, not additional membership or a verified scientific connection."))
            alternatives.sort(key=lambda a: (-a.similarity, a.cluster_id))
            if alternatives:
                runner = alternatives[0]
                detail.runner_up_cluster_id, detail.runner_up_similarity = runner.cluster_id, runner.similarity
                if detail.assigned_similarity is not None:
                    detail.margin = detail.assigned_similarity - runner.similarity
                    detail.near_boundary = (runner.similarity >= settings.related_similarity_threshold
                                            and abs(detail.margin) <= settings.related_boundary_margin)
            detail.also_related = [a for a in alternatives if a.similarity >= settings.related_similarity_threshold][:settings.related_cluster_limit]
    return clusters


def independent_label_payload(papers: list[Paper], texts: list[str], group: list[int], settings: Settings,
                              fix: str | None = None) -> dict:
    return {"papers": [{"id": papers[i].id, "title": papers[i].title,
                        "text": texts[i][:settings.cluster_label_text_chars]} for i in group], "fix": fix}


def label_side_groups(papers: list[Paper], texts: list[str], groups: list[list[int]], similarity: np.ndarray,
                      ids: list[str], settings: Settings, fix: str | None = None,
                      only_ids: set[str] | None = None, deadline: float | None = None) -> dict[str, ClusterLabel]:
    selected = set(ids) if only_ids is None else only_ids
    assessed = (contrastive_labels(papers, texts, groups, similarity, settings, fix, ids, selected, deadline)
                if settings.cluster_label_strategy == "contrastive" and not settings.sample_mode else {})
    labels = {}
    for cid, group in zip(ids, groups):
        # A selected contrastive batch may also regenerate other members of that batch.
        if cid not in selected and cid not in assessed:
            continue
        members = [papers[i] for i in group]
        if settings.sample_mode:
            label = ClusterLabel(label=f"Sample side: {members[0].title}",
                                 rationale="Sample mode: placeholder; no scientific topic assessment was performed.")
        elif len(members) == 1:
            label = ClusterLabel(label=f"Unclustered: {members[0].title}",
                                 rationale="Single-paper side group retained in this map; no shared topic was inferred or generated.")
        elif settings.cluster_label_strategy == "contrastive":
            a = assessed[cid]
            label = ClusterLabel(label=a.label, rationale=f"[{a.coherence}] {a.rationale} {a.contrast}")
        else:
            if deadline is not None and time.monotonic() >= deadline:
                raise TokenBudgetExceeded("Regeneration time limit reached during labeling")
            label = complete_json(load_prompt("cluster_labeler"),
                                  json.dumps(independent_label_payload(papers, texts, group, settings, fix), ensure_ascii=False),
                                  ClusterLabel, settings=settings, agent_name="clusterer")
        labels[cid] = label
    return labels


def build_question_clusters(papers: list[Paper], texts: list[str], vectors: np.ndarray, question_vector: np.ndarray,
                            scores: SimilarityMatrices, question: str, settings: Settings,
                            fix: str | None = None) -> list[QuestionCluster]:
    validate_settings(settings)
    query_scores = cosine_matrix(np.vstack([question_vector, vectors]))[0, 1:]
    details = assess_relevance(papers, texts, query_scores, question, settings, fix)
    keywords = paper_keywords(papers, settings.paper_keyword_limit)
    for pid, values in keywords.items():
        details[pid].keywords = values
    main = sorted((i for i, p in enumerate(papers) if details[p.id].relevance == "direct"), key=lambda i: papers[i].id)
    side = sorted((i for i, p in enumerate(papers) if details[p.id].relevance != "direct"), key=lambda i: papers[i].id)
    side_matrix = scores.combined[np.ix_(side, side)]
    side_groups = linkage_groups(side_matrix, settings.side_similarity_threshold, settings.side_linkage)
    ids = [cluster_id([papers[side[i]].id for i in group]) for group in side_groups]
    if "MAIN" in {p.id for p in papers} or set(ids) & {p.id for p in papers}:
        raise ValueError("Paper ids conflict with generated cluster ids")
    if trace := current_trace():
        trace.log("clusterer", "select main papers", f"{len(main)} direct, {len(side)} remaining",
                  result={"question": question, "main_paper_ids": [papers[i].id for i in main]},
                  threshold=settings.main_similarity_threshold,
                  why="Uncentered query cosine screens candidates; original-source relevance assessment admits main papers")
        trace.log("clusterer", "group papers", f"{len(side_groups)} side topics", threshold=settings.side_similarity_threshold,
                  metrics=partition_metrics(side_matrix, side_groups),
                  why=f"{settings.side_linkage} linkage on remaining papers only; main is a relevance pool, not a topic-diameter cluster")
    clusters = []
    if main:
        clusters.append(QuestionCluster(id="MAIN", label=f"Main: {question}", kind="main", question=question,
            paper_ids=[papers[i].id for i in main], paper_details=[details[papers[i].id] for i in main],
            rationale="These papers passed the question-similarity gate and a source-backed direct-relevance assessment."))
    side_papers, side_texts = [papers[i] for i in side], [texts[i] for i in side]
    labels = label_side_groups(side_papers, side_texts, side_groups, side_matrix, ids, settings, fix)
    for cid, group in zip(ids, side_groups):
        members = [side_papers[i] for i in group]
        label = labels[cid]
        clusters.append(QuestionCluster(id=cid, kind="side", question=question, label=label.label,
            rationale=label.rationale, paper_ids=[p.id for p in members], paper_details=[details[p.id] for p in members]))
    return attach_alternatives(clusters, papers, scores.combined, settings)
