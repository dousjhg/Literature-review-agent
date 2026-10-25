"""Source-grounded subtopic proposals, projected onto hard pairwise similarity constraints.

An experimental inference-time constrained clusterer, not a trained semantic classifier.
The LLM may reorganize papers but may never lose IDs, invent quotes, or bypass the diameter bound.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from litagent.cluster_audit import duplicate_components
from litagent.cluster_labels import evidence_source, source_spans
from litagent.config import Settings
from litagent.integrations.llm import LLMError, complete_json, load_prompt
from litagent.schemas import Cluster, ClusterAssessment, Evidence, Paper, TopicProposal, TopicProposals
from litagent.similarity import complete_linkage_groups, partition_metrics
from litagent.trace import current_trace


def validate_topics(proposal: TopicProposals, sources: dict[str, dict], settings: Settings) -> None:
    members = [pid for topic in proposal.topics for pid in topic.paper_ids]
    if Counter(members) != Counter(sources.keys()):
        counts = Counter(members)
        raise ValueError(f"Partition mismatch: missing={sorted(set(sources) - set(members))}; "
                         f"unknown={sorted(set(members) - set(sources))}; "
                         f"repeated={sorted(pid for pid, n in counts.items() if n > 1)}. "
                         "Return a complete topics list; do not return input paper objects.")
    for topic in proposal.topics:
        referenced = set()
        for ref in topic.evidence:
            if ref.paper_id not in topic.paper_ids:
                raise ValueError("Topic evidence cannot cite another topic's members")
            spans = {s["span_id"]: s["quote"] for s in sources[ref.paper_id]["spans"]}
            if ref.span_id not in spans or len(spans[ref.span_id].strip()) < settings.cluster_evidence_min_chars:
                raise ValueError(f"Invalid/short evidence span {ref.paper_id}/{ref.span_id}")
            referenced.add(ref.paper_id)
        if len(topic.paper_ids) > 1:
            if referenced != set(topic.paper_ids) or not all(sources[pid]["has_body"] for pid in topic.paper_ids):
                raise ValueError("Multi-paper topics require body text and a source-span reference from EVERY member; retain uncertain records separately")


def retain_unassigned(proposal: TopicProposals, sources: dict[str, dict]) -> tuple[TopicProposals, list[str], list[str]]:
    """Resolve only coverage ambiguity, conservatively: no arbitrary choice of competing topic."""
    counts = Counter(pid for topic in proposal.topics for pid in topic.paper_ids)
    missing = sorted(set(sources) - set(counts))
    ambiguous = sorted(pid for pid, n in counts.items() if n > 1 and pid in sources)
    topics = []
    for topic in proposal.topics:
        members = [pid for pid in topic.paper_ids if pid not in ambiguous]
        if members:
            topics.append(topic.model_copy(update={"paper_ids": members,
                "evidence": [e for e in topic.evidence if e.paper_id not in ambiguous]}))
    for pid in missing + ambiguous:
        topics.append(TopicProposal(label=f"Unclustered: {sources[pid]['title']}",
                                    rationale="Assignment omitted or ambiguous; retained separately for review.", paper_ids=[pid]))
    return TopicProposals(topics=topics), missing, ambiguous


def retain_unsupported(proposal: TopicProposals, sources: dict[str, dict]) -> tuple[TopicProposals, list[str]]:
    """Keep an evidence-backed core; a member without a citation remains a singleton.

    This does not repair invented references: those remain subject to strict validation.
    """
    topics, retained = [], []
    for topic in proposal.topics:
        cited = {e.paper_id for e in topic.evidence}
        unsupported = [pid for pid in topic.paper_ids if pid in sources and pid not in cited] if len(topic.paper_ids) > 1 else []
        supported = [pid for pid in topic.paper_ids if pid not in unsupported]
        if supported:
            topics.append(topic.model_copy(update={"paper_ids": supported}))
        for pid in unsupported:
            topics.append(TopicProposal(label=f"Unclustered: {sources[pid]['title']}", paper_ids=[pid],
                                        rationale="No source evidence supplied for this group assignment; retained for review."))
        retained.extend(unsupported)
    return TopicProposals(topics=topics), retained


def semantic_clusters(papers: list[Paper], texts: list[str], similarity: np.ndarray,
                      settings: Settings, fix: str | None = None) -> list[Cluster]:
    threshold = settings.cluster_similarity_threshold
    if settings.cluster_threshold_mode != "fixed":
        raise ValueError("Semantic grouping requires an explicit fixed similarity threshold")
    if (settings.cluster_semantic_text_chars < 1 or settings.cluster_semantic_neighbor_limit < 1
            or settings.cluster_label_validation_retries < 0):
        raise ValueError("Invalid semantic grouping limits")
    if settings.cluster_evidence_span_chars < settings.cluster_evidence_min_chars or settings.cluster_evidence_min_chars < 1:
        raise ValueError("Invalid evidence span limits")
    by_id = {p.id: i for i, p in enumerate(papers)}
    sources = {}
    for i in sorted(range(len(papers)), key=lambda i: papers[i].id):
        source_field, original = evidence_source(papers[i], texts[i])
        text = original[:settings.cluster_semantic_text_chars]
        neighbors = sorted((j for j in range(len(papers)) if i != j and similarity[i, j] >= threshold),
                           key=lambda j: (-similarity[i, j], papers[j].id))[:settings.cluster_semantic_neighbor_limit]
        sources[papers[i].id] = {"id": papers[i].id, "title": papers[i].title,
                                "has_body": source_field != "title", "source_field": source_field,
                                "spans": source_spans(text, settings.cluster_evidence_span_chars),
                                "candidate_neighbors": [papers[j].id for j in neighbors]}
    # Deterministic abstention for title-only records, and isolated points with no compatible pair.
    eligible = {pid: s for pid, s in sources.items() if s["has_body"] and s["candidate_neighbors"]}
    fixed = [pid for pid in sources if pid not in eligible]
    payload = {"task_version": "constrained-subtopics-v1", "fix": fix,
               "target_paper_ids": sorted(eligible),
               "threshold": threshold, "min_quote_chars": settings.cluster_evidence_min_chars,
               "papers": [{**s, "candidate_neighbors": [pid for pid in s["candidate_neighbors"] if pid in eligible],
                           "spans": [{"span_id": v["span_id"], "quote": v["quote"]} for v in s["spans"]]}
                          for s in eligible.values()],
               "possible_duplicate_records": [g for g in duplicate_components(papers) if len(g) > 1]}
    proposal = TopicProposals(topics=[])
    if eligible:
        for attempt in range(settings.cluster_label_validation_retries + 1):
            proposal = complete_json(load_prompt("cluster_subtopics"), json.dumps(payload, ensure_ascii=False),
                                     TopicProposals, settings=settings, agent_name="clusterer")
            if trace := current_trace():
                trace.log("clusterer", "propose subtopics", f"{len(proposal.topics)} model topics",
                          result=proposal.model_dump(), why="Unverified proposal before coverage, source and geometry checks")
            # Omission is treated as abstention, never as permission to discard a source.
            # Ambiguous memberships become singletons. Unknown IDs and invalid evidence still fail validation.
            proposal, missing, ambiguous = retain_unassigned(proposal, eligible)
            proposal, unsupported = retain_unsupported(proposal, eligible)
            if missing and (trace := current_trace()):
                trace.log("clusterer", "retain omitted papers", f"{len(missing)} singletons",
                          result={"paper_ids": missing}, why="Missing assignments are abstentions, not missing outputs")
            if ambiguous and (trace := current_trace()):
                trace.log("clusterer", "retain ambiguous papers", f"{len(ambiguous)} singletons",
                          result={"paper_ids": ambiguous}, why="Multiple proposed topics; do not choose an unsupported membership")
            if unsupported and (trace := current_trace()):
                trace.log("clusterer", "retain unsupported papers", f"{len(unsupported)} singletons",
                          result={"paper_ids": unsupported}, why="No supporting source reference for the proposed membership")
            try:
                validate_topics(proposal, eligible, settings)
                break
            except ValueError as error:
                if trace := current_trace():
                    trace.log("clusterer", "validate subtopic proposal", "rejected", why=str(error),
                              result={"proposed_groups": [t.paper_ids for t in proposal.topics]})
                if attempt == settings.cluster_label_validation_retries:
                    raise LLMError("Subtopic proposal failed bounded source validation") from error
                payload.update(validation_error=str(error), rejected_proposal=proposal.model_dump())
    pieces = [(sorted([by_id[pid] for pid in topic.paper_ids], key=lambda i: papers[i].id), topic)
              for topic in proposal.topics]
    pieces += [([by_id[pid]], None) for pid in fixed]
    accepted = []
    splits = []
    for indices, topic in pieces:
        subgroups = complete_linkage_groups(similarity[np.ix_(indices, indices)], threshold)
        if len(subgroups) > 1:
            splits.append({"proposed_paper_ids": [papers[i].id for i in indices], "pieces": len(subgroups)})
        for subgroup in subgroups:
            accepted.append(([indices[j] for j in subgroup], topic))
    accepted.sort(key=lambda part: min(papers[i].id for i in part[0]))
    metrics = partition_metrics(similarity, [indices for indices, _ in accepted])
    trace = current_trace()
    if trace:
        trace.log("clusterer", "group papers", f"{len(accepted)} constrained subtopics", threshold=threshold,
                  metrics=metrics, sizes={"counts": metrics["sizes"]},
                  projection={"split_proposals": splits},
                  why="Source-grounded task/method/domain proposals; complete-linkage projection preserves every-pair cutoff. "
                      "Size is diagnostic; no automatic threshold relaxation to reduce singletons.")
    result = []
    for index, (indices, topic) in enumerate(accepted, 1):
        ids = [papers[i].id for i in indices]
        if len(ids) == 1:
            label = f"Unclustered: {papers[indices[0]].title}"
            rationale = "Retained individually: no supported shared subtopic met the pairwise content constraint."
        else:
            label = topic.label
            rationale = topic.rationale
            evidence = []
            for ref in topic.evidence:
                if ref.paper_id in ids:
                    span = next(s for s in sources[ref.paper_id]["spans"] if s["span_id"] == ref.span_id)
                    evidence.append(Evidence(paper_id=ref.paper_id, quote=span["quote"]))
            if trace:
                assessment = ClusterAssessment(cluster_id=f"C{index}", label=label, rationale=rationale,
                                               coherence="coherent", evidence=evidence,
                                               contrast="Model-proposed concrete subtopic, constrained by original semantic similarity; expert review pending.")
                trace.log("clusterer", "assess cluster", "coherent", result=assessment.model_dump(),
                          why="Model hypothesis with verified source spans, not a human judgment")
        result.append(Cluster(id=f"C{index}", label=label, rationale=rationale, paper_ids=ids))
    return result
