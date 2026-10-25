"""Shared bounded source contexts, pair selection, validation and exact-input task caches."""

from __future__ import annotations

import itertools
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai import OpenAIError

from litagent.cluster_audit import duplicate_components, partition_indices
from litagent.cluster_labels import evidence_source, source_spans
from litagent.cluster_state import digest, file_hash
from litagent.config import Settings
from litagent.integrations.embeddings import embed_documents
from litagent.integrations.llm import LLMError, TokenBudgetExceeded, complete_json, load_prompt
from litagent.schemas import (Evidence, FindingAudit, FindingCacheRecord, FindingDependency,
                             FindingVerifications, Paper, QuestionCluster)
from litagent.similarity import paper_similarity
from litagent.trace import current_trace


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FindingContext:
    question: str | None = None
    similarity: np.ndarray | None = None
    use_full_text: bool = False
    deadline: float | None = None
    cache: dict[str, FindingCacheRecord] = field(default_factory=dict)
    dependencies: dict[str, FindingDependency] = field(default_factory=dict)
    audits: list[FindingAudit] = field(default_factory=list)
    partial_connections: list = field(default_factory=list)
    partial_gaps: list = field(default_factory=list)


def check_deadline(context: FindingContext) -> None:
    if context.deadline is not None and time.monotonic() >= context.deadline:
        raise TokenBudgetExceeded("Finding analysis time limit reached")


def prepare(papers, clusters, settings: Settings, context: FindingContext | None) -> FindingContext:
    partition_indices(papers, clusters)
    if (min(settings.finding_source_chars, settings.finding_span_chars, settings.finding_min_quote_chars,
            settings.connection_within_pairs_per_cluster, settings.connection_max_pairs_per_paper,
            settings.gap_max_candidates, settings.gap_coverage_queries_per_gap,
            settings.gap_coverage_hits_per_query, settings.gap_coverage_source_chars) < 1
            or min(settings.connection_max_within_pairs, settings.connection_max_between_pairs,
                   settings.finding_validation_retries) < 0
            or settings.finding_span_chars < settings.finding_min_quote_chars
            or not 0 <= settings.connection_similarity_min <= 1
            or not 0 <= settings.gap_similarity_min <= settings.gap_similarity_max <= 1
            or not np.isfinite(settings.gap_coverage_ttl_s) or settings.gap_coverage_ttl_s < 0):
        raise ValueError("Invalid finding limits, thresholds or coverage cache lifetime")
    context = context or FindingContext()
    context.question = context.question or next((c.question for c in clusters if isinstance(c, QuestionCluster)), None)
    if context.deadline is None:
        context.deadline = time.monotonic() + settings.run_timeout_s
    check_deadline(context)
    if context.similarity is None:
        texts = [p.full_text if context.use_full_text and p.full_text else f"{p.title}. {p.abstract or ''}" for p in papers]
        context.similarity = paper_similarity(papers, embed_documents(texts, settings), settings).combined
    matrix = np.asarray(context.similarity, dtype=float)
    if (matrix.shape != (len(papers), len(papers)) or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T) or (matrix < -1 - 1e-8).any() or (matrix > 1 + 1e-8).any()):
        raise ValueError("Finding similarity matrix must match the input paper rows")
    context.similarity = matrix
    return context


def sources(papers: list[Paper], settings: Settings, context: FindingContext, *, limit: int | None = None) -> dict:
    out = {}
    for paper in papers:
        selected = paper.full_text if context.use_full_text and paper.full_text else f"{paper.title}. {paper.abstract or ''}"
        field_name, text = evidence_source(paper, selected)
        if field_name == "title" or len(text.strip()) < settings.finding_min_quote_chars:
            continue
        out[paper.id] = {"id": paper.id, "title": paper.title, "source_field": field_name,
                         "spans": [{"span_id": s["span_id"], "quote": s["quote"]} for s in source_spans(
                             text[:limit or settings.finding_source_chars], settings.finding_span_chars)]}
    return out


def choose_pairs(stage: str, papers, clusters, settings: Settings, context: FindingContext, body_ids: set[str]):
    owner = {pid: c for c in clusters for pid in c.paper_ids}
    dup = {pid: n for n, group in enumerate(duplicate_components(papers)) for pid in group}
    indices = {p.id: i for i, p in enumerate(papers)}
    skipped, within, between = 0, [], []
    for a, b in itertools.combinations(sorted(body_ids), 2):
        if dup[a] == dup[b]:
            skipped += 1
            continue
        same = owner[a].id == owner[b].id
        score = float(context.similarity[indices[a], indices[b]])
        if stage == "gaps" and (not same or not settings.gap_similarity_min <= score <= settings.gap_similarity_max):
            continue
        if stage == "connections" and score < settings.connection_similarity_min:
            continue
        cids = sorted({owner[a].id, owner[b].id})
        pair = {"pair_id": "PAIR-" + digest([a, b])[:16], "paper_ids": [a, b], "cluster_ids": cids,
                "kind": "within_cluster" if same else "between_clusters", "similarity": score,
                "main": any(getattr(owner[pid], "kind", "side") == "main" for pid in (a, b))}
        (within if same else between).append(pair)
    order = lambda p: (not p["main"], -p["similarity"], p["paper_ids"])
    within.sort(key=order)
    between.sort(key=order)
    selected, counts, cluster_counts, seen_cross = [], Counter(), Counter(), set()
    for pool, maximum in ((within, settings.gap_max_candidates if stage == "gaps" else settings.connection_max_within_pairs),
                          (between, 0 if stage == "gaps" else settings.connection_max_between_pairs)):
        used = 0
        for pair in pool:
            if used >= maximum:
                break
            cid = pair["cluster_ids"][0]
            if stage == "gaps":
                if cluster_counts[cid]:
                    continue
            elif any(counts[p] >= settings.connection_max_pairs_per_paper for p in pair["paper_ids"]):
                continue
            elif pair["kind"] == "within_cluster" and cluster_counts[cid] >= settings.connection_within_pairs_per_cluster:
                continue
            elif pair["kind"] == "between_clusters" and tuple(pair["cluster_ids"]) in seen_cross:
                continue
            selected.append(pair)
            counts.update(pair["paper_ids"])
            cluster_counts[cid] += 1
            seen_cross.add(tuple(pair["cluster_ids"]))
            used += 1
    if trace := current_trace():
        trace.log(stage + "_finder", "select scientific pairs", f"{len(selected)} bounded comparisons",
                  result={"duplicate_pairs_skipped": skipped, "papers_with_body": len(body_ids), "pairs": selected},
                  why="Question main pairs first; original-source availability, deterministic cosine ranking, duplicate-study exclusion and per-paper/group caps")
    return selected, skipped


def pair_payload(pair, source_map, clusters) -> dict:
    by_id = {c.id: c for c in clusters}
    return {**{k: v for k, v in pair.items() if k != "main"},
            "clusters": [{"id": cid, "kind": getattr(by_id[cid], "kind", "topic"), "label": by_id[cid].label,
                          "paper_ids": sorted(by_id[cid].paper_ids)} for cid in pair["cluster_ids"]],
            "papers": [source_map[pid] for pid in pair["paper_ids"]]}


def exact_ids(received, expected, noun: str) -> None:
    if len(received) != len(set(received)) or set(received) != set(expected):
        raise ValueError(f"Return each supplied {noun} exactly once; expected {sorted(expected)}")


def resolve_evidence(refs, source_map, expected_ids: set[str], settings: Settings) -> list[Evidence]:
    evidence, seen = [], set()
    for ref in refs:
        if ref.paper_id not in expected_ids or ref.paper_id not in source_map:
            raise ValueError("Evidence must refer to an original supplied target paper")
        spans = {s["span_id"]: s["quote"] for s in source_map[ref.paper_id]["spans"]}
        if ref.span_id not in spans or len(spans[ref.span_id].strip()) < settings.finding_min_quote_chars:
            raise ValueError("Evidence must select a sufficiently long original source span")
        if (ref.paper_id, ref.span_id) in seen:
            raise ValueError("Duplicate source reference")
        seen.add((ref.paper_id, ref.span_id))
        evidence.append(Evidence(paper_id=ref.paper_id, quote=spans[ref.span_id]))
    if {e.paper_id for e in evidence} != expected_ids:
        raise ValueError("A supporting quote is required from BOTH target papers")
    return evidence


def request(prompt: str, payload: dict, schema, validate, settings: Settings, context: FindingContext, stage: str):
    message = dict(payload)
    for attempt in range(settings.finding_validation_retries + 1):
        check_deadline(context)
        try:
            output = complete_json(load_prompt(prompt), json.dumps(message, ensure_ascii=False), schema,
                                   settings=settings, agent_name=stage + "_finder")
        except OpenAIError as error:
            raise LLMError(f"{stage} provider request failed ({type(error).__name__})") from error
        try:
            validate(output)
            return output
        except ValueError as error:
            if trace := current_trace():
                trace.log(stage + "_finder", "validate source references", "rejected", why=str(error),
                          next_step="bounded repair" if attempt < settings.finding_validation_retries else "raise")
            if attempt == settings.finding_validation_retries:
                raise LLMError(f"{stage} evidence failed bounded source validation: {error}") from error
            message.update(validation_error=str(error), rejected_proposal=output.model_dump(mode="json"))


def verify(proposals: list[dict], settings: Settings, context: FindingContext, stage: str) -> dict:
    if not proposals:
        return {}
    payload = {"task_version": stage + "-verify-v1", "question": context.question, "claims": proposals}
    def validate(output):
        exact_ids([v.pair_id for v in output.verdicts], [p["pair_id"] for p in proposals], "pair_id")
    output = request(stage + "_verify", payload, FindingVerifications, validate, settings, context, stage)
    return {v.pair_id: v for v in output.verdicts}


def cache_key(stage: str, payload: dict, schema, settings: Settings) -> str:
    names = [stage[:-1] + "_finder", stage + "_verify"]  # connections -> connection_finder
    if stage == "gaps":
        names.append("gap_coverage")
    params = {k: v for k, v in vars(settings).items() if k.startswith(("finding_", "connection_", "gap_"))}
    files = [Path(__file__), Path(__file__).parent / "agents" / (stage[:-1] + "_finder.py")]
    if stage == "gaps":
        files.append(Path(__file__).parent / "tools" / "openalex.py")
    return digest({"payload": payload, "settings": params, "model": settings.llm_model,
                   "prompts": {name: load_prompt(name) for name in names}, "schema": schema.model_json_schema(),
                   "verification_schema": FindingVerifications.model_json_schema(),
                   "implementation": {str(p.name): file_hash(p) for p in files}})


def cached(key: str, context: FindingContext, settings: Settings) -> FindingCacheRecord | None:
    record = context.cache.get(key)
    if record and record.stage == "gaps":
        if any(g.coverage.status == "unavailable" for g in record.gaps):
            return None
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record.created_at)).total_seconds()
        if age < 0 or age >= settings.gap_coverage_ttl_s:
            return None
    return record


def merge_record(report: FindingAudit, record: FindingCacheRecord, context: FindingContext, *, reused: bool) -> None:
    context.dependencies.update(record.dependencies)
    for field_name in ("accepted", "abstained", "rejected"):
        setattr(report, field_name, getattr(report, field_name) + getattr(record.audit, field_name))
    report.decisions.extend(record.audit.decisions)
    report.reused_tasks += int(reused)
    report.generated_tasks += int(not reused)
    if trace := current_trace():
        trace.log(report.stage + "_finder", "finding task cache", "reuse" if reused else "generated",
                  result=record.audit.model_dump(), why="Full task inputs, original sources, model, prompts and schemas are bound to the cache key")
