"""4. Connection finder agent. Spec: docs/agents/connection_finder.md

Finds connections within each cluster first, then between clusters.
Every connection must carry Evidence (paper id + verbatim quote).
"""

from __future__ import annotations

from collections import defaultdict

from litagent.config import Settings
from litagent.finding_engine import (FindingContext, cached, cache_key, choose_pairs, exact_ids, merge_record, now,
    pair_payload, prepare, request, resolve_evidence, sources, verify)
from litagent.schemas import (Cluster, Connection, ConnectionDecisions, Evidence, FindingAudit, FindingCacheRecord,
                             FindingDependency, GroundedConnection, Paper)


def find_connections(papers: list[Paper], clusters: list[Cluster], settings: Settings, fix: str | None = None,
                     *, context: FindingContext | None = None) -> list[Connection]:
    if settings.sample_mode:
        return _fake_connections(papers, clusters)

    context = prepare(papers, clusters, settings, context)
    source_map = sources(papers, settings, context)
    pairs, skipped = choose_pairs("connections", papers, clusters, settings, context, set(source_map))
    report = FindingAudit(stage="connections", candidate_pairs=len(pairs), duplicate_pairs_skipped=skipped)
    groups = defaultdict(list)
    for pair in pairs:
        groups[tuple(pair["cluster_ids"])].append(pair)
    out = []
    context.partial_connections = []
    try:
        for group in groups.values():
            payload = {"task_version": "connections-v1", "question": context.question, "fix": fix,
                       "min_quote_chars": settings.finding_min_quote_chars,
                       "pairs": [pair_payload(p, source_map, clusters) for p in group]}
            key = cache_key("connections", payload, ConnectionDecisions, settings)
            record = cached(key, context, settings)
            reused = record is not None
            if record is None:
                record = _assess(group, payload, source_map, settings, context)
                context.cache[key] = record
            merge_record(report, record, context, reused=reused)
            out.extend(c.model_copy(deep=True) for c in record.connections)
            context.partial_connections = list(out)
        return out
    finally:
        context.audits.append(report)


def _assess(pairs, payload, source_map, settings, context):
    targets = {p["pair_id"]: p for p in pairs}
    evidence = {}
    def validate(output):
        exact_ids([d.pair_id for d in output.decisions], targets, "pair_id")
        for d in output.decisions:
            if d.decision == "supported":
                if not d.relation or not d.description.strip():
                    raise ValueError("A supported connection needs a relation and description")
                evidence[d.pair_id] = resolve_evidence(d.evidence, source_map, set(targets[d.pair_id]["paper_ids"]), settings)
    output = request("connection_finder", payload, ConnectionDecisions, validate, settings, context, "connections")
    claims = [{"pair_id": d.pair_id, "relation": d.relation, "description": d.description,
               "evidence": [e.model_dump() for e in evidence[d.pair_id]],
               "papers": [source_map[pid] for pid in targets[d.pair_id]["paper_ids"]]}
              for d in output.decisions if d.decision == "supported"]
    checked = verify(claims, settings, context, "connections")
    connections, deps = [], {}
    audit = FindingAudit(stage="connections", candidate_pairs=len(pairs))
    for d in output.decisions:
        verdict = checked.get(d.pair_id)
        accepted = d.decision == "supported" and verdict.verdict == "supported"
        audit.decisions.append({"pair_id": d.pair_id, "paper_ids": targets[d.pair_id]["paper_ids"],
                               "decision": "accepted" if accepted else "abstain" if verdict is None else "rejected",
                               "reason": verdict.reason if verdict else d.reason})
        if not accepted:
            if verdict is None:
                audit.abstained += 1
            else:
                audit.rejected += 1
            continue
        pair = targets[d.pair_id]
        cid = "CONN-" + d.pair_id.removeprefix("PAIR-")
        connections.append(GroundedConnection(id=cid, kind=pair["kind"], paper_ids=pair["paper_ids"],
            cluster_ids=pair["cluster_ids"], description=d.description, relation=d.relation,
            similarity=pair["similarity"], evidence=evidence[d.pair_id]))
        deps[cid] = FindingDependency(global_map=False, cluster_ids=pair["cluster_ids"], paper_ids=pair["paper_ids"])
        audit.accepted += 1
    return FindingCacheRecord(stage="connections", created_at=now(), connections=connections, dependencies=deps, audit=audit)


def _fake_connections(papers: list[Paper], clusters: list[Cluster]) -> list[Connection]:
    """Sample mode: link the first two papers of each cluster, quoting their abstracts verbatim."""
    by_id = {p.id: p for p in papers}
    out = []
    for c in clusters:
        if len(c.paper_ids) >= 2:
            a, b = c.paper_ids[:2]
            out.append(
                Connection(
                    id=f"{c.id}-K1",
                    kind="within_cluster",
                    paper_ids=[a, b],
                    cluster_ids=[c.id],
                    description="Sample mode: placeholder connection between two papers in the same group.",
                    evidence=[Evidence(paper_id=a, quote=first_sentence(by_id.get(a))),
                              Evidence(paper_id=b, quote=first_sentence(by_id.get(b)))],
                )
            )
    return out


def first_sentence(paper: Paper | None, limit: int = 200) -> str:
    """A verbatim snippet from the abstract (or title), used as sample-mode evidence."""
    if paper is None:
        return ""
    text = paper.abstract or paper.title
    end = text.find(". ")
    return text[: end + 1 if 0 < end < limit else limit]
