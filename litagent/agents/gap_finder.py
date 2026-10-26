"""5. Gap finder agent. Spec: docs/agents/gap_finder.md

Looks for research gaps inside a cluster, only between papers whose similarity is in a
set range (related enough to be comparable, different enough to leave room for a gap).
"""

from __future__ import annotations

import re
import time

from litagent.agents.connection_finder import first_sentence
from litagent.config import Settings
from litagent.finding_engine import (FindingContext, cached, cache_key, check_deadline, choose_pairs, exact_ids,
    merge_record, now, pair_payload, prepare, request, resolve_evidence, sources, verify)
from litagent.schemas import (Cluster, Connection, CoverageSearch, CoverageVerdicts, Evidence, FindingAudit,
    FindingCacheRecord, FindingDependency, Gap, GapCoverage, GapDecisions, GroundedGap, Paper, SearchCriteria)
from litagent.tools.openalex import OpenAlexError, search_openalex
from litagent.trace import current_trace, mask

# "SFT" or "fine-tuning" alone does not establish which parameters were trained (NusaMT-7B
# says SFT in its abstract but uses LoRA in its full text). A proposal may only assert this
# scope when a supplied source of the pair states it.
_FULL_PARAMETER = re.compile(r"\bfull(?:[- ]parameter|[- ]model|[- ]weight)s?\b|\bfull[- ]?(?:fine[- ]?tun\w*|sft)\b"
                             r"|全参数|全量(?:微调|参数)", re.IGNORECASE)


def unsupported_scope_claims(texts: list[str], sources: list[dict]) -> list[str]:
    """Parameter-scope terms in a proposal that no supplied source span of the pair states."""
    found = {m.group() for text in texts for m in _FULL_PARAMETER.finditer(text)}
    stated = any(_FULL_PARAMETER.search(span["quote"]) for source in sources for span in source["spans"])
    return [] if stated else sorted(found)


def find_gaps(
    papers: list[Paper],
    clusters: list[Cluster],
    connections: list[Connection],
    settings: Settings,
    fix: str | None = None,
    *, context: FindingContext | None = None,
) -> list[Gap]:
    if settings.sample_mode:
        return _fake_gaps(papers, clusters)

    context = prepare(papers, clusters, settings, context)
    source_map = sources(papers, settings, context)
    pairs, skipped = choose_pairs("gaps", papers, clusters, settings, context, set(source_map))
    report = FindingAudit(stage="gaps", candidate_pairs=len(pairs), duplicate_pairs_skipped=skipped)
    context.partial_gaps = []
    try:
        if not pairs:
            return []
        relevant_clusters = {cid for p in pairs for cid in p["cluster_ids"]}
        related = [c for c in connections if set(c.cluster_ids) & relevant_clusters]
        payload = {"task_version": "gaps-v1", "question": context.question, "fix": fix,
                   "min_quote_chars": settings.finding_min_quote_chars,
                   "pairs": [pair_payload(p, source_map, clusters) for p in pairs],
                   "connections": [c.model_dump(mode="json") for c in related]}
        key = cache_key("gaps", {"request": payload, "use_full_text": context.use_full_text,
                                "coverage_schema": CoverageVerdicts.model_json_schema(),
                                "output_schema": GroundedGap.model_json_schema(),
                                "corpus": [p.model_dump(mode="json") for p in papers]}, GapDecisions, settings)
        record = cached(key, context, settings)
        reused = record is not None
        if record is None:
            record = _assess(pairs, payload, source_map, papers, related, settings, context)
            context.cache[key] = record
        merge_record(report, record, context, reused=reused)
        context.partial_gaps = [g.model_copy(deep=True) for g in record.gaps]
        return context.partial_gaps
    finally:
        context.audits.append(report)


def _assess(pairs, payload, source_map, papers, connections, settings, context):
    targets, evidence = {p["pair_id"]: p for p in pairs}, {}
    def validate(output):
        exact_ids([d.pair_id for d in output.decisions], targets, "pair_id")
        for d in output.decisions:
            if d.decision == "propose":
                if (not d.basis or not all(x.strip() for x in (d.research_question, d.proposed_test, d.why_it_matters))
                        or not 1 <= len(d.search_queries) <= settings.gap_coverage_queries_per_gap
                        or any(not q.strip() or len(q) > 240 for q in d.search_queries)):
                    raise ValueError("Propose needs a grounded question, basis, test, importance and bounded positive-study queries")
                scope = unsupported_scope_claims([d.research_question, d.proposed_test, d.why_it_matters],
                                                 [source_map[pid] for pid in targets[d.pair_id]["paper_ids"]])
                if scope:
                    raise ValueError(f"{d.pair_id}: {scope} is not stated in either supplied source. SFT or "
                                     "fine-tuning alone does not specify trained parameters; describe each method "
                                     "exactly as the source does (e.g. 'SFT, parameter scope not specified') or abstain")
                try:
                    evidence[d.pair_id] = resolve_evidence(d.evidence, source_map, set(targets[d.pair_id]["paper_ids"]), settings)
                except ValueError as error:
                    raise ValueError(f"{d.pair_id}: {error}; ONLY these paper IDs are allowed for this pair: "
                                     f"{targets[d.pair_id]['paper_ids']}; received {[r.paper_id for r in d.evidence]}") from error
    proposed = request("gap_finder", payload, GapDecisions, validate, settings, context, "gaps")
    claims = [{"pair_id": d.pair_id, "basis": d.basis, "research_question": d.research_question,
               "proposed_test": d.proposed_test, "why_it_matters": d.why_it_matters,
               "evidence": [e.model_dump() for e in evidence[d.pair_id]],
               "papers": [source_map[pid] for pid in targets[d.pair_id]["paper_ids"]]}
              for d in proposed.decisions if d.decision == "propose"]
    checked = verify(claims, settings, context, "gaps")
    accepted = [d for d in proposed.decisions if d.decision == "propose" and checked[d.pair_id].verdict == "supported"]
    coverage = _coverage(accepted, papers, settings, context) if accepted else {}
    gaps, deps = [], {}
    audit = FindingAudit(stage="gaps", candidate_pairs=len(pairs))
    for d in proposed.decisions:
        verdict, check = checked.get(d.pair_id), coverage.get(d.pair_id)
        kept = check is not None and check.status != "addressed"
        decision = "accepted_provisional" if kept else "addressed" if check is not None else "abstain" if verdict is None else "rejected"
        item = {"pair_id": d.pair_id, "paper_ids": targets[d.pair_id]["paper_ids"],
                "cluster_ids": targets[d.pair_id]["cluster_ids"], "scope": targets[d.pair_id].get("scope", "within_cluster"),
                "proposal": {"research_question": d.research_question, "proposed_test": d.proposed_test,
                             "basis": d.basis, "evidence": [e.model_dump() for e in evidence.get(d.pair_id, [])]} if d.decision == "propose" else None,
                "decision": decision,
                "reason": check.reason if check else verdict.reason if verdict else d.reason}
        if check:
            item["coverage"] = check.model_dump(mode="json")
        audit.decisions.append(item)
        if not kept:
            if verdict is None:
                audit.abstained += 1
            else:
                audit.rejected += 1
            continue
        pair, gid = targets[d.pair_id], "GAP-" + d.pair_id.removeprefix("PAIR-")
        gaps.append(GroundedGap(id=gid, cluster_id=pair.get("anchor_cluster_id", pair["cluster_ids"][0]),
            cluster_ids=pair["cluster_ids"], scope=pair.get("scope", "within_cluster"), between_paper_ids=pair["paper_ids"],
            description=d.research_question, why_it_matters=d.why_it_matters, proposed_test=d.proposed_test,
            basis=d.basis, similarity=pair["similarity"], evidence=evidence[d.pair_id], coverage=check))
        deps[gid] = FindingDependency(global_map=False, cluster_ids=pair["cluster_ids"], paper_ids=pair["paper_ids"],
            finding_ids=[c.id for c in connections if set(c.cluster_ids) & set(pair["cluster_ids"])])
        audit.accepted += 1
    return FindingCacheRecord(stage="gaps", created_at=now(), gaps=gaps, dependencies=deps, audit=audit)


def _coverage(proposals, papers, settings, context):
    searches, all_papers, by_query = {}, {p.id: p for p in papers}, {}
    for proposal in proposals:
        checks = []
        for query in dict.fromkeys(q.strip() for q in proposal.search_queries):
            check_deadline(context)
            if query not in by_query:
                try:
                    # Coverage is not restricted by OA status, dates or citation popularity.
                    timeout = min(settings.gap_coverage_timeout_s, max(.1, context.deadline - time.monotonic()))
                    found = search_openalex(query, SearchCriteria(topic=context.question or "research coverage"),
                        settings.gap_coverage_hits_per_query, open_access_only=False, timeout_s=timeout, attempts=1)
                    by_query[query] = CoverageSearch(query=query, checked_at=now(), status="ok", papers=found)
                    for paper in found:
                        all_papers.setdefault(paper.id, paper)
                except OpenAlexError as error:
                    by_query[query] = CoverageSearch(query=query, checked_at=now(), status="unavailable", error=mask(str(error)))
                if trace := current_trace():
                    trace.log("gap_finder", "targeted coverage search", by_query[query].status,
                              result={"query": query, "paper_ids": [p.id for p in by_query[query].papers]},
                              why="Bounded OpenAlex relevance results including non-OA and older papers; failure is not evidence of absence")
            checks.append(by_query[query])
        searches[proposal.pair_id] = checks
    source_map = sources(list(all_papers.values()), settings, context, limit=settings.gap_coverage_source_chars)
    coverage_payload = {"task_version": "gap-coverage-v1", "question": context.question,
        "hypotheses": [{"gap_id": d.pair_id, "research_question": d.research_question, "proposed_test": d.proposed_test,
                        "searches": [{"query": s.query, "status": s.status, "paper_ids": [p.id for p in s.papers]}
                                     for s in searches[d.pair_id]]} for d in proposals],
        "local_paper_ids": [p.id for p in papers], "papers": list(source_map.values()),
        "papers_without_body": sorted(set(all_papers) - set(source_map))}
    resolved = {}
    def validate(output):
        exact_ids([v.gap_id for v in output.verdicts], searches, "gap_id")
        for verdict in output.verdicts:
            evidence = []
            for ref in verdict.evidence:
                if ref.paper_id not in source_map:
                    raise ValueError("Coverage must reference a supplied original source with body text")
                spans = {s["span_id"]: s["quote"] for s in source_map[ref.paper_id]["spans"]}
                quote = spans.get(ref.span_id, "")
                if len(quote.strip()) < settings.finding_min_quote_chars:
                    raise ValueError("Coverage evidence must select an original sufficiently long span")
                evidence.append(Evidence(paper_id=ref.paper_id, quote=quote))
            if verdict.verdict == "addressed" and not evidence:
                raise ValueError("An addressing study needs exact source evidence, not keyword similarity")
            resolved[verdict.gap_id] = evidence
    output = request("gap_coverage", coverage_payload, CoverageVerdicts, validate, settings, context, "gaps")
    out = {}
    for verdict in output.verdicts:
        checks = searches[verdict.gap_id]
        if verdict.verdict == "addressed":
            status = "addressed"
        elif any(s.status == "unavailable" for s in checks):
            status = "unavailable"
        elif verdict.verdict == "inconclusive" or set(all_papers) - set(source_map):
            status = "inconclusive"
        else:
            status = "no_address_found_in_checked_sources"
        out[verdict.gap_id] = GapCoverage(status=status, checked_at=now(),
            local_papers_checked=len({p.id for p in papers} & set(source_map)), local_papers_considered=len(papers),
            searches=checks, addressing_paper_ids=sorted({e.paper_id for e in resolved[verdict.gap_id]}) if status == "addressed" else [],
            evidence=resolved[verdict.gap_id], reason=verdict.reason)
    return out


def _fake_gaps(papers: list[Paper], clusters: list[Cluster]) -> list[Gap]:
    """Sample mode: one placeholder gap per cluster, quoting the abstract of its first paper that has one."""
    by_id = {p.id: p for p in papers}
    gaps = []
    for c in clusters:
        quotable = [pid for pid in c.paper_ids if pid in by_id and by_id[pid].abstract]
        if not quotable:
            continue
        gaps.append(Gap(
            id=f"{c.id}-G1",
            cluster_id=c.id,
            description="Sample mode: placeholder gap between the first two papers of this group.",
            between_paper_ids=quotable[:2],
            why_it_matters="Sample mode: illustrates the output shape; no actual gap or novelty assessment was performed.",
            evidence=[Evidence(paper_id=quotable[0], quote=first_sentence(by_id[quotable[0]]))],
        ))
    return gaps
