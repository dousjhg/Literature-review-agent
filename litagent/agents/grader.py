"""Grading agent. Spec: docs/agents/grading_agent.md

One grader with a different rubric per stage (clusters, connections, gaps).
Scores accuracy, appropriateness and actionability from 0 to 3 each (out of 9);
passes at settings.grade_pass_mark (default 7). On a fail it returns one concrete fix.

Cheap rule-based checks run first; the LLM grader only runs if they pass.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from litagent.config import CLUSTER_MAX_PAPER_FRACTION, CLUSTER_MAX_SINGLETON_FRACTION, Settings
from litagent.integrations.llm import complete_json, load_prompt
from litagent.schemas import Cluster, Grade, Paper, QuestionCluster
from litagent.trace import current_trace

Stage = Literal["clusters", "connections", "gaps"]


def grade(stage: Stage, output: list[BaseModel], papers: list[Paper], settings: Settings,
          *, question: str | None = None, reviewed_pairs: list[dict] | None = None) -> Grade:
    if stage == "clusters":
        problem = check_cluster_sizes(output, len(papers))  # type: ignore[arg-type]
        question_map = any(isinstance(c, QuestionCluster) for c in output)
        if problem and not question_map and (settings.sample_mode or settings.cluster_grouping_strategy != "semantic"):
            return Grade(stage=stage, accuracy=1, appropriateness=1, actionability=1, fix=problem, reasoning="Rule-based size check failed")
        if problem and (trace := current_trace()):
            trace.log("grader", "cluster size diagnostic", "semantic review required", why=problem,
                      next_step="judge against source papers")

    if settings.sample_mode:
        return Grade(stage=stage, accuracy=3, appropriateness=3, actionability=2, reasoning="Sample mode: placeholder grade")

    referenced = {pid for item in output for pid in getattr(item, "paper_ids", getattr(item, "between_paper_ids", []))}
    if stage != "clusters":
        by_id = {p.id: p for p in papers}
        for item in output:
            ids = set(getattr(item, "paper_ids", getattr(item, "between_paper_ids", [])))
            if len(ids) != 2 or not ids <= set(by_id) or {e.paper_id for e in item.evidence} != ids or any(
                len(e.quote.strip()) < settings.finding_min_quote_chars or not any(e.quote in source for source in
                    (by_id[e.paper_id].abstract, by_id[e.paper_id].full_text) if source) for e in item.evidence):
                return Grade(stage=stage, accuracy=0, appropriateness=0, actionability=0,
                    fix="Remove unsupported findings; cite original body spans from both supplied papers.",
                    reasoning="Deterministic original-source evidence validation failed")
            if stage == "gaps" and hasattr(item, "coverage") and item.coverage.status == "addressed":
                return Grade(stage=stage, accuracy=0, appropriateness=0, actionability=0,
                    fix="Remove hypotheses already addressed by an evidenced study.", reasoning="Coverage counterevidence contradicts the reported gap")

    # The grader sees the output and the evidence, not the producing agent's reasoning,
    # so it isn't talked into agreeing with it.
    payload = {
            "stage": stage,
            "output": [o.model_dump(mode="json", exclude={"rationale", "paper_details"} if stage == "clusters" else set()) for o in output],
            "question": question or next((c.question for c in output if isinstance(c, QuestionCluster)), None),
            "main_selection_evidence": [e.model_dump() for c in output if isinstance(c, QuestionCluster) and c.kind == "main"
                                        for p in c.paper_details for e in p.evidence],
            "papers": [{"id": p.id, "title": p.title, "abstract": p.abstract} for p in papers],
        }
    if stage != "clusters":
        payload["scope"] = "Only the supplied reviewed_pairs were assessed under resource caps, not every possible pair in the corpus. Abstention is valid. No accepted finding does not prove that none exist. Do not demand a gap outside the reviewed pairs or specific research question. Gaps are provisional hypotheses, never established global novelty."
        payload["reviewed_pairs"] = reviewed_pairs
        referenced |= {pid for pair in reviewed_pairs or [] for pid in pair["paper_ids"]}
        for item in payload["output"]:
            item.pop("verification", None)
            item.pop("similarity", None)
            if "coverage" in item:
                item["coverage"].pop("reason", None)
        payload["papers"] = [{"id": p.id, "title": p.title,
            "abstract": (p.abstract or "")[:settings.finding_grader_source_chars],
            "cited_original_spans": [{"field": field, "quote": e.quote} for item in output for e in item.evidence
                if e.paper_id == p.id for field in ("abstract", "full_text") if getattr(p, field) and e.quote in getattr(p, field)]}
            for p in papers if (not referenced and reviewed_pairs is None) or p.id in referenced]
    return complete_json(load_prompt("grader"), json.dumps(payload, ensure_ascii=False), Grade, settings=settings, agent_name="grader")


def finding_scope(context, stage: str) -> dict:
    """Give the grader candidate scope, never producer/verifier decisions or reasoning."""
    pairs = {d["pair_id"]: {"pair_id": d["pair_id"], "paper_ids": d["paper_ids"]}
             for audit in context.audits if audit.stage == stage for d in audit.decisions}
    return {"question": context.question, "reviewed_pairs": list(pairs.values())}


def passed(g: Grade, settings: Settings) -> bool:
    return g.total >= settings.grade_pass_mark


def check_cluster_sizes(clusters: list[Cluster], n_papers: int) -> str | None:
    """Too broad: one cluster holds most papers. Too detailed: mostly single-paper clusters."""
    if not clusters or n_papers == 0:
        return "No clusters were produced."
    if any(isinstance(c, QuestionCluster) for c in clusters):
        clusters = [c for c in clusters if isinstance(c, QuestionCluster) and c.kind == "side"]
        n_papers = sum(len(c.paper_ids) for c in clusters)
        if len(clusters) < 2:
            return None  # Main is a relevance pool; a single side topic need not be split for size.
    largest = max(len(c.paper_ids) for c in clusters)
    singles = sum(1 for c in clusters if len(c.paper_ids) == 1)
    if largest / n_papers > CLUSTER_MAX_PAPER_FRACTION:
        return "One cluster is too broad; split the largest cluster."
    if singles / len(clusters) > CLUSTER_MAX_SINGLETON_FRACTION:
        return "Too many single-paper clusters; lower the similarity threshold or merge."
    return None
