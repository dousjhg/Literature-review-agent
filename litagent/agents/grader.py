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
from litagent.schemas import Cluster, Grade, Paper
from litagent.trace import current_trace

Stage = Literal["clusters", "connections", "gaps"]


def grade(stage: Stage, output: list[BaseModel], papers: list[Paper], settings: Settings) -> Grade:
    if stage == "clusters":
        problem = check_cluster_sizes(output, len(papers))  # type: ignore[arg-type]
        if problem and (settings.sample_mode or settings.cluster_grouping_strategy != "semantic"):
            return Grade(stage=stage, accuracy=1, appropriateness=1, actionability=1, fix=problem, reasoning="Rule-based size check failed")
        if problem and (trace := current_trace()):
            trace.log("grader", "cluster size diagnostic", "semantic review required", why=problem,
                      next_step="judge against source papers")

    if settings.sample_mode:
        return Grade(stage=stage, accuracy=3, appropriateness=3, actionability=2, reasoning="Sample mode: placeholder grade")

    # The grader sees the output and the evidence, not the producing agent's reasoning,
    # so it isn't talked into agreeing with it.
    user_message = json.dumps(
        {
            "stage": stage,
            "output": [o.model_dump(mode="json", exclude={"rationale"} if stage == "clusters" else set()) for o in output],
            "papers": [{"id": p.id, "title": p.title, "abstract": p.abstract} for p in papers],
        },
        ensure_ascii=False,
    )
    return complete_json(load_prompt("grader"), user_message, Grade, settings=settings, agent_name="grader")


def passed(g: Grade, settings: Settings) -> bool:
    return g.total >= settings.grade_pass_mark


def check_cluster_sizes(clusters: list[Cluster], n_papers: int) -> str | None:
    """Too broad: one cluster holds most papers. Too detailed: mostly single-paper clusters."""
    if not clusters or n_papers == 0:
        return "No clusters were produced."
    largest = max(len(c.paper_ids) for c in clusters)
    singles = sum(1 for c in clusters if len(c.paper_ids) == 1)
    if largest / n_papers > CLUSTER_MAX_PAPER_FRACTION:
        return "One cluster is too broad; split the largest cluster."
    if singles / len(clusters) > CLUSTER_MAX_SINGLETON_FRACTION:
        return "Too many single-paper clusters; lower the similarity threshold or merge."
    return None
