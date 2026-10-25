"""Run actual scientific finders on immutable cached maps, or resume a partial analysis."""

from __future__ import annotations

import json
import time
import uuid

from openai import OpenAIError

from litagent.agents import connection_finder, gap_finder, grader
from litagent.cluster_state import (VersionError, atomic_json, effective_settings, head, load_version, path_for,
                                    save_version, set_head, version_lock)
from litagent.config import EFFORT_PROFILES, Settings
from litagent.finding_engine import FindingContext, check_deadline
from litagent.integrations.llm import LLMError, TokenMeter
from litagent.integrations.meterless import ResearchWorld
from litagent.schemas import FindingAnalysis
from litagent.trace import Trace, current_trace, mask, new_run_id


def run_finding_stages(result, papers, settings, world, context: FindingContext) -> None:
    """Completed tasks survive a provider/budget failure; an explicit new request can resume."""
    analysis = result.analysis
    stage, phase, written = "connections", "produce", set()
    try:
        for stage in ("connections", "gaps"):
            phase = "produce"
            check_deadline(context)
            if stage == "connections":
                result.connections = connection_finder.find_connections(papers, result.clusters, settings, context=context)
            else:
                result.gaps = gap_finder.find_gaps(papers, result.clusters, result.connections, settings, context=context)
            output = getattr(result, stage)
            {"connections": world.write_connections, "gaps": world.write_gaps}[stage](output)
            written.add(stage)
            phase = "grade"
            check_deadline(context)
            grade = grader.grade(stage, output, papers, settings, **grader.finding_scope(context, stage))
            if grade.stage != stage:
                raise LLMError("Independent grader returned the wrong finding stage")
            analysis.grades[stage] = grade
            world.record_grade(stage, grade)
            analysis.completed_stages.append(stage)
            if trace := current_trace():
                trace.log("grader", "grade " + stage, f"{grade.total}/9", result=grade,
                          why=grade.reasoning or grade.fix or "Independent source-based finding grade")
            if not grader.passed(grade, settings):
                result.warnings.append(f"{stage} did not pass independent grading; retained without automatic score-driven regeneration.")
    except (LLMError, OpenAIError) as error:
        result.status = "partial"
        if stage == "connections" and phase == "produce":
            result.connections = list(context.partial_connections)
        elif stage == "gaps" and phase == "produce":
            result.gaps = list(context.partial_gaps)
        if stage not in written:
            {"connections": world.write_connections, "gaps": world.write_gaps}[stage](getattr(result, stage))
        analysis.pending_stages = (["connections_grade" if phase == "grade" else "connections", "gaps"]
                                   if stage == "connections" else ["gaps_grade" if phase == "grade" else "gaps"])
        result.warnings.append(mask(str(error)) + "; completed finding tasks retained. Resume this partial version with a new analysis request.")
    finally:
        analysis.audits = context.audits
    if any(g.coverage.status in {"unavailable", "inconclusive"} for g in result.gaps if hasattr(g, "coverage")):
        result.warnings.append("Some research hypotheses have unavailable or inconclusive coverage checks; global novelty is not established.")


def _active_or_partial_descendant(source, settings):
    active = head(source.root_run_id, settings)
    current, visited = source, set()
    while current.run_id != active["run_id"] and current.status == "partial" and current.parent_run_id:
        if current.run_id in visited:
            raise VersionError("Invalid cyclic version ancestry")
        visited.add(current.run_id)
        current, _, _ = load_version(current.parent_run_id, settings)
    if current.run_id != active["run_id"]:
        raise VersionError("Source is no longer active; refresh or explicitly activate it before analysis")


def analyze_cluster_map(run_id: str, expected_version_token: str, settings: Settings,
                        request_id: str | None = None):
    request_id = request_id or uuid.uuid4().hex
    source, state, arrays = load_version(run_id, settings)
    with version_lock(settings, source.root_run_id):
        receipt_path = path_for(settings, "cluster-analysis", request_id)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["source_run_id"] != run_id or receipt["source_version_token"] != expected_version_token:
                raise VersionError("Analysis request ID belongs to another source version")
            if path_for(settings, "cluster-state", receipt["child_run_id"]).exists():
                child, _, _ = load_version(receipt["child_run_id"], settings)
                return child
            raise VersionError("This request already started without committing. Inspect its trace and use a new request ID for a deliberate retry")
        if source.version_token != expected_version_token:
            raise VersionError("Source version token changed; refresh the map")
        _active_or_partial_descendant(source, settings)
        cfg = effective_settings(state, settings)
        child_id = new_run_id()
        receipt = {"status": "running", "source_run_id": run_id, "source_version_token": expected_version_token,
                   "child_run_id": child_id}
        with version_lock(settings, child_id):
            atomic_json(receipt_path, receipt)
            try:
                child = _analyze(source, state, arrays, cfg, child_id, request_id)
                if child.status == "done":
                    set_head(child, settings)
                atomic_json(receipt_path, {**receipt, "status": child.status})
                return child
            except Exception:
                atomic_json(receipt_path, {**receipt, "status": "failed"})
                raise


def _analyze(source, state, arrays, cfg, child_id, request_id):
    with Trace(cfg.logs_dir, child_id) as trace, TokenMeter(EFFORT_PROFILES[state.effort].token_budget) as meter:
        trace.log("orchestrator", "analyze cached question map", "reuse papers, embeddings, membership and labels",
                  result={"source_run_id": source.run_id, "request_id": request_id},
                  why="Actual source-grounded finders; immutable child version with its own Nightshift partition")
        child = source.model_copy(deep=True, update={"run_id": child_id, "stage": "findings", "status": "done",
            "parent_run_id": source.run_id, "revision": source.revision + 1, "version_token": uuid.uuid4().hex,
            "regeneration": None, "connections": [], "gaps": [], "paper_map": None,
            "analysis": FindingAnalysis(request_id=request_id, source_run_id=source.run_id),
            "warnings": [w for w in source.warnings if w.startswith(("Sample mode:", "No paper passed", "Cluster map did not pass"))]})
        context = FindingContext(question=child.question, similarity=arrays.similarity, use_full_text=child.use_full_text,
            deadline=time.monotonic() + cfg.run_timeout_s, cache={k: v.model_copy(deep=True) for k, v in state.finding_cache.items()})
        world = ResearchWorld(child_id, cfg.logs_dir)
        world.start_attempt(1, child.question)
        world.add_papers(state.papers)
        world.write_clusters(child.clusters)
        if child.grade:
            world.record_grade("clusters", child.grade)
        if child.grade is None:
            try:
                check_deadline(context)
                child.grade = grader.grade("clusters", child.clusters, state.papers, cfg)
                world.record_grade("clusters", child.grade)
                if not grader.passed(child.grade, cfg):
                    child.warnings.append("Cluster map did not pass grading; findings remain subject to map quality review.")
            except (LLMError, OpenAIError) as error:
                child.status = "partial"
                child.analysis.pending_stages = ["clusters_grade", "connections", "gaps"]
                child.warnings.append(mask(str(error)))
        if child.grade is not None:
            run_finding_stages(child, state.papers, cfg, world, context)
        child.paper_map, child.usage = world.to_graph_json(), meter.usage
        ids = {f.id for f in [*child.connections, *child.gaps]}
        save_version(child, state.papers, state.effort, arrays, cfg, keys=state.label_keys,
            dependencies={k: v for k, v in context.dependencies.items() if k in ids}, finding_cache=context.cache)
        trace.log("orchestrator", "save finding version", child.status,
                  result={"connections": len(child.connections), "gaps": len(child.gaps),
                          "pending_stages": child.analysis.pending_stages},
                  why="Grade failures remain visible; source snapshots are unchanged; hypotheses never establish global novelty")
        return child
