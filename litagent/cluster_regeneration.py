"""User-selected, previewed side-cluster moves with versioned local cache reuse."""

from __future__ import annotations

import json
import time
import uuid

import numpy as np

from litagent.agents import grader
from litagent.cluster_state import (VersionError, atomic_json, digest, effective_settings, head, label_keys,
    label_policy, load_version, path_for, save_version, set_head, side_inputs, version_lock)
from litagent.config import EFFORT_PROFILES, Settings
from litagent.integrations.llm import TokenBudgetExceeded, TokenMeter, load_prompt
from litagent.integrations.meterless import ResearchWorld
from litagent.finding_analysis import run_finding_stages
from litagent.finding_engine import FindingContext
from litagent.question_clusters import attach_alternatives, label_side_groups
from litagent.schemas import (ClusterMapResult, ClusterMove, MembershipChange, MoveCandidate, QuestionCluster,
    FindingAnalysis, RegenerationPlan, RegenerationSummary)
from litagent.trace import Trace, new_run_id


def _require_head(result: ClusterMapResult, settings: Settings) -> None:
    active = head(result.root_run_id, settings)
    if active["run_id"] != result.run_id or active["version_token"] != result.version_token:
        raise VersionError("Source version is no longer active; preview against the active head or explicitly activate this version")


def _policy(settings: Settings) -> str:
    return digest({"labels": label_policy(settings), "grader": load_prompt("grader"),
                   "grade_pass_mark": settings.grade_pass_mark, "max_moves": settings.regeneration_max_moves,
                   "max_revisions": settings.regeneration_max_revisions})


def _limits(settings: Settings) -> None:
    if min(settings.regeneration_max_moves, settings.regeneration_candidate_limit, settings.regeneration_max_revisions) < 1:
        raise VersionError("Regeneration limits must be positive", 422)


def version_history(run_id: str, settings: Settings) -> dict:
    result, _, _ = load_version(run_id, settings)
    versions = []
    for path in path_for(settings, "cluster-state", run_id).parent.glob("cluster-state-*.json"):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope["state"].get("root_run_id") != result.root_run_id:
            continue
        version, _, _ = load_version(envelope["state"]["run_id"], settings)
        versions.append({"run_id": version.run_id, "parent_run_id": version.parent_run_id,
                         "revision": version.revision, "status": version.status, "version_token": version.version_token})
    versions.sort(key=lambda v: (v["revision"], v["run_id"]))
    return {"root_run_id": result.root_run_id, "active_head": head(result.root_run_id, settings), "versions": versions}


def candidates(run_id: str, settings: Settings, *, near_only: bool = False, paper_id: str | None = None) -> dict:
    result, state, arrays = load_version(run_id, settings)
    _limits(settings)
    indices = {p.id: i for i, p in enumerate(state.papers)}
    if paper_id is not None and paper_id not in indices:
        raise VersionError("Unknown paper in this map", 404)
    cfg = effective_settings(state, settings)
    items = []
    for source in result.clusters:
        if source.kind == "main":
            continue
        for detail in source.paper_details:
            if paper_id is not None and detail.paper_id != paper_id:
                continue
            i = indices[detail.paper_id]
            for target in result.clusters:
                if target.kind == "main" or target.id == source.id:
                    continue
                scores = arrays.similarity[i, [indices[pid] for pid in target.paper_ids]]
                mean, minimum = float(scores.mean()), float(scores.min())
                margin = detail.assigned_similarity - mean if detail.assigned_similarity is not None else None
                eligible = (minimum if cfg.side_linkage == "complete" else mean) >= cfg.side_similarity_threshold
                if near_only and (margin is None or mean < cfg.related_similarity_threshold or abs(margin) > cfg.related_boundary_margin):
                    continue
                items.append(MoveCandidate(paper_id=detail.paper_id, source_cluster_id=source.id,
                    target_cluster_id=target.id, assigned_similarity=detail.assigned_similarity,
                    target_similarity=mean, target_min_similarity=minimum, margin=margin, eligible=eligible,
                    reason=f"{cfg.side_linkage} cutoff {cfg.side_similarity_threshold:.4f}; individual numerical candidate only. "
                           "A simultaneous-move preview must still validate final membership. This is not a quality recommendation."))
    items.sort(key=lambda x: (not x.eligible, abs(x.margin) if x.margin is not None else float("inf"),
                              -x.target_similarity, x.paper_id, x.target_cluster_id))
    return {"run_id": run_id, "version_token": result.version_token, "active_head": head(result.root_run_id, settings),
            "sample_mode": result.sample_mode, "near_only": near_only, "total_candidates": len(items),
            "truncated": len(items) > settings.regeneration_candidate_limit,
            "candidates": [x.model_dump() for x in items[:settings.regeneration_candidate_limit]]}


def _simulate(result: ClusterMapResult, moves: list[ClusterMove], state, arrays, cfg: Settings):
    if not moves or len(moves) > cfg.regeneration_max_moves:
        raise VersionError(f"Select between 1 and {cfg.regeneration_max_moves} moves", 422)
    if len({m.paper_id for m in moves}) != len(moves):
        raise VersionError("Select each paper at most once", 422)
    clusters = {c.id: c.model_copy(deep=True) for c in result.clusters}
    owner = {pid: c.id for c in result.clusters for pid in c.paper_ids}
    details = {d.paper_id: d.model_copy(deep=True) for c in result.clusters for d in c.paper_details}
    for move in moves:
        if move.paper_id not in owner or move.target_cluster_id not in clusters:
            raise VersionError("Unknown paper or target cluster", 422)
        source, target = clusters[owner[move.paper_id]], clusters[move.target_cluster_id]
        if source.kind == "main" or target.kind == "main":
            raise VersionError("Main membership is frozen; re-assess the question and its source evidence in a new map", 422)
        if source.id == target.id:
            raise VersionError("Source and target cluster must differ", 422)
    # Apply all membership edits before checking scores; sequential admission is unsafe.
    for move in moves:
        clusters[owner[move.paper_id]].paper_ids.remove(move.paper_id)
        clusters[move.target_cluster_id].paper_ids.append(move.paper_id)
    changes = [MembershipChange(cluster_id=c.id, before=c.paper_ids,
                               after=sorted(clusters[c.id].paper_ids)) for c in result.clusters
               if sorted(c.paper_ids) != sorted(clusters[c.id].paper_ids)]
    output = []
    for cluster in clusters.values():
        if not cluster.paper_ids:
            continue
        cluster.paper_ids = sorted(cluster.paper_ids)
        cluster.paper_details = []
        for pid in cluster.paper_ids:
            detail = details[pid]
            detail.runner_up_cluster_id = detail.runner_up_similarity = detail.margin = None
            detail.also_related, detail.near_boundary = [], False
            cluster.paper_details.append(detail)
        output.append(QuestionCluster.model_validate(cluster.model_dump()))
    indices = {p.id: i for i, p in enumerate(state.papers)}
    changed = {c.cluster_id for c in changes}
    errors = []
    for cluster in output:
        if cluster.kind == "main" or cluster.id not in changed or len(cluster.paper_ids) < 2:
            continue
        idx = [indices[pid] for pid in cluster.paper_ids]
        block = arrays.similarity[np.ix_(idx, idx)]
        pairs = block[np.triu_indices(len(idx), 1)]
        criterion = float(pairs.min() if cfg.side_linkage == "complete" else pairs.mean())
        if criterion < cfg.side_similarity_threshold:
            errors.append(f"{cluster.id}: final {cfg.side_linkage} score {criterion:.6f} < {cfg.side_similarity_threshold:.6f}")
        # With average linkage, an incoming paper cannot hide behind many strong old pairs.
        if cfg.side_linkage == "average":
            for move in moves:
                if move.target_cluster_id == cluster.id:
                    rest = [indices[pid] for pid in cluster.paper_ids if pid != move.paper_id]
                    score = float(arrays.similarity[indices[move.paper_id], rest].mean())
                    if score < cfg.side_similarity_threshold:
                        errors.append(f"{move.paper_id}: mean similarity to final target members is below the cutoff")
    attach_alternatives(output, state.papers, arrays.similarity, cfg)
    return output, changes, errors


def invalidated_findings(result, state, changed_clusters: set[str], moved_papers: set[str]):
    findings = [*result.connections, *result.gaps]
    known_findings = {item.id for item in findings}
    known_clusters = {c.id for c in result.clusters}
    known_papers = {p.id for p in state.papers}
    invalid = set()
    while True:
        before = set(invalid)
        for item in findings:
            dep = state.finding_dependencies.get(item.id)
            intrinsic_clusters = set(getattr(item, "cluster_ids", [getattr(item, "cluster_id", "")]))
            intrinsic_papers = set(getattr(item, "paper_ids", getattr(item, "between_paper_ids", [])))
            if (dep is None or dep.global_map or intrinsic_clusters & changed_clusters or intrinsic_papers & moved_papers
                    or set(dep.cluster_ids) & changed_clusters or set(dep.paper_ids) & moved_papers
                    or set(dep.finding_ids) & invalid or set(dep.finding_ids) - known_findings
                    or set(dep.cluster_ids) - known_clusters or set(dep.paper_ids) - known_papers):
                invalid.add(item.id)
        if invalid == before:
            break
    return ([c.id for c in result.connections if c.id in invalid], [g.id for g in result.gaps if g.id in invalid])


def preview_moves(run_id: str, moves: list[ClusterMove], settings: Settings) -> RegenerationPlan:
    _limits(settings)
    result, state, arrays = load_version(run_id, settings)
    # Preview needs no lock or model; apply rechecks the head under a lock.
    _require_head(result, settings)
    cfg = effective_settings(state, settings)
    output, changes, errors = _simulate(result, moves, state, arrays, cfg)
    keys = label_keys(state.papers, output, arrays.similarity, result.use_full_text, cfg)
    dirty = sorted(cid for cid, key in keys.items() if state.label_keys.get(cid) != key)
    reused = [c.id for c in output if c.id not in dirty]
    invalid_connections, invalid_gaps = invalidated_findings(result, state,
        {c.cluster_id for c in changes} | set(dirty), {m.paper_id for m in moves})
    if len(version_history(run_id, settings)["versions"]) - 1 >= cfg.regeneration_max_revisions:
        errors.append("Maximum saved regenerations reached; start a new map")
    plan = RegenerationPlan(preview_id=uuid.uuid4().hex, sample_mode=result.sample_mode,
        source_run_id=run_id, source_version_token=result.version_token,
        root_run_id=result.root_run_id, proposed_run_id=new_run_id(), revision=result.revision + 1,
        moves=moves, changes=changes, relabel_cluster_ids=dirty, reused_label_cluster_ids=reused,
        invalidated_connection_ids=invalid_connections, invalidated_gap_ids=invalid_gaps,
        applicable=not errors, errors=errors)
    payload = {"plan": plan.model_dump(mode="json"), "policy": _policy(cfg), "label_keys": keys}
    atomic_json(path_for(settings, "cluster-preview", plan.preview_id), {"checksum": digest(payload), "payload": payload})
    return plan


def apply_moves(run_id: str, preview_id: str, settings: Settings) -> ClusterMapResult:
    path = path_for(settings, "cluster-preview", preview_id)
    if not path.exists():
        raise VersionError("Unknown move preview", 404)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if envelope["checksum"] != digest(payload):
        raise VersionError("Move preview integrity check failed")
    plan = RegenerationPlan.model_validate(payload["plan"])
    if plan.source_run_id != run_id or plan.preview_id != preview_id:
        raise VersionError("Preview belongs to a different source version", 422)
    if not plan.applicable:
        raise VersionError("Move preview is not applicable: " + "; ".join(plan.errors), 422)
    with version_lock(settings, plan.root_run_id), version_lock(settings, plan.proposed_run_id):
        receipt_path = path_for(settings, "cluster-apply", preview_id)
        if path_for(settings, "cluster-state", plan.proposed_run_id).exists():
            child, _, _ = load_version(plan.proposed_run_id, settings)
            if child.regeneration and child.regeneration.preview_id == preview_id:
                # An idempotent retry returns its version without changing a later rollback/head.
                return child
            raise VersionError("Proposed version ID is already in use")
        if receipt_path.exists():
            raise VersionError("This apply was already started but did not commit. Inspect its trace and create a new preview; it will not repeat paid calls automatically")
        result, state, arrays = load_version(run_id, settings)
        _require_head(result, settings)
        cfg = effective_settings(state, settings)
        if result.version_token != plan.source_version_token or _policy(cfg) != payload["policy"]:
            raise VersionError("Source version or generation policy changed; create a fresh preview")
        if len(version_history(run_id, settings)["versions"]) - 1 >= cfg.regeneration_max_revisions:
            raise VersionError("Maximum saved regenerations reached; start a new map")
        output, changes, errors = _simulate(result, plan.moves, state, arrays, cfg)
        keys = label_keys(state.papers, output, arrays.similarity, result.use_full_text, cfg)
        if errors or keys != payload["label_keys"] or [c.model_dump() for c in changes] != [c.model_dump() for c in plan.changes]:
            raise VersionError("Move inputs changed; create a fresh preview")
        if any(path_for(settings, prefix, plan.proposed_run_id, suffix).exists() for prefix, suffix in (
                ("cluster-map", "json"), ("world", "json"), ("trace", "jsonl"))):
            raise VersionError("Proposed version ID is already in use; create a fresh preview")
        atomic_json(receipt_path, {"status": "running", "child_run_id": plan.proposed_run_id})
        try:
            child = _generate_child(result, state, arrays, output, plan, keys, cfg)
            # Ready snapshots remain inspectable even if grading exhausted its budget.
            if child.status == "done":
                set_head(child, settings)
            atomic_json(receipt_path, {"status": child.status, "child_run_id": child.run_id})
            return child
        except Exception:
            atomic_json(receipt_path, {"status": "failed", "child_run_id": plan.proposed_run_id})
            raise


def _generate_child(source, state, arrays, output, plan, keys, cfg) -> ClusterMapResult:
    started = time.monotonic()
    with Trace(cfg.logs_dir, plan.proposed_run_id) as trace, TokenMeter(EFFORT_PROFILES[state.effort].token_budget) as meter:
        trace.log("clusterer", "regenerate selected papers", "reuse frozen numeric cache", result=plan.model_dump(),
                  why="Explicit simultaneous side moves; main and sources frozen; labels keyed by full batch and neighbor context")
        try:
            texts, groups, ids = side_inputs(state.papers, output, source.use_full_text)
            labels = label_side_groups(state.papers, texts, groups, arrays.similarity, ids, cfg,
                                       only_ids=set(plan.relabel_cluster_ids), deadline=started + cfg.run_timeout_s)
            if set(labels) != set(plan.relabel_cluster_ids):
                raise VersionError("Label cache plan does not match regenerated batch targets")
            for cluster in output:
                if cluster.id in labels:
                    cluster.label, cluster.rationale = labels[cluster.id].label, labels[cluster.id].rationale
            child = source.model_copy(deep=True, update={"run_id": plan.proposed_run_id, "parent_run_id": source.run_id,
                "revision": plan.revision, "version_token": uuid.uuid4().hex, "clusters": output, "status": "done",
                "grade": None, "paper_map": None, "usage": None,
                "warnings": [w for w in source.warnings if w.startswith(("Sample mode:", "No paper passed"))]})
            child.connections = [c for c in source.connections if c.id not in plan.invalidated_connection_ids]
            child.gaps = [g for g in source.gaps if g.id not in plan.invalidated_gap_ids]
            analyze = source.analysis is not None
            if analyze:
                child.connections, child.gaps = [], []
                child.analysis = FindingAnalysis(request_id=plan.preview_id, source_run_id=source.run_id)
            pending = []
            if plan.invalidated_connection_ids and not analyze:
                pending.append("connections")
            if plan.invalidated_gap_ids and not analyze:
                pending.append("gaps")
            child.regeneration = RegenerationSummary(preview_id=plan.preview_id, moved_paper_ids=[m.paper_id for m in plan.moves],
                relabelled_cluster_ids=plan.relabel_cluster_ids, reused_label_cluster_ids=plan.reused_label_cluster_ids,
                invalidated_connection_ids=plan.invalidated_connection_ids, invalidated_gap_ids=plan.invalidated_gap_ids,
                pending_stages=pending)
            world = ResearchWorld(child.run_id, cfg.logs_dir)
            world.start_attempt(1, child.question)
            world.add_papers(state.papers)
            world.write_clusters(output)
            world.write_connections(child.connections)
            world.write_gaps(child.gaps)
            try:
                if time.monotonic() - started > cfg.run_timeout_s:
                    raise TokenBudgetExceeded("Regeneration time limit reached before grading")
                child.grade = grader.grade("clusters", output, state.papers, cfg)
                world.record_grade("clusters", child.grade)
                trace.log("grader", "grade regenerated map", f"{child.grade.total}/9", result=child.grade,
                          why=child.grade.reasoning or child.grade.fix or "Independent whole-map grade")
                if not grader.passed(child.grade, cfg):
                    child.warnings.append("Cluster map did not pass grading; retained for review without automatic regeneration.")
            except TokenBudgetExceeded as error:
                pending.append("clusters_grade")
                child.warnings.append(str(error))
            context = FindingContext(question=child.question, similarity=arrays.similarity, use_full_text=child.use_full_text,
                deadline=started + cfg.run_timeout_s, cache={k: v.model_copy(deep=True) for k, v in state.finding_cache.items()})
            if analyze:
                run_finding_stages(child, state.papers, cfg, world, context)
                pending.extend(child.analysis.pending_stages)
            if pending:
                child.status = "partial"
                child.warnings.append("Pending stages were not recomputed. The source remains active; inspect this child version separately.")
            child.regeneration.pending_stages = pending
            child.paper_map, child.usage = world.to_graph_json(), meter.usage
            dependencies = {fid: dep for fid, dep in (context.dependencies if analyze else state.finding_dependencies).items()
                            if fid in {x.id for x in [*child.connections, *child.gaps]}}
            save_version(child, state.papers, state.effort, arrays, cfg, dependencies=dependencies, keys=keys,
                         finding_cache=context.cache if analyze else state.finding_cache)
            trace.log("clusterer", "save regenerated version", child.status,
                      result={"parent": source.run_id, "relabelled": plan.relabel_cluster_ids,
                              "reused_labels": plan.reused_label_cluster_ids, "pending_stages": pending},
                      why="Original snapshot retained; alternatives recomputed for every paper; no embedding call")
            return child
        except Exception as error:
            trace.error("clusterer", "regenerate selected papers", error, next_step="keep source version active")
            raise


def activate_version(root_id: str, target_id: str, expected_head_id: str, expected_head_token: str,
                     settings: Settings) -> dict:
    with version_lock(settings, root_id):
        active = head(root_id, settings)
        if active["run_id"] != expected_head_id or active["version_token"] != expected_head_token:
            raise VersionError("Active version changed; refresh before activating another version")
        target, _, _ = load_version(target_id, settings)
        if target.root_run_id != root_id:
            raise VersionError("Target belongs to another lineage", 422)
        if target.status != "done":
            raise VersionError("Complete the target version's pending stages before activation")
        set_head(target, settings)
        with Trace(settings.logs_dir, root_id) as trace:
            trace.log("clusterer", "activate saved version", target_id,
                      why=f"Explicit version selection; previous active version {expected_head_id} retained")
        return head(root_id, settings)
