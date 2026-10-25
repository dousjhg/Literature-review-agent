"""Regeneration invariants, full-input cache reuse and version conflicts; all offline."""

import json
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from litagent import api, cluster_labels, cluster_regeneration as regen
from litagent.agents import clusterer, grader
from litagent.cluster_preview import run_cluster_map
from litagent.cluster_state import (ClusteringArtifacts, VersionError, head, load_version, path_for,
                                    save_version, set_head, version_lock)
from litagent.config import Effort, Settings
from litagent.integrations.llm import TokenBudgetExceeded
from litagent.integrations.meterless import ResearchWorld
from litagent.question_clusters import attach_alternatives, label_side_groups
from litagent.schemas import (ClusterAssessmentProposal, ClusterAssessmentProposals, ClusterMapResult, ClusterMove,
    Connection, Evidence, EvidenceReference, FindingDependency, Gap, Grade, Paper, PaperClusterDetail, QuestionCluster)
from litagent.similarity import cosine_matrix


@pytest.fixture
def models(monkeypatch):
    calls = []
    def label(prompt, message, output_type, **kwargs):
        payload = json.loads(message)
        calls.append(payload)
        return ClusterAssessmentProposals(assessments=[ClusterAssessmentProposal(cluster_id=c["cluster_id"],
            label="Grounded experiment group", rationale="The sources report controlled experiments.",
            coherence="coherent", contrast="Evidence from neighboring topics needs human interpretation.",
            evidence=[EvidenceReference(paper_id=p["id"], span_id=p["spans"][0]["span_id"]) for p in c["papers"]])
            for c in payload["groups"]])
    monkeypatch.setattr(cluster_labels, "complete_json", label)
    monkeypatch.setattr(grader, "grade", lambda *a, **kw: Grade(stage="clusters", accuracy=3, appropriateness=3, actionability=2))
    monkeypatch.setattr(clusterer, "embed", lambda *a, **kw: pytest.fail("Regeneration must never embed papers again"))
    return calls


def snapshot(tmp_path, *, neighbors=0, batch_size=1, sample=False, dependency_case=False, vectors=None, settings=None):
    cfg = settings or Settings(sample_mode=sample, logs_dir=str(tmp_path), cluster_label_neighbors=neighbors,
                               cluster_label_batch_size=batch_size)
    ps = [Paper(id=pid, title=f"Controlled task {pid}", source="test",
                abstract=f"We investigate controlled task {pid} using scientific experiments and report direct empirical measurements.")
          for pid in ["M", "A", "B", "C", "D", "E"]]
    vectors = np.asarray(vectors if vectors is not None else [
        [1, 0, 0], [0, 1, 0], [0, .9, .436], [0, .99, .141], [0, 0, 1], [0, .1, .995]], dtype=float)
    arrays = ClusteringArtifacts(paper_vectors=vectors, question_vector=np.array([1., 0, 0]), similarity=cosine_matrix(vectors))
    groups = {"MAIN": ["M"], "S1": ["A", "B"], "S2": ["C"], "S3": ["D", "E"]}
    query_scores = cosine_matrix(np.vstack([arrays.question_vector, vectors]))[0, 1:]
    idx = {p.id: i for i, p in enumerate(ps)}
    clusters = [QuestionCluster(id=cid, label=cid, rationale="Original source grouping", kind="main" if cid == "MAIN" else "side",
        question="Scientific task question", paper_ids=ids,
        paper_details=[PaperClusterDetail(paper_id=pid, question_similarity=float(query_scores[idx[pid]]),
            relevance="direct" if cid == "MAIN" else "supporting", reason="Original source decision",
            evidence=[Evidence(paper_id=pid, quote=ps[idx[pid]].abstract)]) for pid in ids]) for cid, ids in groups.items()]
    labels = label_side_groups(ps, [clusterer._text_for(p, False) for p in ps],
        [[idx[pid] for pid in ids] for cid, ids in groups.items() if cid != "MAIN"], arrays.similarity,
        [cid for cid in groups if cid != "MAIN"], cfg)
    for c in clusters:
        if c.id in labels:
            c.label, c.rationale = labels[c.id].label, labels[c.id].rationale
    attach_alternatives(clusters, ps, arrays.similarity, cfg)
    result = ClusterMapResult(run_id="root", question="Scientific task question", main_similarity_threshold=.5,
        sample_mode=sample, clusters=clusters, main_cluster_id="MAIN", side_linkage=cfg.side_linkage,
        side_similarity_threshold=cfg.side_similarity_threshold)
    deps = {}
    if dependency_case:
        result.connections = [Connection(id=cid, kind="within_cluster", paper_ids=members, cluster_ids=[cluster],
            description="Source-supported connection", evidence=[Evidence(paper_id=members[0], quote=ps[idx[members[0]]].abstract)])
            for cid, members, cluster in [("local", ["D", "E"], "S3"), ("changed", ["A", "B"], "S1"), ("unknown", ["D", "E"], "S3")]]
        result.gaps = [Gap(id="transitive", cluster_id="S3", between_paper_ids=["D", "E"],
            description="Depends on a changed connection", why_it_matters="Controlled experimental question", evidence=[])]
        deps = {"local": FindingDependency(global_map=False, cluster_ids=["S3"]),
                "changed": FindingDependency(global_map=False, cluster_ids=["S1"]),
                "transitive": FindingDependency(global_map=False, finding_ids=["changed"])}
    world = ResearchWorld(result.run_id, cfg.logs_dir)
    world.start_attempt(1, result.question)
    world.add_papers(ps)
    world.write_clusters(clusters)
    world.write_connections(result.connections)
    world.write_gaps(result.gaps)
    result.paper_map = world.to_graph_json()
    save_version(result, ps, Effort.NORMAL, arrays, cfg, dependencies=deps)
    set_head(result, cfg)
    return result, cfg, ps, arrays


def move(pid="A", target="S2"):
    return ClusterMove(paper_id=pid, target_cluster_id=target)


def test_preview_is_zero_call_and_reuses_exactly_unaffected_labels(models, tmp_path):
    root, cfg, _, _ = snapshot(tmp_path)
    models.clear()
    original = path_for(cfg, "cluster-map", "root").read_bytes()
    plan = regen.preview_moves("root", [move()], cfg)
    assert plan.applicable and plan.relabel_cluster_ids == ["S1", "S2"]
    assert plan.reused_label_cluster_ids == ["MAIN", "S3"] and not models
    assert head("root", cfg)["run_id"] == "root"
    assert path_for(cfg, "cluster-map", "root").read_bytes() == original
    assert {c.cluster_id: c.after for c in plan.changes} == {"S1": ["B"], "S2": ["A", "C"]}


def test_apply_preserves_root_and_main_recomputes_hints_and_is_idempotent(models, tmp_path):
    root, cfg, _, _ = snapshot(tmp_path)
    before = {p: path_for(cfg, p, "root").read_bytes() for p in ["cluster-map", "cluster-state", "world"]}
    models.clear()
    plan = regen.preview_moves("root", [move()], cfg)
    child = regen.apply_moves("root", plan.preview_id, cfg)
    assert child.run_id != root.run_id and child.parent_run_id == root.run_id and child.revision == 1
    assert head("root", cfg)["run_id"] == child.run_id
    assert child.clusters[0].paper_ids == root.clusters[0].paper_ids == ["M"]
    assert child.clusters[0].paper_details[0].evidence == root.clusters[0].paper_details[0].evidence
    by_id = {c.id: c for c in child.clusters}
    assert by_id["S1"].paper_ids == ["B"] and by_id["S2"].paper_ids == ["A", "C"]
    assert by_id["S1"].paper_details[0].assigned_similarity is None
    assert all(a.cluster_id != c.id for c in child.clusters for d in c.paper_details for a in d.also_related)
    assert by_id["S3"].label == next(c.label for c in root.clusters if c.id == "S3")
    assert len(models) == 1 and models[0]["target_cluster_ids"] == ["S2"]
    assert child.regeneration.reused_embeddings and child.regeneration.reused_similarity_matrix
    assert all(path_for(cfg, p, "root").read_bytes() == value for p, value in before.items())
    assert sum(e["type"] == "member-of" for e in child.paper_map["edges"]) == 6
    assert regen.apply_moves("root", plan.preview_id, cfg) == child and len(models) == 1
    assert load_version(child.run_id, cfg)[0] == child


@pytest.mark.parametrize("neighbors,batch", [(2, 1), (0, 4)])
def test_neighbor_or_batch_changes_invalidate_unchanged_members(models, tmp_path, neighbors, batch):
    _, cfg, _, _ = snapshot(tmp_path, neighbors=neighbors, batch_size=batch)
    models.clear()
    plan = regen.preview_moves("root", [move()], cfg)
    assert "S3" in plan.relabel_cluster_ids
    child = regen.apply_moves("root", plan.preview_id, cfg)
    assert "S3" in child.regeneration.relabelled_cluster_ids
    assert any("S3" in p["target_cluster_ids"] for p in models)


def test_simultaneous_moves_are_validated_together(models, tmp_path):
    # A and B each fit C at .8 but do not fit each other (.28).
    vectors = [[1, 0, 0], [.8, .6, 0], [.8, -.6, 0], [1, 0, 0], [0, 1, 0], [0, 1, .01]]
    root, cfg, ps, arrays = snapshot(tmp_path, vectors=vectors)
    # Use independently cached singleton source groups to begin with a valid complete-link map.
    s1 = next(c for c in root.clusters if c.id == "S1")
    b_detail = s1.paper_details[1].model_copy(deep=True)
    s1.paper_ids, s1.paper_details = ["A"], [s1.paper_details[0]]
    extra = s1.model_copy(deep=True, update={"id": "S4", "paper_ids": ["B"],
        "paper_details": [b_detail]})
    root.clusters.append(extra)
    root.run_id, root.root_run_id, root.version_token = "simultaneous", None, None
    attach_alternatives(root.clusters, ps, arrays.similarity, cfg)
    save_version(root, ps, Effort.NORMAL, arrays, cfg)
    set_head(root, cfg)
    assert regen.preview_moves(root.run_id, [move("A")], cfg).applicable
    assert regen.preview_moves(root.run_id, [move("B")], cfg).applicable
    bad = regen.preview_moves(root.run_id, [move("A"), move("B")], cfg)
    assert not bad.applicable and "final complete" in bad.errors[0]
    with pytest.raises(VersionError, match="not applicable"):
        regen.apply_moves(root.run_id, bad.preview_id, cfg)


@pytest.mark.parametrize("moves", [[move("M")], [move("A", "MAIN")], [move("A", "S1")],
    [move("unknown")], [move("A", "unknown")], [move(), move()]])
def test_invalid_or_main_moves_are_rejected(models, tmp_path, moves):
    _, cfg, _, _ = snapshot(tmp_path)
    with pytest.raises(VersionError) as error:
        regen.preview_moves("root", moves, cfg)
    assert error.value.status == 422


def test_stale_preview_policy_conflict_rollback_and_lineage_guard(models, tmp_path):
    root, cfg, _, _ = snapshot(tmp_path)
    first = regen.preview_moves("root", [move()], cfg)
    stale = regen.preview_moves("root", [move("B")], cfg)
    with pytest.raises(VersionError, match="policy changed"):
        regen.apply_moves("root", first.preview_id, replace(cfg, llm_model="changed-model"))
    child = regen.apply_moves("root", first.preview_id, cfg)
    with pytest.raises(VersionError, match="no longer active"):
        regen.apply_moves("root", stale.preview_id, cfg)
    with pytest.raises(VersionError, match="Active version changed"):
        regen.activate_version("root", "root", "root", root.version_token, cfg)
    regen.activate_version("root", "root", child.run_id, child.version_token, cfg)
    assert head("root", cfg)["run_id"] == "root"
    # Retrying a completed apply after rollback must not re-activate it.
    assert regen.apply_moves("root", first.preview_id, cfg).run_id == child.run_id
    assert head("root", cfg)["run_id"] == "root"
    history = regen.version_history("root", cfg)
    assert len(history["versions"]) == 2
    other = run_cluster_map([Paper(id="other", title="Other source", source="test")], "Other question",
                             Effort.NORMAL, replace(cfg, sample_mode=True), "other")
    with pytest.raises(VersionError, match="another lineage"):
        regen.activate_version("root", other.run_id, "root", root.version_token, cfg)


def test_dependency_invalidation_is_conservative_and_transitive(models, tmp_path):
    _, cfg, _, _ = snapshot(tmp_path, dependency_case=True)
    plan = regen.preview_moves("root", [move()], cfg)
    assert plan.invalidated_connection_ids == ["changed", "unknown"]
    assert plan.invalidated_gap_ids == ["transitive"]
    child = regen.apply_moves("root", plan.preview_id, cfg)
    assert child.status == "partial" and [c.id for c in child.connections] == ["local"] and not child.gaps
    assert child.regeneration.pending_stages == ["connections", "gaps"]
    assert head("root", cfg)["run_id"] == "root"
    assert sum(e["type"] == "connects-to" for e in child.paper_map["edges"]) == 1
    with pytest.raises(VersionError, match="pending stages"):
        root = load_version("root", cfg)[0]
        regen.activate_version("root", child.run_id, root.run_id, root.version_token, cfg)


def test_grading_budget_retains_child_without_advancing_head(models, tmp_path, monkeypatch):
    _, cfg, _, _ = snapshot(tmp_path)
    monkeypatch.setattr(grader, "grade", lambda *a, **kw: (_ for _ in ()).throw(TokenBudgetExceeded("Budget reached")))
    plan = regen.preview_moves("root", [move()], cfg)
    child = regen.apply_moves("root", plan.preview_id, cfg)
    assert child.status == "partial" and child.grade is None and child.clusters
    assert child.regeneration.pending_stages == ["clusters_grade"]
    assert head("root", cfg)["run_id"] == "root" and load_version(child.run_id, cfg)[0] == child


def test_label_deadline_stops_before_another_call(models, tmp_path):
    _, cfg, _, _ = snapshot(tmp_path)
    cfg = replace(cfg, run_timeout_s=-1)
    models.clear()
    plan = regen.preview_moves("root", [move()], cfg)
    with pytest.raises(TokenBudgetExceeded, match="time limit"):
        regen.apply_moves("root", plan.preview_id, cfg)
    assert not models and head("root", cfg)["run_id"] == "root"


def test_failed_grade_is_retained_without_automatic_moves(models, tmp_path, monkeypatch):
    _, cfg, _, _ = snapshot(tmp_path)
    monkeypatch.setattr(grader, "grade", lambda *a, **kw: Grade(stage="clusters", accuracy=1, appropriateness=1, actionability=1))
    plan = regen.preview_moves("root", [move()], cfg)
    child = regen.apply_moves("root", plan.preview_id, cfg)
    assert child.status == "done" and child.grade.total == 3
    assert head("root", cfg)["run_id"] == child.run_id and any("did not pass" in w for w in child.warnings)
    assert child.regeneration.moved_paper_ids == ["A"]


def test_failed_label_call_never_repeats_automatically(models, tmp_path, monkeypatch):
    _, cfg, _, _ = snapshot(tmp_path)
    plan = regen.preview_moves("root", [move()], cfg)
    def fail(*a, **kw):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr(cluster_labels, "complete_json", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        regen.apply_moves("root", plan.preview_id, cfg)
    assert head("root", cfg)["run_id"] == "root"
    with pytest.raises(VersionError, match="will not repeat paid calls"):
        regen.apply_moves("root", plan.preview_id, cfg)


def test_empty_source_is_removed_without_stale_runners(models, tmp_path):
    _, cfg, _, _ = snapshot(tmp_path)
    plan = regen.preview_moves("root", [move("C", "S1")], cfg)
    assert plan.applicable and next(c.after for c in plan.changes if c.cluster_id == "S2") == []
    child = regen.apply_moves("root", plan.preview_id, cfg)
    assert "S2" not in {c.id for c in child.clusters}
    assert all(d.runner_up_cluster_id != "S2" and all(a.cluster_id != "S2" for a in d.also_related)
               for c in child.clusters for d in c.paper_details)
    assert sum(e["type"] == "member-of" for e in child.paper_map["edges"]) == 6


def test_average_rejects_incoming_paper_hidden_by_strong_existing_pairs(models, tmp_path):
    cfg = Settings(sample_mode=False, logs_dir=str(tmp_path), side_linkage="average", cluster_label_neighbors=0,
                   cluster_label_batch_size=1)
    vectors = [[1, 0, 0], [0, 1, 0], [0, .9, .436], [0, .99, .141], [0, .65, .759934], [0, .65, .759934]]
    _, cfg, _, _ = snapshot(tmp_path, vectors=vectors, settings=cfg)
    plan = regen.preview_moves("root", [move("A", "S3")], cfg)
    assert not plan.applicable and any("mean similarity to final target" in e for e in plan.errors)
    assert regen.preview_moves("root", [move("A", "S2")], cfg).applicable


def test_independent_strategy_has_its_own_cache_and_generation_path(models, tmp_path, monkeypatch):
    from litagent import question_clusters
    from litagent.schemas import ClusterLabel
    calls = []
    def independent(prompt, message, schema, **kw):
        calls.append(json.loads(message))
        return ClusterLabel(label="Independent topic", rationale="Described experiments share a controlled task.")
    monkeypatch.setattr(question_clusters, "complete_json", independent)
    cfg = Settings(sample_mode=False, logs_dir=str(tmp_path), cluster_label_strategy="independent")
    _, cfg, _, _ = snapshot(tmp_path, settings=cfg)
    calls.clear()
    plan = regen.preview_moves("root", [move()], cfg)
    assert plan.reused_label_cluster_ids == ["MAIN", "S3"]
    child = regen.apply_moves("root", plan.preview_id, cfg)
    assert len(calls) == 1 and {p["id"] for p in calls[0]["papers"]} == {"A", "C"}
    assert child.status == "done"


def test_candidate_filters_limits_and_revision_cap_cover_rollback_branches(models, tmp_path):
    root, cfg, _, _ = snapshot(tmp_path)
    choices = regen.candidates("root", cfg)
    assert all(x["paper_id"] != "M" and x["target_cluster_id"] != "MAIN" for x in choices["candidates"])
    assert any(x["paper_id"] == "A" and x["target_cluster_id"] == "S2" and x["eligible"] for x in choices["candidates"])
    limited = regen.candidates("root", replace(cfg, regeneration_candidate_limit=1))
    assert limited["truncated"] and len(limited["candidates"]) == 1
    near = regen.candidates("root", cfg, near_only=True)
    assert all(abs(c["margin"]) <= cfg.related_boundary_margin for c in near["candidates"])
    capped = replace(cfg, regeneration_max_revisions=1)
    plan = regen.preview_moves("root", [move()], capped)
    child = regen.apply_moves("root", plan.preview_id, capped)
    regen.activate_version("root", "root", child.run_id, child.version_token, capped)
    assert not regen.preview_moves("root", [move("B")], capped).applicable
    assert root.version_token == head("root", capped)["version_token"]


@pytest.mark.parametrize("target", ["cluster-map", "cluster-arrays", "cluster-state"])
def test_corrupt_cache_is_rejected_before_model_calls(models, tmp_path, target):
    _, cfg, _, _ = snapshot(tmp_path)
    models.clear()
    path = path_for(cfg, target, "root", "npz" if target == "cluster-arrays" else "json")
    if target == "cluster-arrays":
        path.write_bytes(b"corrupt numeric cache")
    else:
        payload = json.loads(path.read_text())
        if target == "cluster-map": payload["question"] = "Edited outside the version API"
        else: payload["state"]["papers"][0]["title"] = "Changed source title"
        path.write_text(json.dumps(payload))
    with pytest.raises(VersionError, match="integrity"):
        regen.preview_moves("root", [move()], cfg)
    assert not models


def test_legacy_cache_sample_boundary_frozen_scores_and_lock_conflicts(models, tmp_path):
    _, cfg, _, _ = snapshot(tmp_path)
    path_for(cfg, "cluster-map", "legacy").write_bytes(path_for(cfg, "cluster-map", "root").read_bytes())
    with pytest.raises(VersionError, match="no reusable vector cache"):
        regen.preview_moves("legacy", [move()], cfg)
    with pytest.raises(VersionError, match="cannot be mixed"):
        regen.preview_moves("root", [move()], replace(cfg, sample_mode=True))
    # Runtime scoring tweaks cannot silently change a cached lineage's metric or cutoff.
    assert regen.preview_moves("root", [move()], replace(cfg, side_similarity_threshold=1)).applicable
    plan = regen.preview_moves("root", [move()], cfg)
    with version_lock(cfg, "root"):
        with pytest.raises(VersionError, match="Another operation"):
            regen.apply_moves("root", plan.preview_id, cfg)
    with version_lock(cfg, plan.proposed_run_id):
        with pytest.raises(VersionError, match="Another operation"):
            regen.apply_moves("root", plan.preview_id, cfg)


def test_api_sample_preview_apply_history_and_rollback(tmp_path):
    cfg = Settings(sample_mode=True, logs_dir=str(tmp_path))
    api.app.dependency_overrides[api.get_settings] = lambda: cfg
    try:
        with TestClient(api.app) as client:
            papers = [Paper(id=f"P{i}", title=f"Example paper {i}", source="test").model_dump() for i in range(6)]
            body = {"run_id": "api-root", "input": {"query": "Sample research question", "papers": papers}}
            root = client.post("/cluster", json=body).json()["output"]
            assert root["sample_mode"] and root["usage"]["llm_calls"] == 0 and root["version_token"]
            assert client.post("/cluster", json=body).status_code == 409
            assert client.post("/run", json={"run_id": "api-root", "input": {"query": "Sample research question"}}).status_code == 409
            choices = client.get("/cluster/api-root/regenerate/candidates").json()["output"]
            candidate = next(c for c in choices["candidates"] if c["eligible"])
            req = {"moves": [{"paper_id": candidate["paper_id"], "target_cluster_id": candidate["target_cluster_id"]}]}
            plan = client.post("/cluster/api-root/regenerate/preview", json=req).json()["output"]
            assert plan["applicable"]
            child = client.post("/cluster/api-root/regenerate", json={"preview_id": plan["preview_id"]}).json()["output"]
            assert child["usage"]["llm_calls"] == 0 and child["regeneration"]["reused_embeddings"]
            assert client.get(f"/world/{child['run_id']}").status_code == 200
            history = client.get("/cluster/api-root/versions").json()["output"]
            assert history["active_head"]["run_id"] == child["run_id"] and len(history["versions"]) == 2
            reply = client.post("/cluster/api-root/activate", json={"target_run_id": "api-root",
                "expected_head_run_id": child["run_id"], "expected_head_version_token": child["version_token"]})
            assert reply.status_code == 200 and reply.json()["output"]["run_id"] == "api-root"
            assert client.get("/cluster/api-root").json()["output"] == root
            assert client.post("/cluster/api-root/regenerate", json={"preview_id": "../bad"}).status_code == 422
    finally:
        api.app.dependency_overrides.clear()


def test_sample_ids_do_not_alias_paper_nodes_and_question_is_normalized(tmp_path):
    cfg = Settings(sample_mode=True, logs_dir=str(tmp_path))
    papers = [Paper(id=f"C{i}", title=f"Source {i}", source="test") for i in range(1, 4)]
    result = run_cluster_map(papers, "  Sample question  ", Effort.NORMAL, cfg, "sample-identities")
    assert result.question == "Sample question"
    assert not {c.id for c in result.clusters} & {p.id for p in papers}
    assert len({n["id"] for n in result.paper_map["nodes"]}) == len(result.paper_map["nodes"])
    assert all(e["source"] != e["target"] for e in result.paper_map["edges"] if e["type"] == "member-of")
