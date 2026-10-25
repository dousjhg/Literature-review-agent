"""Source fidelity, abstention, counterevidence, exact task reuse and immutable analysis."""

import json
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from litagent import api, cluster_labels, finding_engine as engine
from litagent.agents import connection_finder, gap_finder, grader
from litagent.agents.grader import grade as real_grade
from litagent.cluster_regeneration import apply_moves, preview_moves
from litagent.cluster_state import VersionError, head, load_version, path_for
from litagent.config import Settings
from litagent.finding_analysis import analyze_cluster_map
from litagent.finding_engine import FindingContext
from litagent.integrations.llm import LLMError, TokenBudgetExceeded
from litagent.integrations.meterless import ResearchWorld
from litagent.schemas import (Cluster, ClusterAssessmentProposal, ClusterAssessmentProposals, ClusterMove,
    ConnectionDecision, ConnectionDecisions, CoverageVerdict, CoverageVerdicts, EvidenceReference,
    FindingVerification, FindingVerifications, GapDecision, GapDecisions, Grade, GroundedConnection, GroundedGap, Paper, RunResult)
from litagent.tools.openalex import OpenAlexError
from tests.test_cluster_regeneration import snapshot


@pytest.fixture
def model(monkeypatch):
    state = {"calls": [], "searches": [], "abstain": False, "reject": False, "covered": None, "search_error": False,
             "bad_refs": False, "fail_after": None, "hits": []}
    def complete(prompt, message, schema, **kwargs):
        data = json.loads(message)
        state["calls"].append((schema, data))
        if state["fail_after"] is not None and len(state["calls"]) > state["fail_after"]:
            raise TokenBudgetExceeded("Test model budget exhausted")
        if schema is ConnectionDecisions:
            return ConnectionDecisions(decisions=[ConnectionDecision(pair_id=p["pair_id"],
                decision="abstain" if state["abstain"] else "supported", relation=None if state["abstain"] else "shared_task",
                description="" if state["abstain"] else "Both papers report controlled experiments on a scientific task.",
                reason="Selected original text supports the bounded task statement.",
                evidence=[] if state["abstain"] else [EvidenceReference(paper_id=x["id"],
                    span_id="invented" if state["bad_refs"] else x["spans"][0]["span_id"]) for x in p["papers"]]) for p in data["pairs"]])
        if schema is FindingVerifications:
            assert all("reason" not in c and "decision" not in c and "similarity" not in c for c in data["claims"])
            return FindingVerifications(verdicts=[FindingVerification(pair_id=c["pair_id"],
                verdict="insufficient" if state["reject"] else "supported", reason="Independent source check.") for c in data["claims"]])
        if schema is GapDecisions:
            return GapDecisions(decisions=[GapDecision(pair_id=p["pair_id"], decision="propose",
                basis="cross_paper_hypothesis", research_question="Could the two controlled task approaches be compared under a shared evaluation?",
                proposed_test="Compare both approaches with the same dataset and report held-out accuracy.",
                why_it_matters="Controls evaluation differences between the two scientific approaches.", reason="A testable source-based hypothesis.",
                search_queries=["controlled task comparison evaluation"],
                evidence=[EvidenceReference(paper_id=x["id"], span_id=x["spans"][0]["span_id"]) for x in p["papers"]]) for p in data["pairs"]])
        if schema is CoverageVerdicts:
            return CoverageVerdicts(verdicts=[CoverageVerdict(gap_id=h["gap_id"],
                verdict="addressed" if state["covered"] else "not_addressed_in_checked_sources", reason="Checked the supplied original source excerpts.",
                evidence=[EvidenceReference(paper_id=state["covered"], span_id="S1")] if state["covered"] else []) for h in data["hypotheses"]])
        raise AssertionError(schema)
    def label(prompt, message, schema, **kwargs):
        data = json.loads(message)
        return ClusterAssessmentProposals(assessments=[ClusterAssessmentProposal(cluster_id=c["cluster_id"], label="Controlled experiments",
            rationale="Original experiment texts.", coherence="coherent", contrast="Neighboring source topics differ.",
            evidence=[EvidenceReference(paper_id=p["id"], span_id=p["spans"][0]["span_id"]) for p in c["papers"]]) for c in data["groups"]])
    def search(query, criteria, limit, open_access_only, **kwargs):
        state["searches"].append((query, criteria, limit, open_access_only, kwargs))
        if state["search_error"]:
            raise OpenAlexError("service unavailable")
        return state["hits"]
    monkeypatch.setattr(engine, "complete_json", complete)
    monkeypatch.setattr(cluster_labels, "complete_json", label)
    monkeypatch.setattr(gap_finder, "search_openalex", search)
    monkeypatch.setattr(grader, "grade", lambda stage, *a, **kw: Grade(stage=stage, accuracy=3, appropriateness=3, actionability=2))
    monkeypatch.setattr(engine, "embed_documents", lambda *a, **kw: pytest.fail("Finders should reuse the cached matrix"))
    return state


def context(root, arrays, **kw):
    return FindingContext(question=root.question, similarity=arrays.similarity, **kw)


def test_connections_have_both_original_quotes_verified_semantics_and_typed_graph(model, tmp_path):
    root, cfg, papers, arrays = snapshot(tmp_path)
    ctx = context(root, arrays)
    found = connection_finder.find_connections(papers, root.clusters, cfg, context=ctx)
    assert found and all(isinstance(c, GroundedConnection) for c in found)
    assert {c.kind for c in found} == {"within_cluster", "between_clusters"}
    by_id = {p.id: p for p in papers}
    assert all({e.paper_id for e in c.evidence} == set(c.paper_ids) and
               all(e.quote in by_id[e.paper_id].abstract for e in c.evidence) for c in found)
    assert all(not dep.global_map and dep.cluster_ids for dep in ctx.dependencies.values())
    result = RunResult(run_id="shape", status="done", connections=found)
    assert isinstance(RunResult.model_validate_json(result.model_dump_json()).connections[0], GroundedConnection)
    world = ResearchWorld("finding-shape", str(tmp_path))
    world.start_attempt(1, root.question)
    world.add_papers(papers)
    world.write_clusters(root.clusters)
    world.write_connections(found)
    assert all(e["verification"] == "source_supported" for e in world.to_graph_json()["edges"] if e["type"] == "connects-to")


def test_bad_source_spans_fail_after_bounded_repair(model, tmp_path):
    root, cfg, papers, arrays = snapshot(tmp_path)
    model["bad_refs"] = True
    with pytest.raises(LLMError, match="bounded source validation"):
        connection_finder.find_connections(papers, root.clusters, cfg, context=context(root, arrays))
    assert len(model["calls"]) == cfg.finding_validation_retries + 1


def test_independent_verifier_can_reject_plausible_but_unsupported_claims(model, tmp_path):
    root, cfg, papers, arrays = snapshot(tmp_path)
    model["reject"] = True
    ctx = context(root, arrays)
    assert connection_finder.find_connections(papers, root.clusters, cfg, context=ctx) == []
    assert ctx.audits[0].rejected > 0 and ctx.audits[0].accepted == 0


@pytest.mark.parametrize("abstain", [False, True])
def test_exact_task_cache_reuses_accepted_and_negative_decisions(model, tmp_path, abstain):
    root, cfg, papers, arrays = snapshot(tmp_path)
    model["abstain"] = abstain
    ctx = context(root, arrays)
    initial = connection_finder.find_connections(papers, root.clusters, cfg, context=ctx)
    count = len(model["calls"])
    repeated = connection_finder.find_connections(papers, root.clusters, cfg, context=ctx)
    assert repeated == initial and len(model["calls"]) == count
    assert ctx.audits[-1].reused_tasks and not ctx.audits[-1].generated_tasks
    connection_finder.find_connections(papers, root.clusters, replace(cfg, llm_model="different-model"), context=ctx)
    assert len(model["calls"]) > count


def test_title_only_and_duplicate_studies_do_not_generate_evidence(model):
    ps = [Paper(id="A", title="Same original study", doi="10.1000/test", abstract="Source reports a controlled scientific experiment with measured outcomes.", source="test"),
          Paper(id="B", title="Alternate record", doi="https://doi.org/10.1000/test", abstract="Source reports a controlled scientific experiment with measured outcomes.", source="test"),
          Paper(id="C", title="Only a title", source="test")]
    clusters = [Cluster(id="S", label="test", rationale="", paper_ids=[p.id for p in ps])]
    ctx = FindingContext(similarity=np.full((3, 3), .8))
    assert connection_finder.find_connections(ps, clusters, Settings(sample_mode=False), context=ctx) == []
    assert ctx.audits[0].duplicate_pairs_skipped == 1 and not model["calls"]


def test_gap_is_provisional_and_search_checks_older_non_oa_sources(model, tmp_path):
    root, cfg, papers, arrays = snapshot(tmp_path)
    ctx = context(root, arrays)
    found = gap_finder.find_gaps(papers, root.clusters, [], cfg, context=ctx)
    assert found and isinstance(found[0], GroundedGap) and found[0].hypothesis_status == "provisional"
    assert found[0].coverage.status == "no_address_found_in_checked_sources"
    assert found[0].coverage.local_papers_checked == len(papers)
    _, criteria, limit, oa_only, opts = model["searches"][0]
    assert not oa_only and criteria.year_from is None and criteria.min_citations is None
    assert limit == cfg.gap_coverage_hits_per_query and opts["attempts"] == 1 and opts["timeout_s"] <= cfg.gap_coverage_timeout_s
    assert set(next(data for schema, data in model["calls"] if schema is CoverageVerdicts)["local_paper_ids"]) == {p.id for p in papers}
    count, searches = len(model["calls"]), len(model["searches"])
    assert gap_finder.find_gaps(papers, root.clusters, [], cfg, context=ctx) == found
    assert len(model["calls"]) == count and len(model["searches"]) == searches
    gap_finder.find_gaps(papers, root.clusters, [], replace(cfg, gap_coverage_ttl_s=0), context=ctx)
    assert len(model["searches"]) > searches


@pytest.mark.parametrize("external", [False, True])
def test_coverage_suppresses_hypotheses_answered_by_local_or_external_study(model, tmp_path, external):
    root, cfg, papers, arrays = snapshot(tmp_path)
    if external:
        answer = Paper(id="EXTERNAL", title="A controlled comparison", abstract="We compare both scientific task approaches on the same dataset and report held-out accuracy.", source="openalex")
        model["hits"], model["covered"] = [answer], answer.id
    else:
        model["covered"] = "D"  # Not a boundary member; the whole local corpus is checked.
    ctx = context(root, arrays)
    assert gap_finder.find_gaps(papers, root.clusters, [], cfg, context=ctx) == []
    decision = next(d for d in ctx.audits[0].decisions if d["decision"] == "addressed")
    assert decision["coverage"]["addressing_paper_ids"] == [model["covered"]]
    assert decision["coverage"]["evidence"][0]["quote"]


def test_unavailable_search_never_becomes_negative_novelty_evidence(model, tmp_path):
    root, cfg, papers, arrays = snapshot(tmp_path)
    model["search_error"] = True
    ctx = context(root, arrays)
    found = gap_finder.find_gaps(papers, root.clusters, [], cfg, context=ctx)
    assert found[0].coverage.status == "unavailable" and found[0].hypothesis_status == "provisional"
    count = len(model["searches"])
    gap_finder.find_gaps(papers, root.clusters, [], cfg, context=ctx)
    assert len(model["searches"]) > count  # A failed search is not reusable absence evidence.


def test_missing_body_coverage_reports_considered_and_actually_quoted_counts(model, tmp_path):
    root, cfg, papers, arrays = snapshot(tmp_path)
    papers[-1].abstract = None
    found = gap_finder.find_gaps(papers, root.clusters, [], cfg, context=context(root, arrays))
    assert found[0].coverage.local_papers_considered == 6
    assert found[0].coverage.local_papers_checked == 5
    assert found[0].coverage.status == "inconclusive"


def test_grader_receives_question_and_only_reviewed_scope_without_verifier_reasoning(model, tmp_path, monkeypatch):
    root, cfg, papers, arrays = snapshot(tmp_path)
    ctx = context(root, arrays)
    found = connection_finder.find_connections(papers, root.clusters, cfg, context=ctx)
    seen = []
    def grade_response(prompt, message, schema, **kw):
        seen.append(json.loads(message))
        return Grade(stage="connections", accuracy=2, appropriateness=2, actionability=2)
    monkeypatch.setattr(grader, "complete_json", grade_response)
    real_grade("connections", found, papers, cfg, **grader.finding_scope(ctx, "connections"))
    assert seen[0]["question"] == root.question
    assert all(set(p) == {"pair_id", "paper_ids"} for p in seen[0]["reviewed_pairs"])
    assert all("verification" not in c and "similarity" not in c for c in seen[0]["output"])
    real_grade("gaps", [], papers, cfg, question=root.question,
               reviewed_pairs=[{"pair_id": "pair", "paper_ids": ["A", "B"]}])
    assert {p["id"] for p in seen[1]["papers"]} == {"A", "B"}


def test_invented_addressing_study_cannot_suppress_a_hypothesis(model, tmp_path):
    root, cfg, papers, arrays = snapshot(tmp_path)
    model["covered"] = "NOT_A_SUPPLIED_STUDY"
    with pytest.raises(LLMError, match="Coverage must reference"):
        gap_finder.find_gaps(papers, root.clusters, [], cfg, context=context(root, arrays))


def test_supplied_full_text_is_quoted_only_when_selected_and_titles_are_never_substitutes(model):
    ps = [Paper(id=pid, title=f"Title {pid}", abstract=None, full_text=f"Full source {pid} reports a controlled scientific experiment and its independently measured outcome.", source="test")
          for pid in ["A", "B"]]
    cs = [Cluster(id="S", label="Task", rationale="", paper_ids=["A", "B"])]
    matrix = np.array([[1, .8], [.8, 1]])
    found = connection_finder.find_connections(ps, cs, Settings(sample_mode=False),
        context=FindingContext(similarity=matrix, use_full_text=True))
    assert found and all(e.quote in next(p.full_text for p in ps if p.id == e.paper_id) for e in found[0].evidence)
    count = len(model["calls"])
    assert connection_finder.find_connections(ps, cs, Settings(sample_mode=False), context=FindingContext(similarity=matrix)) == []
    assert len(model["calls"]) == count


def test_analysis_api_commits_child_without_touching_source_and_idempotent_retry(model, tmp_path):
    root, cfg, _, _ = snapshot(tmp_path)
    before = path_for(cfg, "cluster-map", root.run_id).read_bytes()
    api.app.dependency_overrides[api.get_settings] = lambda: cfg
    try:
        with TestClient(api.app) as client:
            req = {"expected_version_token": root.version_token, "request_id": "a" * 32}
            out = client.post("/cluster/root/analyze", json=req)
            assert out.status_code == 200, out.json()
            child = out.json()["output"]
            assert child["stage"] == "findings" and child["analysis"]["completed_stages"] == ["connections", "gaps"]
            assert child["connections"] and child["gaps"] and child["parent_run_id"] == "root"
            assert child["connections"][0]["relation"] == "shared_task" and child["gaps"][0]["hypothesis_status"] == "provisional"
            assert path_for(cfg, "cluster-map", "root").read_bytes() == before
            assert head("root", cfg)["run_id"] == child["run_id"]
            count = len(model["calls"])
            assert client.post("/cluster/root/analyze", json=req).json()["output"] == child
            assert len(model["calls"]) == count
            assert client.post("/cluster/root/analyze", json={**req, "request_id": "b" * 32}).status_code == 409
            saved, state, _ = load_version(child["run_id"], cfg)
            assert state.finding_cache and state.finding_dependencies
            assert isinstance(saved.connections[0], GroundedConnection) and isinstance(saved.gaps[0], GroundedGap)
    finally:
        api.app.dependency_overrides.clear()


def test_partial_analysis_preserves_completed_tasks_and_resume_reuses_them(model, tmp_path):
    root, cfg, _, _ = snapshot(tmp_path)
    model["fail_after"] = 2  # First producer + independent verifier complete one cache task.
    partial = analyze_cluster_map("root", root.version_token, cfg, "c" * 32)
    assert partial.status == "partial" and partial.connections and not partial.gaps
    assert load_version(partial.run_id, cfg)[1].finding_cache and head("root", cfg)["run_id"] == "root"
    model["fail_after"] = None
    resumed = analyze_cluster_map(partial.run_id, partial.version_token, cfg, "d" * 32)
    assert resumed.status == "done" and resumed.analysis.completed_stages == ["connections", "gaps"]
    assert resumed.analysis.audits[0].reused_tasks >= 1 and not resumed.analysis.pending_stages
    assert head("root", cfg)["run_id"] == resumed.run_id


def test_regeneration_runs_real_finders_for_analyzed_maps_and_reuses_unaffected_tasks(model, tmp_path):
    root, cfg, _, _ = snapshot(tmp_path)
    analyzed = analyze_cluster_map("root", root.version_token, cfg, "e" * 32)
    plan = preview_moves(analyzed.run_id, [ClusterMove(paper_id="A", target_cluster_id="S2")], cfg)
    child = apply_moves(analyzed.run_id, plan.preview_id, cfg)
    assert child.status == "done" and child.analysis.completed_stages == ["connections", "gaps"]
    assert not child.regeneration.pending_stages and child.analysis.audits[0].reused_tasks >= 1
    assert child.clusters[0].paper_ids == ["M"]
    assert not set(plan.invalidated_gap_ids) & {g.id for g in child.gaps}
