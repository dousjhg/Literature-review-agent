"""Question maps, provenance, boundary scores and the standalone API; no external calls."""

import json
from collections import Counter

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litagent import api, cluster_labels, question_clusters
from litagent.agents import clusterer, grader
from litagent.cluster_preview import run_cluster_map
from litagent.config import Effort, Settings
from litagent.integrations.llm import LLMError, TokenBudgetExceeded
from litagent.integrations.meterless import ResearchWorld, load_world
from litagent.schemas import (ClusterAssessmentProposal, ClusterAssessmentProposals, ClusterMapResult, Evidence,
    EvidenceReference, Grade, Paper, PaperClusterDetail, QuestionCluster, QuestionRelevanceBatch,
    QuestionRelevanceProposal, RunResult)
from litagent.similarity import linkage_groups
from litagent.trace import Trace


def corpus():
    ps = [Paper(id=f"P{i}", title=f"Task P{i}", abstract=f"We investigate task P{i} with controlled scientific experiments and report direct measurements.",
                source="test") for i in range(5)]
    ps[2].abstract = None
    return ps


@pytest.fixture
def models(monkeypatch):
    ps = corpus()
    vectors = {"P0": [.9, .4, 0], "P1": [.8, .6, 0], "P2": [.8, .6, 0],
               "P3": [.2, 0, .98], "P4": [.2, 0, .98]}
    texts = {clusterer._text_for(p, False): p.id for p in ps}
    calls = []
    def embed(values, settings):
        return np.array([vectors[texts[t]] if t in texts else ([0, 0, 1] if t == "Second question" else [1, 0, 0]) for t in values])
    def relevance(prompt, message, output_model, **kwargs):
        payload = json.loads(message)
        calls.append(payload)
        assert output_model is QuestionRelevanceBatch
        direct = {"P3", "P4"} if payload["question"] == "Second question" else {"P0"}
        return QuestionRelevanceBatch(assessments=[QuestionRelevanceProposal(paper_id=p["id"],
            relevance="direct" if p["id"] in direct else "supporting", reason="Explicit task match in supplied source.",
            evidence=[EvidenceReference(paper_id=p["id"], span_id=p["spans"][0]["span_id"])]) for p in payload["papers"]])
    def label(prompt, message, output_model, **kwargs):
        payload = json.loads(message)
        return ClusterAssessmentProposals(assessments=[ClusterAssessmentProposal(cluster_id=c["cluster_id"],
            label="Controlled task experiments", rationale="The examples report controlled experiments.",
            coherence="coherent" if all(p["has_body"] for p in c["papers"]) else "insufficient_evidence",
            contrast="Neighbor groups study a different task; source review remains necessary.",
            evidence=[EvidenceReference(paper_id=p["id"], span_id=p["spans"][0]["span_id"])
                      for p in c["papers"] if p["has_body"]]) for c in payload["groups"]])
    monkeypatch.setattr(clusterer, "embed", embed)
    monkeypatch.setattr(question_clusters, "complete_json", relevance)
    monkeypatch.setattr(cluster_labels, "complete_json", label)
    monkeypatch.setattr(grader, "complete_json", lambda *a, **kw: Grade(stage="clusters", accuracy=3, appropriateness=3, actionability=2))
    return ps, calls


def test_main_requires_both_query_gate_and_source_judgment(models):
    ps, calls = models
    result = clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False), question="First question")
    assert result[0].kind == "main" and result[0].paper_ids == ["P0"]
    assert Counter(pid for c in result for pid in c.paper_ids) == Counter(p.id for p in ps)
    assert all(isinstance(c, QuestionCluster) for c in result)
    by_id = {d.paper_id: d for c in result for d in c.paper_details}
    assert by_id["P1"].question_similarity > .5 and by_id["P1"].relevance == "supporting"
    assert by_id["P2"].relevance == "uncertain" and not by_id["P2"].evidence
    assert by_id["P3"].question_similarity < .5 and by_id["P3"].relevance == "uncertain"
    assert {p["id"] for call in calls for p in call["papers"]} == {"P0", "P1"}
    assert all(e.quote in ps[0].abstract for e in by_id["P0"].evidence)


def test_relevance_changes_with_the_question_and_ids_are_order_stable(models):
    ps, _ = models
    settings = Settings(sample_mode=False)
    first = clusterer.cluster(ps, Effort.NORMAL, settings, question="First question")
    reverse = clusterer.cluster(list(reversed(ps)), Effort.NORMAL, settings, question="First question")
    assert [c.model_dump() for c in first] == [c.model_dump() for c in reverse]
    second = clusterer.cluster(ps, Effort.NORMAL, settings, question="Second question")
    assert second[0].paper_ids == ["P3", "P4"]
    assert first[0].paper_ids == ["P0"]


def test_no_direct_match_is_a_valid_map(models):
    ps, _ = models
    result = clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False, main_similarity_threshold=1), question="First question")
    assert all(c.kind == "side" for c in result)
    assert sum(len(c.paper_ids) for c in result) == len(ps)


def test_main_is_a_relevance_pool_not_a_pairwise_topic_cluster(monkeypatch, models):
    ps = corpus()[:2]
    monkeypatch.setattr(clusterer, "embed", lambda *a: np.array([[1, 0], [.6, .8], [.6, -.8]]))
    def direct(prompt, message, *args, **kwargs):
        payload = json.loads(message)
        return QuestionRelevanceBatch(assessments=[QuestionRelevanceProposal(paper_id=p["id"], relevance="direct", reason="Direct task evidence",
            evidence=[EvidenceReference(paper_id=p["id"], span_id="S1")]) for p in payload["papers"]])
    monkeypatch.setattr(question_clusters, "complete_json", direct)
    result = clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False), question="First question")
    assert len(result) == 1 and result[0].kind == "main"
    assert result[0].paper_details[0].assigned_similarity == pytest.approx(-.28)
    assert all(p.runner_up_cluster_id is None for p in result[0].paper_details)
    assert grader.check_cluster_sizes(result, len(ps)) is None


@pytest.mark.parametrize("fault", ["foreign_paper", "foreign_quote", "invented_span", "uncited_direct"])
def test_relevance_rejects_bad_evidence_after_bounded_repair(monkeypatch, fault):
    ps = corpus()[:1]
    calls = []
    def bad(*args, **kwargs):
        calls.append(1)
        ref = EvidenceReference(paper_id="P0", span_id="S1")
        assessment = QuestionRelevanceProposal(paper_id="P0", relevance="direct", reason="A claimed direct answer", evidence=[ref])
        if fault == "foreign_paper": assessment.paper_id = "invented"
        elif fault == "foreign_quote": ref.paper_id = "P1"
        elif fault == "invented_span": ref.span_id = "S99"
        else: assessment.evidence = []
        return QuestionRelevanceBatch(assessments=[assessment])
    monkeypatch.setattr(question_clusters, "complete_json", bad)
    with pytest.raises(LLMError, match="bounded source"):
        question_clusters.assess_relevance(ps, [ps[0].abstract], np.array([.9]), "A question", Settings(), None)
    assert len(calls) == 2


def test_candidate_batches_are_bounded(models):
    ps, calls = models
    clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False, main_relevance_batch_size=1), question="First question")
    assert len(calls) == 3 and all(len(c["papers"]) == 1 for c in calls)
    assert sum(c["task_version"].startswith("question_relevance_verify") for c in calls) == 1


def test_independent_verifier_can_downgrade_a_broad_initial_match(monkeypatch):
    ps = corpus()[:1]
    observed = []
    def assess(prompt, message, *args, **kwargs):
        payload = json.loads(message)
        observed.append(payload)
        verify = payload["task_version"].startswith("question_relevance_verify")
        assert "reason" not in payload and "relevance" not in payload
        return QuestionRelevanceBatch(assessments=[QuestionRelevanceProposal(paper_id="P0",
            relevance="supporting" if verify else "direct", reason="Required method is not established by the evidence.",
            evidence=[EvidenceReference(paper_id="P0", span_id="S1")])])
    monkeypatch.setattr(question_clusters, "complete_json", assess)
    result = question_clusters.assess_relevance(ps, [ps[0].abstract], np.array([.9]), "A specific method and task", Settings(), None)
    assert result["P0"].relevance == "supporting" and len(observed) == 2


def side(cid, ids):
    return QuestionCluster(id=cid, label="A topic", rationale="A topic", kind="side", question="A question",
        paper_ids=ids, paper_details=[PaperClusterDetail(paper_id=pid, question_similarity=.2, relevance="uncertain", reason="Not assessed") for pid in ids])


def test_alternatives_exclude_self_and_handle_singletons_without_fake_margin():
    ps = corpus()[:3]
    clusters = [side("S1", ["P0", "P1"]), side("S2", ["P2"])]
    matrix = np.array([[1, .6, .7], [.6, 1, .7], [.7, .7, 1]])
    question_clusters.attach_alternatives(clusters, ps, matrix, Settings())
    a, b = clusters[0].paper_details[0], clusters[1].paper_details[0]
    assert a.assigned_similarity == .6 and a.runner_up_similarity == .7
    assert a.margin == pytest.approx(-.1)
    assert b.assigned_similarity is None and b.margin is None and not b.near_boundary
    assert b.runner_up_cluster_id == "S1"
    assert [c.paper_ids for c in clusters] == [["P0", "P1"], ["P2"]]


def test_high_average_does_not_bypass_complete_link_gate():
    ps = corpus()[:3]
    clusters = [side("S1", ["P0"]), side("S2", ["P1", "P2"])]
    matrix = np.array([[1, .9, .3], [.9, 1, .8], [.3, .8, 1]])
    question_clusters.attach_alternatives(clusters, ps, matrix, Settings(related_similarity_threshold=.4))
    alternative = clusters[0].paper_details[0].also_related[0]
    assert alternative.similarity == pytest.approx(.6) and alternative.min_similarity == .3
    assert not alternative.meets_membership_criteria


def test_average_linkage_is_explicit_and_has_a_different_constraint():
    matrix = np.array([[1, .9, .8], [.9, 1, .6], [.8, .6, 1]])
    assert linkage_groups(matrix, .69, "average") == [[0, 1, 2]]
    assert linkage_groups(matrix, .69, "complete") == [[0, 1], [2]]


def test_keywords_are_source_terms_and_cover_empty_vocabulary():
    ps = [Paper(id="A", title="LoRA improves translation", abstract="Basque translation experiments", source="test"),
          Paper(id="B", title="Protein graph folding", abstract="Molecular structure predictions", source="test")]
    terms = question_clusters.paper_keywords(ps, 4)
    assert all(terms.values())
    assert all(term.casefold() in f"{p.title} {p.abstract}".casefold() for p in ps for term in terms[p.id])
    empty = [Paper(id="X", title="the and or", source="test")]
    assert question_clusters.paper_keywords(empty, 4) == {"X": []}


def test_question_metadata_survives_runresult_json_round_trip(models):
    ps, _ = models
    clusters = clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False), question="First question")
    result = RunResult(run_id="test", status="partial", clusters=clusters)
    restored = RunResult.model_validate_json(result.model_dump_json())
    assert isinstance(restored.clusters[0], QuestionCluster)
    assert restored.clusters[0].paper_details[0].evidence == clusters[0].paper_details[0].evidence


def test_world_stores_hints_without_duplicate_memberships_or_paper_entities(models, tmp_path):
    ps, _ = models
    clusters = clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False, related_similarity_threshold=0), question="First question")
    world = ResearchWorld("main-side-world", str(tmp_path))
    world.start_attempt(1, "First question")
    world.add_papers(ps)
    world.write_clusters(clusters)
    graph = world.to_graph_json()
    assert sum(n["type"] == "paper" for n in graph["nodes"]) == len(ps)
    assert sum(e["type"] == "member-of" for e in graph["edges"]) == len(ps)
    assert any(e["type"] == "also-related-to" for e in graph["edges"])
    p0 = next(n for n in graph["nodes"] if n["id"] == "P0")
    assert p0["cluster_detail"]["relevance"] == "direct"
    assert next(n for n in graph["nodes"] if n["id"] == "MAIN")["kind"] == "main"
    assert Counter(pid for ids in world.read_clusters().values() for pid in ids) == Counter(p.id for p in ps)
    world.start_attempt(2, "Changed question")
    world.add_papers(ps)
    assert all("cluster_detail" not in n for n in world.to_graph_json()["nodes"])


def test_preview_is_saved_even_when_later_agents_are_unimplemented(models, tmp_path):
    ps, _ = models
    result = run_cluster_map(ps, "First question", Effort.NORMAL, Settings(sample_mode=False, logs_dir=str(tmp_path)), "preview")
    assert result.stage == "clusters" and result.main_cluster_id == "MAIN" and result.grade.total == 8
    assert ClusterMapResult.model_validate_json((tmp_path / "cluster-map-preview.json").read_text()) == result
    assert load_world("preview", str(tmp_path)) is not None


def test_grading_budget_failure_preserves_finished_map(models, monkeypatch, tmp_path):
    ps, _ = models
    monkeypatch.setattr(grader, "grade", lambda *a, **kw: (_ for _ in ()).throw(TokenBudgetExceeded("Budget reached")))
    result = run_cluster_map(ps, "First question", Effort.NORMAL, Settings(sample_mode=False, logs_dir=str(tmp_path)), "budget")
    assert result.status == "partial" and result.clusters and result.paper_map
    assert result.grade is None and "Budget reached" in result.warnings


def test_new_api_supports_real_contract_with_offline_mocks(models, tmp_path):
    ps, _ = models
    settings = Settings(sample_mode=False, logs_dir=str(tmp_path))
    api.app.dependency_overrides[api.get_settings] = lambda: settings
    try:
        with TestClient(api.app) as client:
            body = {"run_id": "question-api", "input": {"query": "First question", "papers": [p.model_dump() for p in ps]}}
            response = client.post("/cluster", json=body)
            assert response.status_code == 200
            out = response.json()["output"]
            assert out["main_cluster_id"] == "MAIN" and out["clusters"][0]["kind"] == "main"
            assert client.get("/cluster/question-api").json()["output"] == out
            assert client.get("/world/question-api").status_code == 200
            assert client.get("/cluster/missing").status_code == 404
            body["input"]["papers"].append(body["input"]["papers"][0])
            assert client.post("/cluster", json=body).status_code == 422
            body["run_id"] = "../bad"
            assert client.post("/cluster", json=body).status_code == 422
    finally:
        api.app.dependency_overrides.clear()


def test_preview_sample_mode_with_one_paper_is_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(clusterer, "embed", lambda *a, **kw: pytest.fail("Sample mode loaded an encoder"))
    settings = Settings(sample_mode=True, logs_dir=str(tmp_path))
    result = run_cluster_map(corpus()[:1], "First question", Effort.NORMAL, settings, "sample-preview")
    assert result.sample_mode and result.status == "done" and len(result.clusters) == 1
    assert result.main_cluster_id is None and result.usage.llm_calls == 0
    assert all("Sample mode" in c.label or "Sample mode" in c.rationale for c in result.clusters)


def test_pipeline_passes_the_question_and_preserves_map_if_grading_budget_runs_out(monkeypatch, models, tmp_path):
    from litagent import orchestrator
    from litagent.human_review import AutoApproveReviewer
    ps, _ = models
    settings = Settings(sample_mode=True, logs_dir=str(tmp_path))
    real_clusters = clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False), question="First question")
    received = []
    def cluster_stage(papers, effort, settings, fix=None, **kwargs):
        received.append(kwargs["question"])
        return real_clusters
    monkeypatch.setattr(clusterer, "cluster", cluster_stage)
    monkeypatch.setattr(orchestrator.retriever, "retrieve", lambda *a, **kw: ps)
    monkeypatch.setattr(grader, "grade", lambda *a, **kw: (_ for _ in ()).throw(TokenBudgetExceeded("Budget reached while grading")))
    result = orchestrator.run_pipeline("How do LLMs handle low-resource languages?", Effort.NORMAL, AutoApproveReviewer(), settings)
    assert received == [result.plan.question]
    assert result.status == "partial" and isinstance(result.clusters[0], QuestionCluster)
    assert any(n.get("kind") == "main" for n in result.paper_map["nodes"])


def test_subtopic_review_excludes_main_pairs_from_scoring_and_sampling():
    from evals.cluster_pairs import paper_splits, prepare_review, score_reviews
    ps = [Paper(id=f"R{i}", title=f"Distinct topic {i}", abstract=f"This source describes controlled experiment number {i} and its observations.",
                source="test") for i in range(20)]
    fold = paper_splits(ps)
    eligible = [p for p in ps if fold[p.id] == "development"]
    assert len(eligible) >= 2
    main_paper = eligible[0]
    main = QuestionCluster(id="MAIN", kind="main", question="A broad research question", label="Main", rationale="",
        paper_ids=[main_paper.id], paper_details=[PaperClusterDetail(paper_id=main_paper.id,
        question_similarity=.9, relevance="direct", reason="Direct source support",
        evidence=[Evidence(paper_id=main_paper.id, quote=main_paper.abstract)])])
    side_ids = [p.id for p in ps if p.id != main_paper.id]
    from litagent.schemas import Cluster
    clusters = [main, Cluster(id="S", label="Side", rationale="", paper_ids=side_ids)]
    matrix = np.full((len(ps), len(ps)), .5)
    np.fill_diagonal(matrix, 1)
    pack, _ = prepare_review(ps, clusters, matrix, 2)
    assert all(main_paper.id not in review["paper_ids"] for review in pack["reviews"])
    pack["reviews"].append({"paper_ids": [main_paper.id, eligible[1].id], "judgment": None})
    result = score_reviews(pack, ps, clusters)
    assert result["evaluation_scope"] == "side_topics_only" and result["main_pairs_excluded"] == 1
    assert result["scored"] == 0 and result["accuracy"] is None
