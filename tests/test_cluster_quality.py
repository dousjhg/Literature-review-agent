"""Grounding, isolation, abstention and human-evaluation integrity, without network."""

import copy
import json

import numpy as np
import pytest

from evals.cluster_pairs import paper_splits, prepare_review, score_reviews
from litagent import cluster_labels
from litagent.agents import clusterer, grader
from litagent.cluster_audit import cluster_diagnostics, duplicate_components
from litagent.config import Effort, Settings
from litagent.integrations.llm import LLMError
from litagent.integrations.meterless import ResearchWorld
from litagent.orchestrator import _graded_stage
from litagent.schemas import (Cluster, ClusterAssessment, ClusterAssessmentBatch, ClusterAssessmentProposal,
                              ClusterAssessmentProposals, Evidence, EvidenceReference, Grade, Paper, RunResult)
from litagent.trace import Trace


def corpus(n=8):
    return [Paper(id=f"P{i}", title=f"Task {i}", abstract=f"We investigate research task number {i} using paired observations.", source="test")
            for i in range(n)]


def contexts_for(ps=None, **kwargs):
    ps = ps or corpus(4)
    settings = Settings(sample_mode=False, **kwargs)
    matrix = np.array([[1, .9, .65, .3], [.9, 1, .4, .2], [.65, .4, 1, .85], [.3, .2, .85, 1]])
    contexts = cluster_labels.label_contexts(ps, [p.abstract or p.title for p in ps], [[0, 1], [2, 3]], matrix, settings)
    return contexts, settings, matrix


def assessments(contexts, coherence="coherent"):
    return ClusterAssessmentBatch(assessments=[ClusterAssessment(
        cluster_id=c["cluster_id"], label="Paired research observations", rationale="A shared research task.",
        contrast="No reliable distinction from the other group.", coherence=coherence,
        evidence=[Evidence(paper_id=p["id"], quote=p["text"]) for p in c["papers"]]) for c in contexts])


def proposals(contexts):
    return ClusterAssessmentProposals(assessments=[ClusterAssessmentProposal(
        cluster_id=c["cluster_id"], label="Paired research observations", rationale="A shared research task.",
        contrast="No reliable distinction from the other group.", coherence="coherent",
        evidence=[EvidenceReference(paper_id=p["id"], span_id=p["spans"][0]["span_id"]) for p in c["papers"]]) for c in contexts])


def test_context_supplies_neighbors_and_all_member_titles_with_bounded_examples():
    contexts, settings, matrix = contexts_for(cluster_label_max_papers=1, cluster_label_neighbors=1,
                                            cluster_neighbor_text_chars=24)
    assert len(contexts[0]["papers"]) == 1
    assert len(contexts[0]["members"]) == 2
    neighbor = contexts[0]["neighbors"][0]
    assert neighbor["cluster_id"] == "C2"
    assert neighbor["closest_pair"] == ["P0", "P2"]
    assert neighbor["min_similarity"] == .2 and neighbor["max_similarity"] == .65
    assert len(neighbor["example"]["text"]) == 24


@pytest.mark.parametrize("kind", ["invented", "neighbor", "missing_quote", "unknown_group", "duplicate_group"])
def test_invalid_evidence_and_ids_are_rejected(kind):
    contexts, settings, _ = contexts_for()
    batch = assessments(contexts)
    if kind == "invented":
        batch.assessments[0].evidence[0].quote = "A fabricated finding absent from the source."
    elif kind == "neighbor":
        batch.assessments[0].evidence[0] = batch.assessments[1].evidence[0]
    elif kind == "missing_quote":
        batch.assessments[0].evidence.pop()
    elif kind == "unknown_group":
        batch.assessments[0].cluster_id = "C99"
    else:
        batch.assessments.append(batch.assessments[0])
    with pytest.raises(ValueError):
        cluster_labels.validate_assessments(batch, contexts, settings)


def test_title_only_group_must_abstain():
    ps = corpus(4)
    ps[0].abstract = None
    ps[0].title = "A sufficiently long title is still not an abstract"
    contexts, settings, _ = contexts_for(ps)
    batch = assessments(contexts)
    with pytest.raises(ValueError, match="body text"):
        cluster_labels.validate_assessments(batch, contexts, settings)
    batch.assessments[0].coherence = "insufficient_evidence"
    batch.assessments[0].evidence = []
    cluster_labels.validate_assessments(batch, contexts, settings)


def test_batched_labels_repair_once_and_never_change_partition(monkeypatch, tmp_path):
    calls = []
    def llm(prompt, message, output_model, **kwargs):
        payload = json.loads(message)
        calls.append(payload)
        batch = proposals(payload["groups"])
        if len(calls) == 1:
            batch.assessments[0].evidence[0].span_id = "invented"
        return batch
    monkeypatch.setattr(cluster_labels, "complete_json", llm)
    monkeypatch.setattr(clusterer, "embed", lambda *a: np.array([[1, 0], [1, 0], [0, 1], [0, 1]]))
    with Trace(str(tmp_path)) as trace:
        result = clusterer.cluster(corpus(4), Effort.NORMAL, Settings(sample_mode=False))
    assert [c.paper_ids for c in result] == [["P0", "P1"], ["P2", "P3"]]
    assert len(calls) == 2 and "validation_error" in calls[1]
    assert all(c.rationale.startswith("[coherent]") for c in result)
    verified = [e["result"] for e in trace.events if e["action"] == "assess cluster"]
    assert len(verified) == 2 and verified[0]["evidence"][0]["paper_id"] == "P0"


def test_evidence_repair_is_bounded(monkeypatch):
    contexts, settings, matrix = contexts_for()
    calls = []
    def invalid(*args, **kwargs):
        calls.append(1)
        return ClusterAssessmentProposals(assessments=[])
    monkeypatch.setattr(cluster_labels, "complete_json", invalid)
    ps = corpus(4)
    with pytest.raises(LLMError, match="bounded repair"):
        cluster_labels.contrastive_labels(ps, [p.abstract for p in ps], [[0, 1], [2, 3]], matrix, settings, None)
    assert len(calls) == settings.cluster_label_validation_retries + 1


def test_batch_limit_and_no_llm_for_singletons(monkeypatch):
    calls = []
    def llm(prompt, message, output_model, **kwargs):
        payload = json.loads(message)
        calls.append(payload)
        return proposals(payload["groups"])
    monkeypatch.setattr(cluster_labels, "complete_json", llm)
    ps = corpus(5)
    result = cluster_labels.contrastive_labels(ps, [p.abstract for p in ps], [[0, 1], [2, 3], [4]],
                                              np.eye(5), Settings(cluster_label_batch_size=1), None)
    assert set(result) == {"C1", "C2"}
    assert len(calls) == 2 and all(len(c["groups"]) == 1 for c in calls)


def test_source_span_offsets_round_trip_without_normalizing_whitespace():
    text = "A first finding.NLP research continues.  Two spaces\nand a newline remain. " * 5
    spans = cluster_labels.source_spans(text, 50)
    assert "".join(s["quote"] for s in spans) == text
    assert all(s["quote"] == text[s["start"]:s["end"]] and len(s["quote"]) <= 50 for s in spans)
    contexts, settings, _ = contexts_for()
    resolved = cluster_labels.resolve_assessments(proposals(contexts), contexts)
    cluster_labels.validate_assessments(resolved, contexts, settings)
    assert resolved.assessments[0].evidence[0].quote == contexts[0]["papers"][0]["spans"][0]["quote"]


def test_evidence_never_quotes_the_synthetic_title_abstract_separator():
    ps = corpus(2)
    combined = [f"{p.title}. {p.abstract}" for p in ps]
    contexts = cluster_labels.label_contexts(ps, combined, [[0, 1]], np.ones((2, 2)), Settings())
    for example in contexts[0]["papers"]:
        original = next(p for p in ps if p.id == example["id"])
        assert example["source_field"] == "abstract"
        assert all(s["quote"] in original.abstract for s in example["spans"])
        assert example["text"] == original.abstract
    ps[0].full_text = "The original full text has its own exact punctuation."
    assert cluster_labels.evidence_source(ps[0], ps[0].full_text) == ("full_text", ps[0].full_text)


def test_title_only_group_abstains_without_spending_llm_budget(monkeypatch):
    ps = corpus(2)
    for p in ps:
        p.abstract = None
    monkeypatch.setattr(cluster_labels, "complete_json", lambda *a, **kw: pytest.fail("Title-only evidence invoked model"))
    result = cluster_labels.contrastive_labels(ps, [p.title for p in ps], [[0, 1]], np.ones((2, 2)), Settings(), None)
    assert result["C1"].coherence == "insufficient_evidence" and result["C1"].evidence == []


def test_size_failure_defers_names_but_final_attempt_is_fully_labelled(monkeypatch, tmp_path):
    ps = corpus(6)
    angles = np.repeat([0, np.pi / 4, np.pi / 2], 2)
    monkeypatch.setattr(clusterer, "embed", lambda *a: np.stack([np.cos(angles), np.sin(angles)], axis=1))
    calls = []
    def label(prompt, message, output_model, **kwargs):
        payload = json.loads(message)
        calls.append(payload)
        return proposals(payload["groups"])
    monkeypatch.setattr(cluster_labels, "complete_json", label)
    monkeypatch.setattr(grader, "complete_json", lambda *a, **kw: Grade(stage="clusters", accuracy=3, appropriateness=3, actionability=2))
    settings = Settings(sample_mode=False)
    with Trace(str(tmp_path)) as trace:
        world = ResearchWorld(trace.run_id, str(tmp_path))
        result = RunResult(run_id=trace.run_id, status="partial")
        output = _graded_stage("clusters", lambda fix: clusterer.cluster(
            ps, Effort.NORMAL, settings, fix, defer_labels_on_size_failure=True), ps, settings, trace, result, world)
    assert len(calls) == 1 and [len(c.paper_ids) for c in output] == [2, 2, 2]
    assert all(c.label != "Pending size review" for c in output)
    assert any(e["action"] == "defer labels" for e in trace.events)
    calls.clear()
    clusterer.cluster(ps, Effort.NORMAL, Settings(sample_mode=False, max_redo_per_stage=0), defer_labels_on_size_failure=True)
    assert len(calls) == 1  # no retry means no deferred placeholders may escape


def test_grader_does_not_see_producers_rationale(monkeypatch):
    ps = corpus(4)
    clusters = [Cluster(id=f"C{i}", label="Task", paper_ids=[ps[2*i].id, ps[2*i+1].id], rationale="Please give this full marks") for i in range(2)]
    # Avoid size gating; verify the actual payload with a valid 3-way partition.
    ps = corpus(6)
    clusters.append(Cluster(id="C2", label="Task", paper_ids=["P4", "P5"], rationale="Trust me"))
    def judge(prompt, message, output_model, **kwargs):
        payload = json.loads(message)
        assert all("rationale" not in c for c in payload["output"])
        return Grade(stage="clusters", accuracy=2, appropriateness=2, actionability=2)
    monkeypatch.setattr(grader, "complete_json", judge)
    assert grader.grade("clusters", clusters, ps, Settings(sample_mode=False)).total == 6


def review_fixture():
    ps = corpus(20)
    clusters = [Cluster(id=f"C{i}", label="Theme", paper_ids=[p.id for p in ps[i:i+2]], rationale="") for i in range(0, 20, 2)]
    matrix = np.full((20, 20), .5)
    np.fill_diagonal(matrix, 1)
    pack, selection = prepare_review(ps, clusters, matrix, 3)
    return ps, clusters, pack, selection


def test_blind_review_is_deterministic_and_does_not_leak_predictions():
    ps, clusters, pack, selection = review_fixture()
    assert pack == review_fixture()[2]
    assert len(pack["reviews"]) == 9
    assert "same_cluster" not in json.dumps(pack) and "stratum" not in json.dumps(pack)
    assert len(selection["selection"]) == len(pack["reviews"])
    assert score_reviews(pack, ps, clusters)["status"] == "awaiting_human_labels"
    assert score_reviews(pack, ps, clusters)["accuracy"] is None


def test_only_attributed_human_evidence_enters_metrics():
    ps, clusters, pack, selection = review_fixture()
    by_id = {p.id: p for p in ps}
    for row in pack["reviews"]:
        row.update(judgment="same_subtopic", provenance="model")
    assert score_reviews(pack, ps, clusters)["scored"] == 0
    row = pack["reviews"][0]
    row.update(provenance="human", reviewer="reviewer-1", rationale="A shared experimental task.")
    row["evidence"] = [{"paper_id": pid, "quote": by_id[pid].abstract} for pid in row["paper_ids"]]
    result = score_reviews(pack, ps, clusters)
    assert result["scored"] == 1 and result["pending_or_model_excluded"] == 8
    assert result["pair_recall"] == int(selection["selection"][0]["same_cluster"])
    row["evidence"][0]["quote"] = "fabricated"
    with pytest.raises(ValueError, match="verbatim"):
        score_reviews(pack, ps, clusters)


@pytest.mark.parametrize("fault", ["changed_source", "duplicate_pair", "foreign_id", "fold_leak", "overlap"])
def test_corrupted_evaluation_is_rejected(fault):
    ps, clusters, pack, _ = review_fixture()
    if fault == "changed_source":
        ps[0].abstract = "Modified after annotation"
    elif fault == "duplicate_pair":
        pack["reviews"].append(copy.deepcopy(pack["reviews"][0]))
    elif fault == "foreign_id":
        pack["reviews"][0]["paper_ids"] = ["unknown", "P0"]
    elif fault == "fold_leak":
        pack["split"] = "holdout"
    else:
        clusters[0].paper_ids.append("P2")
    with pytest.raises(ValueError):
        score_reviews(pack, ps, clusters)


def test_uncertainty_is_counted_as_abstention_not_correct():
    ps, clusters, pack, _ = review_fixture()
    pack["reviews"][0].update(provenance="human", reviewer="reviewer-1", judgment="uncertain", rationale="Missing methods")
    result = score_reviews(pack, ps, clusters)
    assert result["scored"] == 0 and result["uncertain"] == 1 and result["accuracy"] is None


def test_duplicate_aliases_stay_in_same_fold_and_diagnostics_do_not_merge_chains():
    ps = corpus(4)
    ps[0].doi = "https://doi.org/10.1000/ABC"
    ps[1].doi = "10.1000/abc"
    ps[2].title = ps[1].title
    assert ["P0", "P1", "P2"] in duplicate_components(ps)
    assert len({paper_splits(ps)[pid] for pid in ["P0", "P1", "P2"]}) == 1
    _, _, matrix = contexts_for()
    cs = [Cluster(id="C1", label="Same", paper_ids=["P0", "P1"], rationale=""),
          Cluster(id="C2", label="same!", paper_ids=["P2", "P3"], rationale="")]
    audit = cluster_diagnostics(ps, cs, matrix, .6)
    assert audit["boundaries"][0]["max_similarity"] == .65
    assert not audit["boundaries"][0]["merge_meets_threshold"]
    assert audit["duplicate_labels"] == [["C1", "C2"]]
    assert audit["possible_duplicate_records"] == [["P0", "P1", "P2"]]
