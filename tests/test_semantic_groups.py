"""The experimental semantic proposal cannot bypass source, coverage or diameter constraints."""

import json

import numpy as np
import pytest

from litagent import semantic_groups
from litagent.agents import clusterer, grader
from litagent.config import Effort, Settings
from litagent.schemas import Cluster, EvidenceReference, Grade, Paper, TopicProposal, TopicProposals
from litagent.similarity import partition_metrics
from litagent.trace import Trace


def papers(n=3):
    return [Paper(id=f"P{i}", title=f"Scientific task {i}", abstract=f"This study investigates topic {i} with direct measurements of the system.", source="test") for i in range(n)]


def one_topic(payload):
    return TopicProposals(topics=[TopicProposal(label="Direct system measurement", rationale="Studies make direct measurements.",
        paper_ids=[p["id"] for p in payload["papers"]],
        evidence=[EvidenceReference(paper_id=p["id"], span_id=p["spans"][0]["span_id"]) for p in payload["papers"]])])


def test_llm_cannot_create_a_similarity_chain(monkeypatch, tmp_path):
    ps = papers()
    matrix = np.array([[1, .8, .3], [.8, 1, .8], [.3, .8, 1]])
    monkeypatch.setattr(semantic_groups, "complete_json", lambda p, m, *a, **kw: one_topic(json.loads(m)))
    with Trace(str(tmp_path)) as trace:
        out = semantic_groups.semantic_clusters(ps, [p.abstract for p in ps], matrix, Settings())
    by_id = {p.id: i for i, p in enumerate(ps)}
    metrics = partition_metrics(matrix, [[by_id[pid] for pid in c.paper_ids] for c in out])
    assert metrics["min_within_similarity"] >= .7
    assert [len(c.paper_ids) for c in out] == [2, 1]
    assert next(e for e in trace.events if e["action"] == "group papers")["projection"]["split_proposals"]


@pytest.mark.parametrize("fault", ["lost_id", "unknown_id", "duplicate_id", "foreign_evidence", "invented_span", "missing_evidence"])
def test_invalid_topic_proposals_are_rejected(fault):
    ps = papers(2)
    sources = {p.id: {"has_body": True, "spans": [{"span_id": "S1", "quote": p.abstract}]} for p in ps}
    proposal = TopicProposals(topics=[TopicProposal(label="Task", rationale="A shared task", paper_ids=["P0", "P1"],
                    evidence=[EvidenceReference(paper_id=p.id, span_id="S1") for p in ps])])
    topic = proposal.topics[0]
    if fault == "lost_id": topic.paper_ids.pop()
    elif fault == "unknown_id": topic.paper_ids[0] = "absent"
    elif fault == "duplicate_id": topic.paper_ids.append("P0")
    elif fault == "foreign_evidence": topic.evidence[0].paper_id = "absent"
    elif fault == "invented_span": topic.evidence[0].span_id = "S99"
    else: topic.evidence.pop()
    with pytest.raises(ValueError):
        semantic_groups.validate_topics(proposal, sources, Settings())


def test_title_only_and_isolated_papers_remain_without_model_calls(monkeypatch):
    ps = papers(3)
    ps[0].abstract = None
    monkeypatch.setattr(semantic_groups, "complete_json", lambda *a, **kw: pytest.fail("No eligible pairs"))
    out = semantic_groups.semantic_clusters(ps, [p.abstract or p.title for p in ps], np.eye(3), Settings())
    assert [c.paper_ids for c in out] == [["P0"], ["P1"], ["P2"]]


def test_omitted_assignment_is_retained_as_an_explicit_singleton(monkeypatch, tmp_path):
    ps = papers()
    def incomplete(prompt, message, *args, **kwargs):
        payload = json.loads(message)
        payload["papers"] = payload["papers"][:2]
        return one_topic(payload)
    monkeypatch.setattr(semantic_groups, "complete_json", incomplete)
    with Trace(str(tmp_path)) as trace:
        result = semantic_groups.semantic_clusters(ps, [p.abstract for p in ps], np.ones((3, 3)), Settings())
    assert [c.paper_ids for c in result] == [["P0", "P1"], ["P2"]]
    event = next(e for e in trace.events if e["action"] == "retain omitted papers")
    assert event["result"]["paper_ids"] == ["P2"]


def test_overlapping_topic_assignment_is_an_abstention_not_an_arbitrary_choice():
    ps = papers(4)
    sources = {p.id: {"title": p.title, "has_body": True,
                       "spans": [{"span_id": "S1", "quote": p.abstract}]} for p in ps}
    topics = [TopicProposal(label="Task", rationale="A shared task", paper_ids=ids,
                            evidence=[EvidenceReference(paper_id=pid, span_id="S1") for pid in ids])
              for ids in [["P0", "P1", "P2"], ["P2", "P3"]]]
    proposal, missing, ambiguous = semantic_groups.retain_unassigned(TopicProposals(topics=topics), sources)
    assert missing == [] and ambiguous == ["P2"]
    assert [t.paper_ids for t in proposal.topics] == [["P0", "P1"], ["P3"], ["P2"]]
    semantic_groups.validate_topics(proposal, sources, Settings())


def test_uncited_member_is_separated_but_an_invented_reference_is_still_rejected():
    ps = papers()
    sources = {p.id: {"title": p.title, "has_body": True,
                       "spans": [{"span_id": "S1", "quote": p.abstract}]} for p in ps}
    proposal = TopicProposals(topics=[TopicProposal(label="A topic", rationale="Shared measurements", paper_ids=list(sources),
                    evidence=[EvidenceReference(paper_id=pid, span_id="S1") for pid in ["P0", "P1"]])])
    result, unsupported = semantic_groups.retain_unsupported(proposal, sources)
    assert unsupported == ["P2"] and [t.paper_ids for t in result.topics] == [["P0", "P1"], ["P2"]]
    semantic_groups.validate_topics(result, sources, Settings())
    result.topics[0].evidence[0].span_id = "invented"
    with pytest.raises(ValueError, match="Invalid/short"):
        semantic_groups.validate_topics(result, sources, Settings())


def test_semantic_mode_does_not_relax_threshold_after_size_feedback(monkeypatch, tmp_path):
    ps = papers(2)
    vectors = np.array([[1, 0], [.68, np.sqrt(1-.68**2)]])
    monkeypatch.setattr(clusterer, "embed", lambda *a: vectors)
    monkeypatch.setattr(semantic_groups, "complete_json", lambda *a, **kw: pytest.fail("Incompatible pair should remain separate"))
    settings = Settings(sample_mode=False, cluster_grouping_strategy="semantic")
    with Trace(str(tmp_path)) as trace:
        result = clusterer.cluster(ps, Effort.NORMAL, settings, fix="Too many single-paper clusters; lower threshold")
    assert [c.paper_ids for c in result] == [["P0"], ["P1"]]
    assert next(e["threshold"] for e in trace.events if e["action"] == "group papers") == .7


def test_semantic_grader_uses_sources_even_when_size_diagnostic_fails(monkeypatch, tmp_path):
    ps = papers(3)
    clusters = [Cluster(id=f"C{i}", label=p.title, paper_ids=[p.id], rationale="") for i, p in enumerate(ps)]
    calls = []
    def judge(*args, **kwargs):
        calls.append(1)
        return Grade(stage="clusters", accuracy=2, appropriateness=1, actionability=2, fix="Check real topic overlap")
    monkeypatch.setattr(grader, "complete_json", judge)
    with Trace(str(tmp_path)) as trace:
        result = grader.grade("clusters", clusters, ps, Settings(sample_mode=False, cluster_grouping_strategy="semantic"))
    assert calls == [1] and result.total == 5
    assert any(e["action"] == "cluster size diagnostic" for e in trace.events)
