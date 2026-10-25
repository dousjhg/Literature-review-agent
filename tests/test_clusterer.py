"""Real clustering contracts, boundary cases, and the grading/redo integration (no network)."""

import json
from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from litagent.agents import clusterer, grader
from litagent.config import Effort, Settings
from litagent.integrations.meterless import ResearchWorld
from litagent.orchestrator import _graded_stage
from litagent.schemas import ClusterLabel, Grade, Paper, RunResult
from litagent.trace import Trace


def papers(n=6):
    return [Paper(id=f"P{i}", title=f"Paper {i}", abstract=f"Abstract {i}", source="test") for i in range(n)]


@pytest.fixture
def labeller(monkeypatch):
    calls = []

    def label(prompt, message, output_model, **kwargs):
        payload = json.loads(message)
        calls.append(payload)
        assert output_model is ClusterLabel
        return ClusterLabel(label="Cross-lingual transfer", rationale="The papers study cross-lingual transfer.")

    monkeypatch.setattr(clusterer, "complete_json", label)
    return calls


def test_groups_are_labelled_and_every_paper_is_kept_once(monkeypatch, labeller, tmp_path):
    monkeypatch.setattr(clusterer, "embed", lambda texts, settings: np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]))
    with Trace(str(tmp_path)) as trace:
        out = clusterer.cluster(papers(5), Effort.NORMAL, Settings(sample_mode=False, cluster_label_strategy="independent", cluster_threshold_mode="fixed", cluster_dense_weight=1))
    assert [c.paper_ids for c in out] == [["P0", "P1"], ["P2", "P3"], ["P4"]]
    assert Counter(p for c in out for p in c.paper_ids) == Counter(p.id for p in papers(5))
    assert [c.id for c in out] == ["C1", "C2", "C3"]
    assert len(labeller) == 2  # singletons are explicitly retained without inventing a shared theme
    assert out[-1].label.startswith("Unclustered:")
    grouping = next(e for e in trace.events if e["action"] == "group papers")
    assert grouping["threshold"] == 0.7 and grouping["sizes"]["counts"] == [2, 2, 1]


def test_high_effort_uses_available_full_text_and_logs_missing_text(monkeypatch, labeller, tmp_path):
    seen = []

    def embedding(texts, settings):
        seen.extend(texts)
        return np.ones((2, 2))

    monkeypatch.setattr(clusterer, "embed", embedding)
    ps = papers(2)
    ps[0].full_text = "Full text with scientific findings."
    with Trace(str(tmp_path)) as trace:
        clusterer.cluster(ps, Effort.HIGH, Settings(sample_mode=False, cluster_label_strategy="independent", cluster_threshold_mode="fixed", cluster_dense_weight=1), "Use precise labels")
    assert seen == [ps[0].full_text, "Paper 1. Abstract 1"]
    assert labeller[0]["fix"] == "Use precise labels"
    assert any(e["decision"] == "metadata fallback" for e in trace.events)


def test_grader_redo_splits_broad_components_and_world_retains_memberships(monkeypatch, labeller, tmp_path):
    # The 45-degree bridges join every paper at 0.7; a 0.75 redo separates three pairs.
    angles = np.repeat([0, np.pi / 4, np.pi / 2], 2)
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    monkeypatch.setattr(clusterer, "embed", lambda texts, settings: embeddings)
    monkeypatch.setattr(grader, "complete_json", lambda *a, **kw: Grade(
        stage="clusters", accuracy=3, appropriateness=3, actionability=2, reasoning="Specific themes"))
    settings = Settings(sample_mode=False, cluster_label_strategy="independent", cluster_threshold_mode="fixed", cluster_dense_weight=1)
    ps = papers()
    with Trace(str(tmp_path)) as trace:
        world = ResearchWorld(trace.run_id)
        world.start_attempt(1)
        world.add_papers(ps)
        result = RunResult(run_id=trace.run_id, status="partial")
        out = _graded_stage("clusters", lambda fix: clusterer.cluster(ps, Effort.NORMAL, settings, fix),
                            ps, settings, trace, result, world)
        world.write_clusters(out)
    assert [len(c.paper_ids) for c in out] == [2, 2, 2]
    assert len(labeller) == 5  # two complete-link groups, then three corrected labels
    assert all(c["fix"] == "One cluster is too broad; split the largest cluster." for c in labeller[2:])
    assert not result.warnings
    assert sorted(p for refs in world.read_clusters().values() for p in refs) == [p.id for p in ps]
    assert [e["threshold"] for e in trace.events if e["action"] == "group papers"] == [0.7, 0.75]


def test_redo_can_merge_excessive_singletons(monkeypatch, labeller):
    angle = np.arccos(0.68)
    monkeypatch.setattr(clusterer, "embed", lambda texts, settings: np.array([[1, 0], [np.cos(angle), np.sin(angle)]]))
    out = clusterer.cluster(papers(2), Effort.NORMAL, Settings(sample_mode=False, cluster_label_strategy="independent", cluster_threshold_mode="fixed", cluster_dense_weight=1), "Merge related singletons")
    assert [c.paper_ids for c in out] == [["P0", "P1"]]


def test_sample_mode_does_not_load_encoder_or_call_llm(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("sample mode attempted real compute")
    monkeypatch.setattr(clusterer, "embed", forbidden)
    monkeypatch.setattr(clusterer, "complete_json", forbidden)
    assert len(clusterer.cluster(papers(), Effort.NORMAL, Settings(sample_mode=True))) == 3


def test_empty_input_needs_no_model(monkeypatch):
    monkeypatch.setattr(clusterer, "embed", lambda *a: pytest.fail("empty input loaded a model"))
    assert clusterer.cluster([], Effort.NORMAL, Settings(sample_mode=False, cluster_label_strategy="independent", cluster_threshold_mode="fixed", cluster_dense_weight=1)) == []


def test_duplicate_paper_ids_are_rejected():
    p = papers(1)[0]
    with pytest.raises(ValueError, match="unique"):
        clusterer.cluster([p, p], Effort.NORMAL, Settings(sample_mode=False, cluster_label_strategy="independent", cluster_threshold_mode="fixed", cluster_dense_weight=1))


@pytest.mark.parametrize("vectors", [np.ones((1, 2)), np.array([[np.nan, 0], [1, 0]]), np.zeros((2, 2))])
def test_bad_embeddings_fail_before_labelling(monkeypatch, vectors, labeller):
    monkeypatch.setattr(clusterer, "embed", lambda *a: vectors)
    with pytest.raises(ValueError, match="finite non-zero"):
        clusterer.cluster(papers(2), Effort.NORMAL, Settings(sample_mode=False, cluster_label_strategy="independent", cluster_threshold_mode="fixed", cluster_dense_weight=1))
    assert not labeller


def test_labels_cannot_be_blank():
    with pytest.raises(ValidationError):
        ClusterLabel(label="  ", rationale="A reason")


def test_grouping_is_stable_when_the_retrieval_order_changes(monkeypatch, labeller):
    ps = papers(3)
    vector_by_id = {"P0": [1, 0], "P1": [np.cos(.6), np.sin(.6)], "P2": [np.cos(1.2), np.sin(1.2)]}
    def embedding(texts, settings):
        return np.array([vector_by_id[f"P{text.split()[1].rstrip('.')}"] for text in texts])
    monkeypatch.setattr(clusterer, "embed", embedding)
    settings = Settings(sample_mode=False, cluster_label_strategy="independent", cluster_dense_weight=1, cluster_threshold_mode="fixed")
    left = clusterer.cluster(ps, Effort.NORMAL, settings)
    right = clusterer.cluster(list(reversed(ps)), Effort.NORMAL, settings)
    assert [c.model_dump() for c in left] == [c.model_dump() for c in right]
