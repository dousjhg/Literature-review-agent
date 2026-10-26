"""The pipeline runs end to end in sample mode (data/sample/, no network). Keep this passing as you replace stubs."""

import numpy as np

from litagent.agents import connection_finder
from litagent.agents.clusterer import cosine_similarity_matrix, group_by_threshold
from litagent.agents.grader import check_cluster_sizes
from litagent.config import EFFORT_PROFILES, Effort, Settings
from litagent.human_review import AutoApproveReviewer
from litagent.integrations.llm import current_meter
from litagent.orchestrator import run_pipeline
from litagent.schemas import Cluster, HumanFeedback


def sample_settings(tmp_path) -> Settings:
    s = Settings()
    s.sample_mode = True
    s.logs_dir = str(tmp_path)
    return s


def test_full_run(tmp_path):
    result = run_pipeline("How do LLMs handle low-resource languages?", Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))
    assert result.status == "done"
    assert len(result.papers) == 30
    assert result.clusters and result.connections and result.gaps
    assert not result.warnings
    assert (tmp_path / f"trace-{result.run_id}.jsonl").exists()


def test_no_topic_asks_user(tmp_path):
    result = run_pipeline("find papers", Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))
    assert result.status == "need_topic"
    assert result.message_to_user


def test_defaults_applied(tmp_path):
    result = run_pipeline("graph neural networks for drug discovery", Effort.HIGH, AutoApproveReviewer(), sample_settings(tmp_path))
    assert result.plan.criteria.year_from is not None
    assert result.plan.criteria.paper_types == ["primary", "review"]
    assert result.plan.max_papers == 100


class RejectFirstReviewer(AutoApproveReviewer):
    def __init__(self):
        self.calls = 0

    def review_papers(self, summary):
        self.calls += 1
        if self.calls == 1:
            return HumanFeedback(approved=False, comment="only review papers")
        return HumanFeedback(approved=True)


def test_human_feedback_replans(tmp_path):
    reviewer = RejectFirstReviewer()
    result = run_pipeline("transformers for protein folding", Effort.NORMAL, reviewer, sample_settings(tmp_path))
    assert result.status == "done"
    assert reviewer.calls == 2


def test_group_by_threshold():
    emb = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
    groups = group_by_threshold(cosine_similarity_matrix(emb), 0.7)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]


def test_cluster_size_check():
    too_broad = [Cluster(id="C1", label="x", paper_ids=[str(i) for i in range(9)], rationale=""),
                 Cluster(id="C2", label="y", paper_ids=["9"], rationale="")]
    assert check_cluster_sizes(too_broad, 10) is not None


def test_time_limit_returns_partial(tmp_path):
    settings = sample_settings(tmp_path)
    settings.run_timeout_s = 0
    result = run_pipeline("graph neural networks for drug discovery", Effort.NORMAL, AutoApproveReviewer(), settings)
    assert result.status == "partial"
    assert any("time limit" in w for w in result.warnings)


def test_sample_mode_uses_matching_sample_file(tmp_path):
    result = run_pipeline("graph neural networks for drug discovery", Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))
    assert all(p.source == "openalex" for p in result.papers)
    assert any("drug" in p.title.lower() for p in result.papers)


def test_token_budget_stops_mid_stage_with_partial_result(tmp_path, monkeypatch):
    """Simulates the Connection finder's LLM calls using up the budget (no real LLM in sample mode)."""
    def spend_tokens(*args, **kwargs):
        meter = current_meter()
        meter.add(meter.usage.budget, 0)
        meter.check()  # what complete_json does before its next call

    monkeypatch.setattr(connection_finder, "find_connections", spend_tokens)
    result = run_pipeline("graph neural networks for drug discovery", Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))
    assert result.status == "partial"
    assert result.papers and result.clusters  # finished stages are kept
    assert result.connections == [] and result.gaps == []
    assert any("Token budget" in w and "connection_finder" in w for w in result.warnings)
    assert result.usage.total >= result.usage.budget == EFFORT_PROFILES[Effort.NORMAL].token_budget


def test_usage_reported_on_every_result(tmp_path):
    done = run_pipeline("graph neural networks for drug discovery", Effort.MEDIUM, AutoApproveReviewer(), sample_settings(tmp_path))
    need_topic = run_pipeline("find papers", Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))
    assert done.usage.budget == EFFORT_PROFILES[Effort.MEDIUM].token_budget and done.usage.llm_calls == 0  # sample mode makes no LLM calls
    assert need_topic.usage is not None


def test_provider_failure_after_review_returns_finished_stages(tmp_path, monkeypatch):
    from openai import APITimeoutError
    import httpx
    def timeout(*args, **kwargs):
        raise APITimeoutError(request=httpx.Request("POST", "http://nightshift"))
    monkeypatch.setattr(connection_finder, "find_connections", timeout)
    result = run_pipeline("graph neural networks for drug discovery", Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))
    assert result.status == "partial" and result.papers and result.clusters and result.connections == []
    assert any("Model provider failed during connection_finder (APITimeoutError)" in w for w in result.warnings)
    assert result.paper_map["nodes"]


def test_provider_failure_before_papers_is_an_error(tmp_path, monkeypatch):
    import pytest
    from litagent.agents import planner
    from litagent.integrations.llm import LLMError
    monkeypatch.setattr(planner, "plan", lambda *a, **k: (_ for _ in ()).throw(LLMError("planner output invalid twice")))
    with pytest.raises(LLMError):
        run_pipeline("graph neural networks for drug discovery", Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))


def test_sample_placeholders_quote_abstracts_verbatim(tmp_path):
    for question in ("graph neural networks for drug discovery", "protein structure prediction with transformers"):
        result = run_pipeline(question, Effort.NORMAL, AutoApproveReviewer(), sample_settings(tmp_path))
        by_id = {p.id: p for p in result.papers}
        quotes = [e for f in [*result.connections, *result.gaps] for e in f.evidence]
        assert quotes and all(e.quote and e.quote in (by_id[e.paper_id].abstract or "") for e in quotes)
