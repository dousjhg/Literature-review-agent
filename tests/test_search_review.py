"""POST /search: web Human review 1 stops for the user, re-plans with feedback, feeds POST /cluster."""

import json

import pytest
from fastapi.testclient import TestClient

from litagent import api, search_review
from litagent.agents import planner
from litagent.config import Effort, Settings
from litagent.schemas import PlannerResult


@pytest.fixture
def settings(tmp_path):
    cfg = Settings(sample_mode=True)
    cfg.logs_dir = str(tmp_path)
    return cfg


@pytest.fixture
def client(settings):
    api.app.dependency_overrides[api.get_settings] = lambda: settings
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_search_returns_plan_papers_and_summary_without_approving(client, tmp_path):
    body = client.post("/search", json={"run_id": "s-1", "input": {
        "query": "How do large language models handle low-resource languages?", "effort": "normal"}}).json()
    assert body["status"] == "success" and body["run_id"] == "s-1"
    out = body["output"]
    assert out["status"] == "ok" and out["sample_mode"] and out["plan"]["max_papers"] == 30
    assert len(out["papers"]) == 30 and out["summary"] and out["follow_up_question"]
    events = [json.loads(line) for line in (tmp_path / "trace-s-1.jsonl").read_text().splitlines()]
    assert {"planner", "retriever", "zetaris", "human"} <= {e["agent_name"] for e in events}
    assert any(e["agent_name"] == "human" and e["decision"] == "waiting for the user" for e in events)
    assert not list(tmp_path.glob("cluster-map-*.json"))  # nothing is clustered until the user decides


def test_kept_papers_feed_the_cluster_map(client):
    out = client.post("/search", json={"input": {"query": "graph neural networks for drug discovery"}}).json()["output"]
    kept = out["papers"][:6]
    made = client.post("/cluster", json={"input": {"query": out["plan"]["question"], "effort": "normal",
                                                    "papers": kept}}).json()
    assert made["status"] == "success"
    assert sorted(p for c in made["output"]["clusters"] for p in c["paper_ids"]) == sorted(p["id"] for p in kept)


def test_no_topic_asks_the_user(client):
    out = client.post("/search", json={"input": {"query": "find papers"}}).json()["output"]
    assert out["status"] == "need_topic" and out["message_to_user"] and not out["papers"]


def test_replan_sends_feedback_and_previous_plan_to_the_planner(settings, monkeypatch):
    first = search_review.run_search("protein structure prediction with transformers", Effort.NORMAL, settings)
    seen = {}
    real = planner.plan
    def spy(question, effort, cfg, feedback=None, previous_plan=None) -> PlannerResult:
        seen.update(feedback=feedback, previous_plan=previous_plan)
        return real(question, effort, cfg, feedback, previous_plan)
    monkeypatch.setattr(planner, "plan", spy)
    again = search_review.run_search(first.plan.question, Effort.NORMAL, settings,
                                     feedback="Only papers since 2023", previous_plan=first.plan)
    assert seen == {"feedback": "Only papers since 2023", "previous_plan": first.plan}
    assert again.status == "ok" and again.search_id != first.search_id


def test_replan_without_feedback_is_rejected(client):
    plan = client.post("/search", json={"input": {"query": "graph neural networks for drug discovery"}}).json()["output"]["plan"]
    r = client.post("/search", json={"input": {"query": plan["question"], "previous_plan": plan}})
    assert r.status_code == 422 and r.json()["error"]["type"] == "InvalidInput"


def test_zetaris_misconfiguration_falls_back_with_a_warning(settings, monkeypatch):
    settings.sample_mode = False
    monkeypatch.delenv("ZETARIS_API_KEY", raising=False)
    result = search_review.SearchResult(search_id="x", status="ok")
    papers = search_review.retriever._fake_papers(2)
    assert search_review._through_zetaris(papers, settings, "x", result) == papers
    assert "Zetaris unavailable" in result.warnings[0]
