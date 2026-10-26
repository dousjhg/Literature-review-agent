"""POST /run in sample mode: JSON shape, errors, logs and secrets."""

import json

import pytest
from fastapi.testclient import TestClient

from litagent import api
from litagent.agents import clusterer
from litagent.config import EFFORT_PROFILES, Effort, Settings

SECRET = "sk-test-secret-token-123456"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("METERLESS_RUNTIME_TOKEN", SECRET)
    settings = Settings()
    settings.sample_mode = True
    settings.logs_dir = str(tmp_path)
    api.app.dependency_overrides[api.get_settings] = lambda: settings
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def read_trace(tmp_path, run_id):
    path = tmp_path / f"trace-{run_id}.jsonl"
    assert path.exists()
    return path.read_text(encoding="utf-8")


def test_run_success_shape(client, tmp_path):
    r = client.post("/run", json={"run_id": "t-1", "input": {"query": "How do LLMs handle low-resource languages?", "effort": "normal"}})
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "t-1"
    assert body["status"] == "success"
    out = body["output"]
    assert out["run_id"] == "t-1" and out["status"] == "done"
    assert len(out["papers"]) == 30
    assert out["clusters"] and out["connections"] and out["gaps"]
    assert out["paper_map"]["nodes"] and out["paper_map"]["edges"]
    assert out["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0, "estimated_calls": 0, "budget": EFFORT_PROFILES[Effort.NORMAL].token_budget}

    events = [json.loads(line) for line in read_trace(tmp_path, "t-1").splitlines()]
    required = {"timestamp", "run_id", "agent_name", "action", "decision", "result", "next", "why"}
    assert all(required <= e.keys() and e["run_id"] == "t-1" for e in events)
    agents = {e["agent_name"] for e in events}
    assert {"planner", "retriever", "zetaris", "meterless_world_model", "grader"} <= agents


def test_run_id_generated_when_missing(client, tmp_path):
    r = client.post("/run", json={"input": {"query": "graph neural networks for drug discovery"}})
    body = r.json()
    assert body["status"] == "success" and body["run_id"]
    read_trace(tmp_path, body["run_id"])


def test_no_topic_is_success_with_message(client):
    body = client.post("/run", json={"input": {"query": "find papers"}}).json()
    assert body["status"] == "success"
    assert body["output"]["status"] == "need_topic"
    assert body["output"]["message_to_user"]


def test_feedback_triggers_one_replan(client, tmp_path):
    body = client.post("/run", json={"run_id": "t-fb", "input": {"query": "transformers for protein folding", "feedback": "only review papers"}}).json()
    assert body["output"]["status"] == "done"
    events = [json.loads(line) for line in read_trace(tmp_path, "t-fb").splitlines()]
    reviews = [e["decision"] for e in events if e["action"] == "review 1"]
    assert reviews == ["not satisfied", "approved"]


def test_failing_stage_returns_structured_error(client, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(f"embedding service rejected token {SECRET}")

    monkeypatch.setattr(clusterer, "cluster", boom)
    r = client.post("/run", json={"run_id": "t-err", "input": {"query": "graph neural networks for drug discovery"}})
    assert r.status_code == 500
    body = r.json()
    assert body["run_id"] == "t-err"
    assert body["status"] == "error"
    assert body["error"]["type"] == "RuntimeError"
    assert "embedding service rejected" in body["error"]["message"]

    trace = read_trace(tmp_path, "t-err")
    last = json.loads(trace.splitlines()[-1])
    assert last["agent_name"] == "clusterer" and last["decision"] == "error"
    assert last["next"] == "return error to caller"
    # Secrets never reach the response or the log
    assert SECRET not in r.text
    assert SECRET not in trace


@pytest.mark.parametrize("payload", [
    {"input": {"query": "x y z", "effort": "extreme"}},
    {"input": {}},
    {"run_id": "../../etc/passwd", "input": {"query": "graph neural networks"}},
])
def test_invalid_input_returns_structured_error(client, payload):
    r = client.post("/run", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "error"
    assert body["error"]["type"] == "InvalidInput"


def test_world_endpoint_returns_the_runs_world_model(client):
    client.post("/run", json={"run_id": "t-world", "input": {"query": "graph neural networks for drug discovery"}})
    body = client.get("/world/t-world").json()
    assert body["status"] == "success"
    out = body["output"]
    types = {e["type"] for e in out["entities"]}
    assert {"paper", "author", "cluster", "gap", "search", "stage"} <= types
    assert {r["type"] for r in out["relationships"]} >= {"found-by", "authored-by", "member-of", "connects-to", "between"}
    assert out["facts"] and out["audit"] and out["events"] > 0


@pytest.mark.parametrize("run_id, status", [("never-ran", 404), ("bad id!", 422)])
def test_world_endpoint_errors_are_structured(client, run_id, status):
    r = client.get(f"/world/{run_id}")
    assert r.status_code == status
    assert r.json()["status"] == "error"
