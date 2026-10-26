"""AI review stays independent, source-bound, cached, and separate from human judgments."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from litagent import api
from litagent.agents import review_agent
from litagent.cluster_state import VersionError, path_for
from litagent.finding_analysis import analyze_cluster_map
from litagent.finding_review import ReviewSubmission, prepare_review, save_review
from litagent.integrations.llm import LLMError
from litagent.schemas import ReviewAgentDecisions
from tests.test_cluster_regeneration import snapshot
from tests.test_findings import model


@pytest.fixture
def reviewed_map(model, tmp_path, monkeypatch):
    root, cfg, papers, arrays = snapshot(tmp_path)
    result = analyze_cluster_map(root.run_id, root.version_token, cfg)
    calls = []
    behavior = {"bad_ref": False, "wrong_ids": False, "weak_support": False}

    def complete(prompt, message, schema, **kwargs):
        payload = json.loads(message)
        calls.append(payload)
        assert schema is ReviewAgentDecisions
        assert "PRIVATE HUMAN OPINION" not in message
        assert all(set(item) == {"id", "kind", "paper_ids", "claim", "proposed_test"} for item in payload["items"])
        sources = {p["id"]: p for p in payload["papers"]}
        return schema.model_validate({"decisions": [{
            "item_id": "wrong" if behavior["wrong_ids"] else item["id"],
            "verdict": "supported",
            "scores": {"source_support": 1 if behavior["weak_support"] else 3, "relevance": 2, "testability": 2},
            "reason": "Both original experiment descriptions support the stated bounded comparison.",
            "limitations": ["No external novelty search performed."],
            "evidence": [{"paper_id": pid, "span_id": "invented" if behavior["bad_ref"] else sources[pid]["spans"][0]["span_id"]}
                         for pid in item["paper_ids"]],
        } for item in payload["items"]]})

    monkeypatch.setattr(review_agent, "complete_json", complete)
    return result, cfg, calls, behavior


def test_ai_reviews_claims_without_human_leakage_and_reuses_exact_cache(reviewed_map):
    result, cfg, calls, _ = reviewed_map
    pack = prepare_review(result.run_id, cfg)
    item = pack["items"][0]
    submission = ReviewSubmission(expected_version_token=result.version_token, verdict="uncertain",
        reviewer="X", rationale="PRIVATE HUMAN OPINION: please preserve this original judgment.")
    save_review(result.run_id, item["id"], submission, cfg)
    human = path_for(cfg, "finding-review", result.run_id)
    original = path_for(cfg, "cluster-map", result.run_id)
    human_before, map_before = human.read_bytes(), original.read_bytes()
    assert review_agent.read_review(result.run_id, cfg)["report"] is None and not calls
    first = review_agent.run_review(result.run_id, result.version_token, cfg)
    assert first["reused"] is False and first["report"]["provenance"] == "model"
    assert len(calls) == 1
    papers = {p["id"]: p["source_text"] for i in pack["items"] for p in i["papers"]}
    for assessment in first["report"]["items"]:
        assert assessment["total"] == 7
        assert len(assessment["evidence"]) == 2
        assert all(e["quote"] in papers[e["paper_id"]] for e in assessment["evidence"])
    second = review_agent.run_review(result.run_id, result.version_token, cfg)
    assert second["reused"] and second["report"] == first["report"] and len(calls) == 1
    assert human.read_bytes() == human_before and original.read_bytes() == map_before
    # Human edits cannot invalidate AI inputs, because they never enter the grader prompt.
    save_review(result.run_id, item["id"], submission.model_copy(update={"rationale": "PRIVATE HUMAN OPINION: a revised explanation."}), cfg)
    assert review_agent.run_review(result.run_id, result.version_token, cfg)["reused"]
    assert len(calls) == 1
    assert review_agent.read_review(result.run_id, replace(cfg, review_agent_source_chars=5000))["report"] is None


@pytest.mark.parametrize("invalid", ["bad_ref", "wrong_ids", "weak_support"])
def test_invalid_ai_quotes_ids_and_score_verdicts_fail_bounded_validation(reviewed_map, invalid):
    result, cfg, calls, behavior = reviewed_map
    behavior[invalid] = True
    with pytest.raises(LLMError, match="validation failed"):
        review_agent.run_review(result.run_id, result.version_token, cfg)
    assert len(calls) == cfg.finding_validation_retries + 1
    assert review_agent.read_review(result.run_id, cfg)["report"] is None
    assert not path_for(cfg, "finding-review", result.run_id).exists()


def test_missing_hypothesis_is_unscored_without_any_llm_call(model, tmp_path, monkeypatch):
    root, cfg, *_ = snapshot(tmp_path)
    monkeypatch.setattr(review_agent, "complete_json", lambda *a, **k: pytest.fail("Missing claims cannot be graded"))
    out = review_agent.run_review(root.run_id, root.version_token, cfg)["report"]
    assert out["items"]
    assert all(i["status"] == "needs_hypothesis" and i["scores"] is None and i["total"] is None for i in out["items"])
    assert out["usage"]["llm_calls"] == 0


def test_api_modes_tokens_cache_integrity_and_source_limits(reviewed_map):
    result, cfg, calls, _ = reviewed_map
    api.app.dependency_overrides[api.get_settings] = lambda: cfg
    try:
        with TestClient(api.app) as client:
            url = f"/cluster/{result.run_id}/review-agent"
            assert client.get(url).json()["output"]["report"] is None
            assert client.post(url, json={"expected_version_token": "0"*32}).status_code == 409
            assert not calls
            out = client.post(url, json={"expected_version_token": result.version_token})
            assert out.status_code == 200, out.json()
            assert client.get(url).json()["output"]["report"]["report_id"] == out.json()["output"]["report"]["report_id"]
            cache = path_for(cfg, "review-agent", out.json()["output"]["report"]["input_fingerprint"])
            envelope = json.loads(cache.read_text())
            envelope["report"]["items"][0]["total"] = 9
            cache.write_text(json.dumps(envelope))
            assert client.get(url).status_code == 409
            with pytest.raises(VersionError, match="modes differ"):
                review_agent.run_review(result.run_id, result.version_token, replace(cfg, sample_mode=True))
            with pytest.raises(VersionError, match="item limit"):
                review_agent.run_review(result.run_id, result.version_token, replace(cfg, review_agent_max_items=1))
    finally:
        api.app.dependency_overrides.clear()
