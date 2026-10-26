"""AI review stays independent, source-bound, cached, and separate from human judgments."""

import json
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from litagent import api
from litagent.agents import review_agent
from litagent.cluster_state import VersionError, path_for
from litagent.config import Settings
from litagent.finding_analysis import analyze_cluster_map
from litagent.finding_review import ReviewSubmission, prepare_review, save_review
from litagent.integrations.llm import LLMError
from litagent.schemas import ReviewAgentDecisions, ReviewExplanationChecks
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
        if schema is ReviewExplanationChecks:
            assert "PRIVATE HUMAN OPINION" not in message
            assert all(set(d) == {"item_id", "reason", "limitations", "statements"} for d in payload["draft_explanations"])
            sources = {p["id"]: p for p in payload["papers"]}
            drafts = {d["item_id"]: d for d in payload["draft_explanations"]}
            return schema(checks=[{"item_id": i["id"], "issues": [], "attributions": [
                {"paper_id": pid, "assertion": "The provided experiment supports the bounded comparison.",
                 "statement_id": statement["statement_id"],
                 "evidence": [{"paper_id": pid, "span_id": sources[pid]["spans"][0]["span_id"]}],
                 "assessment": "grounded", "explanation": "Supported by this paper's original experiment."}
                for statement in drafts[i["id"]]["statements"] for pid in i["paper_ids"]]} for i in payload["items"]])
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


@pytest.mark.parametrize("repair", [True, False])
def test_valid_quotes_do_not_bypass_cross_paper_attribution_audit(monkeypatch, repair):
    """Regression: the auditor rejects prose even when every quote is real.

    Model judgments are stubbed; this checks repair/isolation/fail-closed behavior,
    not the semantic accuracy of a live model.
    """
    cfg = Settings(sample_mode=False)
    sources = {
        "nusa": {"id": "nusa", "title": "NusaMT-7B", "source_field": "abstract", "spans": [
            {"span_id": "n1", "quote": "NusaMT-7B uses LLaMA2-7B and evaluates on FLORES-200 with spBLEU."}]},
        "lsftl": {"id": "lsftl", "title": "LSFTL", "source_field": "abstract", "spans": [
            {"span_id": "l1", "quote": "LSFTL improved COMET scores by 1-3 points on low-resource Asian languages."}]},
    }
    payload = {"question": "Compare low-resource translation methods", "papers": list(sources.values()),
        "items": [{"id": "gap-x", "kind": "gap", "paper_ids": ["lsftl", "nusa"],
                   "claim": "Compare the two methods under controlled conditions.",
                   "proposed_test": "Use COMET and spBLEU on FLORES-200 in a future controlled comparison."}]}
    bad = "NusaMT-7B reports COMET results; LSFTL reports spBLEU on FLORES-200."
    good = "NusaMT-7B 摘要报告 spBLEU；LSFTL 摘要报告 COMET。共同使用两项指标属于拟议实验。"
    calls = []

    def complete(prompt, message, schema, **kwargs):
        data = json.loads(message)
        calls.append(schema)
        if schema is ReviewExplanationChecks:
            assert "validation_error" not in data and "rejected_output" not in data
            assert data["papers"] == payload["papers"]
            draft = data["draft_explanations"][0]
            assert set(draft) == {"item_id", "reason", "limitations", "statements"}
            assert "verdict" not in message and "scores" not in draft
            return schema(checks=[{"item_id": "gap-x", "issues": [], "attributions": [
                {"paper_id": pid, "assertion": draft["reason"],
                 "statement_id": statement["statement_id"],
                 "evidence": [{"paper_id": pid, "span_id": span}],
                 "assessment": "grounded" if draft["reason"] == good else "unsubstantiated",
                 "explanation": "各指标必须对应各自论文摘要；不能混淆未来实验与历史结果。"}
                for statement in draft["statements"] for pid, span in [("nusa", "n1"), ("lsftl", "l1")]]}])
        if "validation_error" in data:
            assert "Explanation source attribution failed" in data["validation_error"]
        return schema(decisions=[{"item_id": "gap-x", "verdict": "uncertain",
            "scores": {"source_support": 1, "relevance": 3, "testability": 2},
            "reason": good if repair and "validation_error" in data else bad,
            "limitations": ["仅据所提供摘要评审。"],
            "evidence": [{"paper_id": "nusa", "span_id": "n1"}, {"paper_id": "lsftl", "span_id": "l1"}]}])

    monkeypatch.setattr(review_agent, "complete_json", complete)
    if repair:
        out = review_agent._grade(payload, sources, cfg, time.monotonic() + 60)
        assert out[0].reason == good
        assert out[0].evidence[0].quote == sources["nusa"]["spans"][0]["quote"]
        assert len(calls) == 4
    else:
        with pytest.raises(LLMError, match="source attribution failed"):
            review_agent._grade(payload, sources, cfg, time.monotonic() + 60)
        assert len(calls) == 2 * (cfg.finding_validation_retries + 1)


@pytest.mark.parametrize("failure", ["wrong_ids", "deadline", "provider", "missing_paper", "foreign_evidence", "missing_statement"])
def test_audit_failure_never_saves_an_unverified_report(reviewed_map, monkeypatch, failure):
    result, cfg, _, _ = reviewed_map
    original = review_agent.complete_json
    human = path_for(cfg, "finding-review", result.run_id)
    map_path = path_for(cfg, "cluster-map", result.run_id)
    before = map_path.read_bytes()

    def complete(prompt, message, schema, **kwargs):
        if schema is not ReviewExplanationChecks:
            return original(prompt, message, schema, **kwargs)
        if failure == "provider":
            raise LLMError("audit provider unavailable")
        if failure == "deadline":
            monkeypatch.setattr(review_agent.time, "monotonic", lambda: float("inf"))
        audit = original(prompt, message, schema, **kwargs)
        if failure == "missing_paper":
            pid = audit.checks[0].attributions[0].paper_id
            audit.checks[0].attributions = [a for a in audit.checks[0].attributions if a.paper_id != pid]
        elif failure == "missing_statement":
            sid = audit.checks[0].attributions[0].statement_id
            audit.checks[0].attributions = [a for a in audit.checks[0].attributions if a.statement_id != sid]
        elif failure == "foreign_evidence":
            audit.checks[0].attributions[0].evidence[0].paper_id = "foreign"
        else:
            audit.checks[0].item_id = "wrong"
        return audit

    monkeypatch.setattr(review_agent, "complete_json", complete)
    with pytest.raises(LLMError):
        review_agent.run_review(result.run_id, result.version_token, cfg)
    assert review_agent.read_review(result.run_id, cfg)["report"] is None
    assert not human.exists() and map_path.read_bytes() == before


def test_literal_metric_guard_rejects_false_omission_even_when_auditor_approves(monkeypatch):
    cfg = Settings(sample_mode=False)
    sources = [
        {"id": "nusa", "spans": [{"span_id": "n1", "quote": "NusaMT evaluates on FLORES-200 using spBLEU."}]},
        {"id": "lsftl", "spans": [{"span_id": "l1", "quote": "LSFTL improved COMET scores by 1-3 points on low-resource languages."}]},
    ]
    payload = {"papers": sources, "items": [{"id": "gap-x", "paper_ids": ["nusa", "lsftl"]}]}
    output = ReviewAgentDecisions(decisions=[{"item_id": "gap-x", "verdict": "uncertain",
        "scores": {"source_support": 1, "relevance": 3, "testability": 2},
        "reason": "NusaMT 报告 spBLEU。", "limitations": ["LSFTL 摘要未说明基座模型、训练数据、评测指标"]}])

    def approve(prompt, message, schema, **kwargs):
        return schema(checks=[{"item_id": "gap-x", "issues": [], "attributions": [
            {"paper_id": pid, "statement_id": sid, "assertion": assertion, "assessment": "grounded",
             "evidence": [{"paper_id": pid, "span_id": span}], "explanation": "Incorrectly approved by auditor."}
            for pid, sid, span, assertion in [("nusa", "s0", "n1", "NusaMT 报告 spBLEU"),
                                             ("lsftl", "s1", "l1", "LSFTL 未说明评测指标")]]}])

    monkeypatch.setattr(review_agent, "complete_json", approve)
    with pytest.raises(ValueError, match="explicitly mentions metrics.*COMET"):
        review_agent._audit_explanations(payload, output, cfg, time.monotonic() + 60)
    assert review_agent._metric_mentions(sources[0], cfg) == [{"term": "spBLEU", "span_id": "n1"}]


@pytest.mark.parametrize("text,expected", [
    ("摘要未说明基座模型、训练数据、评测指标", True),
    ("摘要未报告评测指标。", True),
    ("摘要未说明完整的评测指标。", False),
    ("摘要未说明 COMET 指标版本。", False),
    ("摘要未说明基座模型，但明确报告 COMET 指标。", False),
])
def test_metric_absence_guard_preserves_qualified_missing_details(text, expected):
    assert review_agent._unqualified_metric_omission(text) is expected
