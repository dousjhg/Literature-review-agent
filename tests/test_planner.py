"""Planner (real, non-sample path) with the LLM mocked."""

import json
from functools import partial

import pytest
from pydantic import ValidationError

from litagent.agents import planner
from litagent.agents.planner import PlannerDraft
from litagent.config import EFFORT_PROFILES, Effort, Settings
from litagent.human_review import AutoApproveReviewer
from litagent.integrations.llm import complete_json
from litagent.orchestrator import run_pipeline
from litagent.schemas import HumanFeedback, SearchCriteria, SearchPlan
from litagent.trace import Trace
from tests.fakes import FakeClient

QUESTION = "How do graph neural networks predict drug-target interactions?"


def real_settings() -> Settings:
    s = Settings()
    s.sample_mode = False
    return s


def fake_llm(monkeypatch, draft: PlannerDraft) -> list[str]:
    """Make planner's complete_json return `draft`; returns the user messages it was sent."""
    sent: list[str] = []

    def fake(system_prompt, user_message, output_model, **kwargs):
        assert output_model is PlannerDraft
        sent.append(user_message)
        return draft

    monkeypatch.setattr(planner, "complete_json", fake)
    return sent


def ok_draft(**criteria) -> PlannerDraft:
    return PlannerDraft(
        status="ok",
        criteria=SearchCriteria(topic="GNNs for drug-target interaction", **criteria),
        queries=["graph neural network drug-target", "GNN binding affinity", "gnn BINDING affinity", "  ", "molecular graph learning"],
        rationale="Covers the method and the binding-affinity framing.",
    )


def test_clear_question_builds_plan_with_defaults(monkeypatch):
    fake_llm(monkeypatch, ok_draft())
    result = planner.plan(QUESTION, Effort.MEDIUM, real_settings())
    plan = result.plan
    assert result.status == "ok"
    # blanks and case-insensitive duplicates dropped, capped at Medium's 3 queries
    assert plan.queries == ["graph neural network drug-target", "GNN binding affinity", "molecular graph learning"]
    assert plan.criteria.year_from == real_settings().defaults.year_from
    assert plan.criteria.paper_types == ["primary", "review"]
    assert any(d.startswith("year_from=") for d in plan.used_defaults)
    assert "paper_types=primary+review" in plan.used_defaults
    assert (plan.max_papers, plan.token_budget) == (60, EFFORT_PROFILES[Effort.MEDIUM].token_budget)
    assert plan.rationale == "Covers the method and the binding-affinity framing."
    assert plan.question == QUESTION and plan.effort == Effort.MEDIUM


def test_number_of_queries_follows_effort(monkeypatch):
    sent = fake_llm(monkeypatch, ok_draft())
    plan = planner.plan(QUESTION, Effort.NORMAL, real_settings()).plan
    assert len(plan.queries) == 2
    assert json.loads(sent[0])["number_of_queries"] == 2


def test_user_criteria_are_kept(monkeypatch):
    fake_llm(monkeypatch, ok_draft(year_from=2015, paper_types=["review"], min_citations=50))
    plan = planner.plan(QUESTION, Effort.NORMAL, real_settings()).plan
    assert plan.criteria.year_from == 2015
    assert plan.criteria.paper_types == ["review"]
    assert plan.criteria.min_citations == 50
    assert plan.used_defaults == []


def test_no_default_start_year_after_users_end_year(monkeypatch):
    fake_llm(monkeypatch, ok_draft(year_to=2015))
    plan = planner.plan(QUESTION, Effort.NORMAL, real_settings()).plan
    assert plan.criteria.year_from is None and plan.criteria.year_to == 2015
    assert not any(d.startswith("year_from=") for d in plan.used_defaults)


def test_no_topic_returns_friendly_message(monkeypatch):
    fake_llm(monkeypatch, PlannerDraft(status="need_topic", message_to_user="What topic would you like to explore?"))
    result = planner.plan("Can you find me some good papers?", Effort.NORMAL, real_settings())
    assert result.status == "need_topic" and result.plan is None
    assert result.message_to_user == "What topic would you like to explore?"


def test_empty_question_makes_no_llm_call(monkeypatch):
    def must_not_call(*args, **kwargs):
        raise AssertionError("LLM called for an empty question")

    monkeypatch.setattr(planner, "complete_json", must_not_call)
    result = planner.plan("   ", Effort.NORMAL, real_settings())
    assert result.status == "need_topic" and result.message_to_user


def test_replan_sends_previous_plan_and_feedback(monkeypatch):
    sent = fake_llm(monkeypatch, ok_draft(paper_types=["review"]))
    previous = SearchPlan(
        question=QUESTION, criteria=SearchCriteria(topic="GNNs", year_from=2021, paper_types=["primary", "review"]),
        queries=["gnn drug target"], effort=Effort.NORMAL, max_papers=30, token_budget=50_000,
        used_defaults=["year_from=2021", "paper_types=primary+review"],
    )
    planner.plan(QUESTION, Effort.NORMAL, real_settings(), feedback="Only review papers", previous_plan=previous)
    message = json.loads(sent[0])
    assert message["feedback"] == "Only review papers"
    assert message["previous_plan"]["queries"] == ["gnn drug target"]
    assert message["previous_plan"]["filled_by_defaults"] == ["year_from=2021", "paper_types=primary+review"]
    assert message["previous_plan"]["criteria"]["year_from"] == 2021


def previous_plan_since_2020() -> SearchPlan:
    """The user said "since 2020"; paper types were a default."""
    return SearchPlan(
        question=QUESTION, criteria=SearchCriteria(topic="GNNs", year_from=2020, paper_types=["primary", "review"]),
        queries=["gnn drug target"], effort=Effort.NORMAL, max_papers=30, token_budget=50_000,
        used_defaults=["paper_types=primary+review"],
    )


def test_replan_restores_user_criteria_the_model_dropped(monkeypatch, tmp_path):
    fake_llm(monkeypatch, ok_draft(paper_types=["review"]))  # model forgot year_from=2020
    with Trace(str(tmp_path)) as trace:
        plan = planner.plan(QUESTION, Effort.NORMAL, real_settings(), "only reviews", previous_plan_since_2020()).plan
    assert plan.criteria.year_from == 2020
    assert plan.criteria.paper_types == ["review"]
    assert plan.used_defaults == []
    assert any(e["decision"] == "restored dropped criteria" for e in trace.events)


def test_replan_can_remove_a_criterion_on_purpose(monkeypatch):
    draft = ok_draft()
    draft.removed_criteria = ["year_from"]  # feedback: "any year is fine"
    fake_llm(monkeypatch, draft)
    plan = planner.plan(QUESTION, Effort.NORMAL, real_settings(), "any year is fine", previous_plan_since_2020()).plan
    assert plan.criteria.year_from == real_settings().defaults.year_from
    assert any(d.startswith("year_from=") for d in plan.used_defaults)


def test_replan_does_not_restore_defaults(monkeypatch):
    previous = previous_plan_since_2020()
    fake_llm(monkeypatch, ok_draft(year_from=2020))  # paper_types empty again: was a default, so not restored
    plan = planner.plan(QUESTION, Effort.NORMAL, real_settings(), "more queries", previous).plan
    assert plan.used_defaults == ["paper_types=primary+review"]


def test_first_plan_has_no_previous_plan_or_feedback(monkeypatch):
    sent = fake_llm(monkeypatch, ok_draft())
    planner.plan(QUESTION, Effort.NORMAL, real_settings())
    message = json.loads(sent[0])
    assert message["question"] == QUESTION
    assert "previous_plan" not in message and "feedback" not in message


@pytest.mark.parametrize("draft", [
    {"status": "ok", "criteria": {"topic": "GNNs"}, "queries": []},
    {"status": "ok", "criteria": {"topic": "GNNs"}, "queries": ["  "]},
    {"status": "ok", "criteria": None, "queries": ["gnn"]},
    {"status": "ok", "criteria": {"topic": " "}, "queries": ["gnn"]},
    {"status": "need_topic", "message_to_user": ""},
])
def test_incomplete_drafts_fail_validation(draft):
    with pytest.raises(ValidationError):
        PlannerDraft.model_validate(draft)


def test_incomplete_draft_is_retried_with_the_error(monkeypatch):
    bad = json.dumps({"status": "ok", "criteria": {"topic": "GNNs"}, "queries": []})
    good = json.dumps({"status": "ok", "criteria": {"topic": "GNNs"}, "queries": ["gnn drug target"], "rationale": "r"})
    client = FakeClient([bad, good])
    monkeypatch.setattr(planner, "complete_json", partial(complete_json, client=client))
    result = planner.plan(QUESTION, Effort.NORMAL, real_settings())
    assert result.plan.queries == ["gnn drug target"]
    assert len(client.calls) == 2
    assert "at least one search query" in client.calls[1]["messages"][-1]["content"]


class RejectFirstReviewer(AutoApproveReviewer):
    def __init__(self):
        self.calls = 0

    def review_papers(self, summary):
        self.calls += 1
        return HumanFeedback(approved=self.calls > 1, comment=None if self.calls > 1 else "only review papers")


def test_orchestrator_passes_previous_plan_on_replan(tmp_path, monkeypatch):
    calls = []
    real_plan = planner.plan

    def recording_plan(question, effort, settings, feedback=None, previous_plan=None):
        calls.append((feedback, previous_plan))
        return real_plan(question, effort, settings, feedback, previous_plan)

    monkeypatch.setattr(planner, "plan", recording_plan)
    settings = Settings()
    settings.sample_mode = True
    settings.logs_dir = str(tmp_path)
    run_pipeline(QUESTION, Effort.NORMAL, RejectFirstReviewer(), settings)
    assert calls[0] == (None, None)
    feedback, previous = calls[1]
    assert feedback == "only review papers"
    assert previous is not None and previous.question == QUESTION
