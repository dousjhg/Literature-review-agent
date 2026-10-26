"""Orchestrator: runs the agents in the order shown in ARCHITECTURE.md.

Loops:
- No topic -> return a friendly prompt to the user.
- Human review 1 or 2 not satisfied -> back to the Planner with the user's feedback.
- Grading agent fails a stage -> redo that stage once with the grader's fix,
  then continue with a warning (TODO: decide whether to ask the user instead).

Bounds: replans and redos are capped in config.py. The run stops between stages once
settings.run_timeout_s has passed, and before any LLM call once the effort level's token budget
is used up; both return status="partial" with what was finished. So does a model-provider failure
(timeout, output still invalid after the repair) once the run is past Human review 1: the finished
stages are worth more than an error. Any other exception, or one before papers exist, is logged to
the trace and re-raised for the caller (CLI or API).
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from openai import OpenAIError

from litagent.agents import clusterer, connection_finder, gap_finder, grader, planner, retriever
from litagent.config import EFFORT_PROFILES, Effort, Settings
from litagent.cluster_state import ClusteringArtifacts
from litagent.finding_engine import FindingContext
from litagent.human_review import Reviewer
from litagent.integrations.llm import LLMError, TokenBudgetExceeded, TokenMeter
from litagent.integrations.meterless import ResearchWorld
from litagent.integrations.zetaris import ZetarisStore
from litagent.schemas import RunResult
from litagent.trace import Trace

T = TypeVar("T")


class _OutOfTime(Exception):
    def __init__(self, before: str):
        self.before = before


def run_pipeline(
    question: str,
    effort: Effort,
    reviewer: Reviewer,
    settings: Settings | None = None,
    signed_in: bool = False,
    run_id: str | None = None,
) -> RunResult:
    settings = settings or Settings()
    budget = EFFORT_PROFILES[effort].token_budget
    with Trace(settings.logs_dir, run_id) as trace, TokenMeter(budget) as meter:
        trace.log("orchestrator", "start", "run started", question, "planner",
                  why=f"effort={effort.value}, token budget={budget:,}, sample_mode={settings.sample_mode}")

        def finish(r: RunResult) -> RunResult:
            r.usage = meter.usage
            trace.log("orchestrator", "finish", r.status, meter.usage, "return result",
                      why=f"{meter.usage.total:,} of {budget:,} tokens in {meter.usage.llm_calls} LLM calls")
            return r

        deadline = time.monotonic() + settings.run_timeout_s
        stage = "orchestrator"
        result: RunResult | None = None
        world = ResearchWorld(run_id=trace.run_id, directory=settings.logs_dir)

        def check_time(next_stage: str) -> None:
            if time.monotonic() > deadline:
                raise _OutOfTime(next_stage)

        try:
            stage = "zetaris"
            store = ZetarisStore(sample_mode=settings.sample_mode, timeout_s=settings.zetaris_timeout_s, run_id=trace.run_id)
            feedback: str | None = None
            previous_plan = None

            for attempt in range(1, settings.max_replans + 1):
                # ----- 1. Planner -----
                stage = "planner"
                check_time(stage)
                planned = planner.plan(question, effort, settings, feedback, previous_plan)
                if planned.status == "need_topic" or planned.plan is None:
                    trace.log("planner", "check topic", "no clear topic", next_step="ask user", why="Avoid guessing a topic")
                    return finish(RunResult(run_id=trace.run_id, status="need_topic", message_to_user=planned.message_to_user))
                plan = previous_plan = planned.plan
                world.start_attempt(attempt, question, plan.queries)
                trace.log("planner", f"plan (attempt {attempt})", "plan created", plan, "retriever",
                          why=f"{plan.rationale or 'no rationale given'}; defaults used: {plan.used_defaults}")

                # ----- 2. Retriever -> Zetaris -----
                stage = "retriever"
                check_time(stage)
                papers = retriever.retrieve(plan, settings, signed_in=signed_in)
                if not papers:
                    message = retriever.no_results_message(plan)
                    trace.log("retriever", "retrieve", "0 papers", next_step="ask user to broaden", why=message)
                    return finish(RunResult(run_id=trace.run_id, status="no_results", plan=plan, message_to_user=message))
                store.save_papers(papers)
                world.add_papers(papers)
                trace.log("retriever", "retrieve", f"{len(papers)} papers", next_step="human review 1",
                          why="sample papers from data/sample/" if settings.sample_mode else f"queries: {plan.queries}")

                # ----- Human review 1 -----
                stage = "human"
                fb = reviewer.review_papers(retriever.summarise(papers, plan, settings))
                if not fb.approved:
                    feedback = fb.comment
                    trace.log("human", "review 1", "not satisfied", fb.comment, "planner", why="User feedback")
                    continue
                trace.log("human", "review 1", "approved", next_step="clusterer")
                papers = store.get_papers()

                result = RunResult(run_id=trace.run_id, status="done", plan=plan, papers=papers)
                if store.fallback_reason:
                    result.warnings.append(f"Zetaris unavailable, used the in-memory papers instead ({store.fallback_reason})")

                # ----- 3. Clusterer + grading -----
                stage = "clusterer"
                check_time(stage)
                artifacts = ClusteringArtifacts()
                result.clusters = _graded_stage(
                    "clusters", lambda fix: clusterer.cluster(papers, effort, settings, fix,
                                                              defer_labels_on_size_failure=True, question=plan.question, artifacts=artifacts),
                    papers, settings, trace, result, world
                )
                world.write_clusters(result.clusters)
                finding_context = FindingContext(question=plan.question, similarity=artifacts.similarity,
                    use_full_text=EFFORT_PROFILES[effort].use_full_text, deadline=deadline)

                # ----- 4. Connection finder + grading -----
                stage = "connection_finder"
                check_time(stage)
                result.connections = _graded_stage(
                    "connections",
                    lambda fix: connection_finder.find_connections(papers, result.clusters, settings, fix, context=finding_context),
                    papers, settings, trace, result, world, finding_context=finding_context,
                )
                world.write_connections(result.connections)

                # ----- 5. Gap finder + grading -----
                stage = "gap_finder"
                check_time(stage)
                result.gaps = _graded_stage(
                    "gaps",
                    lambda fix: gap_finder.find_gaps(papers, result.clusters, result.connections, settings, fix, context=finding_context),
                    papers, settings, trace, result, world, finding_context=finding_context,
                )
                world.write_gaps(result.gaps)
                result.paper_map = world.to_graph_json()

                # ----- Human review 2 -----
                stage = "human"
                fb = reviewer.review_result(result)
                if fb.approved:
                    trace.log("human", "review 2", "approved", next_step="done")
                    return finish(result)
                feedback = fb.comment
                trace.log("human", "review 2", "follow-up", fb.comment, "planner", why="User feedback")

            trace.log("orchestrator", "stop", "max replans reached", why=f"limit {settings.max_replans}")
            return finish(RunResult(run_id=trace.run_id, status="max_replans_reached", message_to_user="Let's narrow the question down together."))

        except (_OutOfTime, TokenBudgetExceeded) as e:
            if isinstance(e, _OutOfTime):
                action, reason = "time limit", f"Run time limit ({settings.run_timeout_s:.0f} s) reached before {e.before}"
            else:
                action, reason = "token budget", f"Token budget ({budget:,} tokens) used up during {stage}"
            partial = result or RunResult(run_id=trace.run_id, status="partial")
            partial.status = "partial"
            partial.paper_map = world.to_graph_json()
            partial.warnings.append(f"{reason}; returning what was finished.")
            trace.log("orchestrator", action, "stop early", next_step="return partial result", why=reason)
            return finish(partial)
        except (LLMError, OpenAIError) as e:
            if result is None:  # nothing finished worth returning yet
                trace.error(stage, "run stage", e, next_step="return error to caller")
                raise
            reason = f"Model provider failed during {stage} ({type(e).__name__})"
            trace.error(stage, "run stage", e, next_step="return partial result")
            result.status = "partial"
            result.paper_map = world.to_graph_json()
            result.warnings.append(f"{reason}; returning what was finished. Run again for the remaining stages.")
            return finish(result)
        except Exception as e:
            trace.error(stage, "run stage", e, next_step="return error to caller")
            raise


def _graded_stage(
    stage: grader.Stage,
    produce: Callable[[str | None], list[T]],
    papers,
    settings: Settings,
    trace: Trace,
    result: RunResult,
    world: ResearchWorld,
    finding_context: FindingContext | None = None,
) -> list[T]:
    fix: str | None = None
    for attempt in range(settings.max_redo_per_stage + 1):
        output = produce(fix)
        setattr(result, stage, output)  # Keep completed output if grading exhausts the token budget.
        try:
            scope = grader.finding_scope(finding_context, stage) if finding_context is not None else {}
            g = grader.grade(stage, output, papers, settings, **scope)
        except (TokenBudgetExceeded, LLMError, OpenAIError):
            # No output for this stage has been written yet. Persist the last completed
            # attempt so the partial graph matches the structured partial result.
            {"clusters": world.write_clusters, "connections": world.write_connections,
             "gaps": world.write_gaps}[stage](output)
            raise
        ok = grader.passed(g, settings)
        world.record_grade(stage, g)
        trace.log(
            "grader", f"grade {stage} (attempt {attempt + 1})", f"{g.total}/9 {'pass' if ok else 'fail'}",
            g, "next stage" if ok else f"redo {stage}", why=g.fix or g.reasoning,
        )
        if ok:
            return output
        fix = g.fix
    result.warnings.append(f"{stage} did not pass grading after {settings.max_redo_per_stage + 1} attempts")
    trace.log("orchestrator", f"fallback {stage}", "continue with last attempt", next_step="next stage",
              why=f"{stage} still failed grading after {settings.max_redo_per_stage + 1} attempts")
    return output
