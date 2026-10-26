"""Run the agent on every topic in evals/topics.yaml and save measured results.

    python -m evals.run_eval                    # real run (Nightshift, OpenAlex, Zetaris)
    python -m evals.run_eval --sample-mode      # check the harness works offline
    python -m evals.run_eval --only lrl-llm     # one topic

Workflow metrics are computed in code for every run: quote provenance (each evidence quote is a
verbatim substring of the cited paper's original abstract/full text), first-attempt grading pass
rate, final grades, counts, tokens and wall time. Survey metrics (cluster agreement, gap recall)
need an answer key from a published survey in topics.yaml; an LLM judge matches the key to the
output and code validates its IDs. Without a key they stay null: no answer key is generated here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from litagent.config import Effort, Settings
from litagent.human_review import AutoApproveReviewer
from litagent.integrations.llm import TokenMeter, complete_json, load_prompt
from litagent.orchestrator import run_pipeline
from litagent.schemas import QuestionCluster, RunResult
from litagent.trace import Trace, mask

TOPICS = Path(__file__).parent / "topics.yaml"
OUT = Path(__file__).parent / "results.json"
JUDGE_BUDGET = 20_000  # tokens per topic for the answer-key matcher


class KeyMatch(BaseModel):
    expected_index: int
    produced_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class KeyMatches(BaseModel):
    matches: list[KeyMatch]


def evidence_items(result: RunResult) -> list:
    items = [e for f in [*result.connections, *result.gaps] for e in f.evidence]
    items += [e for c in result.clusters if isinstance(c, QuestionCluster) for d in c.paper_details for e in d.evidence]
    return items


def quote_provenance(result: RunResult) -> dict:
    by_id = {p.id: p for p in result.papers}
    quotes = evidence_items(result)
    verbatim = sum(1 for e in quotes if e.paper_id in by_id and e.quote.strip() and any(
        e.quote in text for text in (by_id[e.paper_id].abstract, by_id[e.paper_id].full_text) if text))
    return {"quotes": len(quotes), "verbatim": verbatim, "rate": verbatim / len(quotes) if quotes else None}


def grading(trace_path: Path) -> dict:
    """First and final grade per stage, read from the run's trace."""
    stages: dict[str, dict] = {}
    if not trace_path.exists():
        return stages
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        action = event["action"]
        if event["agent_name"] != "grader" or not action.startswith("grade ") or "(attempt" not in action:
            continue
        stage = action.split()[1]
        score = int(event["decision"].split("/")[0])
        row = stages.setdefault(stage, {"first": score, "first_pass": "pass" in event["decision"], "attempts": 0})
        row.update(final=score, final_pass="pass" in event["decision"], attempts=row["attempts"] + 1)
    return stages


def match_key(kind: str, expected: list[str], produced: list[dict], settings: Settings) -> dict:
    if not expected:
        return {"status": "awaiting_answer_key", "score": None}
    if settings.sample_mode:
        return {"status": "sample_mode_not_scored", "score": None}
    if not produced:
        return {"status": "scored", "score": 0.0, "matched": 0, "expected": len(expected), "matches": []}
    payload = {"kind": kind, "expected": [{"index": i, "text": t} for i, t in enumerate(expected)], "produced": produced}
    out = complete_json(load_prompt("eval_matcher"), json.dumps(payload, ensure_ascii=False), KeyMatches,
                        settings=settings, agent_name="eval_matcher")
    ids = {p["id"] for p in produced}
    if sorted(m.expected_index for m in out.matches) != list(range(len(expected))) or any(
            set(m.produced_ids) - ids for m in out.matches):
        return {"status": "judge_output_invalid", "score": None}
    matched = sum(bool(m.produced_ids) for m in out.matches)
    return {"status": "scored", "score": matched / len(expected), "matched": matched, "expected": len(expected),
            "matches": [m.model_dump() for m in out.matches],
            "note": "LLM judge against a survey answer key; a diagnostic, not expert agreement."}


def survey_metrics(topic: dict, result: RunResult, settings: Settings) -> dict:
    titles = {p.id: p.title for p in result.papers}
    clusters = [{"id": c.id, "label": c.label, "kind": getattr(c, "kind", "topic"),
                 "paper_titles": [titles.get(pid, pid) for pid in c.paper_ids][:12]}
                for c in result.clusters if len(c.paper_ids) > 1 or getattr(c, "kind", "") == "main"]
    gaps = [{"id": g.id, "question": g.description, "why_it_matters": g.why_it_matters} for g in result.gaps]
    with Trace(settings.logs_dir, f"{result.run_id}-eval") as trace, TokenMeter(JUDGE_BUDGET):
        trace.log("eval_matcher", "match answer key", "start", topic.get("survey", {}).get("title"),
                  why="Offline evaluation only; never part of a user run")
        return {"cluster_agreement": match_key("clusters", topic.get("expected_clusters") or [], clusters, settings),
                "gap_recall": match_key("gaps", topic.get("expected_gaps") or [], gaps, settings)}


def evaluate(topic: dict, settings: Settings) -> dict:
    started = time.monotonic()
    result = run_pipeline(topic["question"], Effort(topic.get("effort", "normal")), AutoApproveReviewer(), settings,
                          run_id=f"eval-{topic['id']}-{time.strftime('%Y%m%d-%H%M%S')}")
    seconds = time.monotonic() - started
    grades = grading(Path(settings.logs_dir) / f"trace-{result.run_id}.jsonl")
    sides = [c for c in result.clusters if getattr(c, "kind", "side") != "main"]
    return {
        "id": topic["id"], "question": topic["question"], "effort": topic.get("effort", "normal"),
        "run_id": result.run_id, "status": result.status, "seconds": round(seconds, 1),
        "usage": result.usage.model_dump() if result.usage else None,
        "papers": len(result.papers),
        "main_papers": sum(len(c.paper_ids) for c in result.clusters if getattr(c, "kind", "") == "main"),
        "side_clusters": len(sides), "singleton_clusters": sum(len(c.paper_ids) == 1 for c in sides),
        "connections": len(result.connections), "gaps": len(result.gaps),
        "quote_provenance": quote_provenance(result), "grades": grades,
        "clusters": [c.label for c in result.clusters], "gap_questions": [g.description for g in result.gaps],
        "warnings": result.warnings, "survey": survey_metrics(topic, result, settings),
    }


def summarise(rows: list[dict]) -> dict:
    stages = [g for r in rows for g in r["grades"].values()]
    quotes = sum(r["quote_provenance"]["quotes"] for r in rows)
    scored = lambda key: [r["survey"][key]["score"] for r in rows if r["survey"][key]["score"] is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    if not rows:
        return {"topics": 0}
    return {"topics": len(rows), "completed": sum(r["status"] == "done" for r in rows),
            "partial": sum(r["status"] == "partial" for r in rows),
            "quote_provenance_rate": sum(r["quote_provenance"]["verbatim"] for r in rows) / quotes if quotes else None,
            "first_attempt_pass_rate": mean([g["first_pass"] for g in stages]),
            "final_pass_rate": mean([g["final_pass"] for g in stages]),
            "mean_cluster_agreement": mean(scored("cluster_agreement")),
            "mean_gap_recall": mean(scored("gap_recall")),
            "mean_seconds": mean([r["seconds"] for r in rows]),
            "mean_tokens": mean([r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in rows if r["usage"]])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--only", action="append", help="Topic id to run (repeatable)")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    settings = Settings()
    settings.sample_mode = args.sample_mode or settings.sample_mode
    topics = [t for t in yaml.safe_load(TOPICS.read_text(encoding="utf-8"))["topics"] if not args.only or t["id"] in args.only]
    rows, errors = [], []
    for topic in topics:
        try:
            rows.append(evaluate(topic, settings))
        except Exception as error:  # one topic's crash is a result too; keep evaluating the others
            errors.append({"id": topic["id"], "error": f"{type(error).__name__}: {mask(str(error))[:300]}"})
            print(f"{topic['id']}: ERROR {errors[-1]['error']}", flush=True)
        else:
            r = rows[-1]
            print(f"{r['id']}: {r['status']} in {r['seconds']} s, {r['papers']} papers, {r['side_clusters']} side clusters, "
                  f"{r['connections']} connections, {r['gaps']} gaps, quotes verbatim {r['quote_provenance']['verbatim']}/"
                  f"{r['quote_provenance']['quotes']}", flush=True)
        report = {"sample_mode": settings.sample_mode, "summary": {**summarise(rows), "errors": len(errors)},
                  "topics": rows, "errors": errors,
                  "scope": "Workflow metrics are measured in code. Survey metrics need a human-chosen answer key and use an "
                           "LLM judge; grades are model judgments. None of this is expert-validated accuracy."}
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # after every topic
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
