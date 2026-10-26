"""Independent, cached per-claim model review. Never writes human labels or map findings."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAIError

from litagent.cluster_state import VersionError, atomic_json, digest, file_hash, load_version, path_for, version_lock
from litagent.finding_engine import FindingContext, exact_ids, resolve_evidence, sources
from litagent.finding_review import prepare_review
from litagent.integrations.llm import LLMError, TokenMeter, complete_json, load_prompt
from litagent.schemas import Evidence, ReviewAgentAssessment, ReviewAgentDecisions, ReviewExplanationChecks
from litagent.trace import Trace, current_trace, new_run_id


def _metric_mentions(source, settings):
    # Literal source mentions, not a claim that the paper measured every named metric.
    return [{"term": term, "span_id": span["span_id"]}
            for term in settings.review_agent_metric_terms for span in source["spans"]
            if re.search(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", span["quote"], re.IGNORECASE)]


def _unqualified_metric_omission(text):
    match = re.search(r"未(?:明确)?(?:说明|给出|披露|报告|提及|列出)[^，,。；;！？!?]*"
                      r"(?:评测|评估|评价)?指标\s*(?:[，,。；;]|$)", text)
    return bool(match and not re.search(r"完整|全部|详细|具体|版本|定义|实现|配置", match.group()))


def _inputs(run_id, settings):
    if (min(settings.review_agent_source_chars, settings.review_agent_max_items,
            settings.review_agent_token_budget, settings.finding_min_quote_chars) < 1
            or settings.finding_span_chars < settings.finding_min_quote_chars
            or settings.finding_validation_retries < 0
            or not math.isfinite(settings.review_agent_timeout_s) or settings.review_agent_timeout_s <= 0):
        raise VersionError("Invalid review Agent limits", 422)
    pack = prepare_review(run_id, settings)
    result, state, _ = load_version(run_id, settings)
    if len(pack["items"]) > settings.review_agent_max_items:
        raise VersionError("Review queue exceeds the configured Agent item limit", 422)
    context = FindingContext(question=result.question, use_full_text=result.use_full_text)
    source_map = sources(state.papers, settings, context, limit=settings.review_agent_source_chars)
    # Explicit allowlist: no human labels/reasons, old model verdicts/scores, or similarities.
    items = [{k: item[k] for k in ("id", "kind", "paper_ids", "claim", "proposed_test")}
             for item in pack["items"]]
    used = {pid for item in items for pid in item["paper_ids"]}
    original = {p.id: p for p in state.papers}
    selected = {pid: source_map[pid] for pid in sorted(used) if pid in source_map}
    for pid, src in selected.items():
        text = getattr(original[pid], src["source_field"]) or ""
        src["truncated"] = len(text) > settings.review_agent_source_chars
        src["source_chars_available"] = len(text)
        src["metric_mentions"] = _metric_mentions(src, settings)
    payload = {"question": result.question, "items": items, "papers": list(selected.values())}
    key = digest({"run_id": result.run_id, "version_token": result.version_token, "corpus_fingerprint": pack["corpus_fingerprint"],
        "payload": payload, "sample_mode": result.sample_mode, "model": settings.llm_model,
        "prompt": load_prompt("review_agent"), "schema": ReviewAgentDecisions.model_json_schema(),
        "audit_prompt": load_prompt("review_explanation_check"), "audit_schema": ReviewExplanationChecks.model_json_schema(),
        "output_schema": ReviewAgentAssessment.model_json_schema(),
        "settings": {k: v for k, v in vars(settings).items() if k.startswith("review_agent_") or
                     k in {"finding_span_chars", "finding_min_quote_chars", "finding_validation_retries"}},
        "implementation": [file_hash(Path(__file__)), file_hash(Path(__file__).parents[1] / "finding_engine.py")]})
    return pack, payload, selected, key


def _cached(key, settings):
    path = path_for(settings, "review-agent", key)
    if not path.exists():
        return None
    envelope = json.loads(path.read_text())
    report = envelope["report"]
    if envelope["checksum"] != digest(report) or report["input_fingerprint"] != key:
        raise VersionError("Cached AI review checksum mismatch")
    for item in report["items"]:
        ReviewAgentAssessment.model_validate(item)
    return report


def read_review(run_id, settings):
    _, _, _, key = _inputs(run_id, settings)
    return {"report": _cached(key, settings)}


def _complete_before_deadline(prompt, payload, schema, settings, deadline, agent_name):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LLMError("AI review deadline reached; no human annotations were changed")
    # complete_json can repair JSON once and the SDK can retry HTTP once.
    cfg = replace(settings, llm_timeout_s=min(settings.llm_timeout_s, max(.1, remaining / 4)))
    try:
        result = complete_json(load_prompt(prompt), json.dumps(payload, ensure_ascii=False),
                               schema, settings=cfg, agent_name=agent_name)
    except OpenAIError as error:
        raise LLMError(f"AI review provider failed ({type(error).__name__}); retry explicitly") from error
    if time.monotonic() >= deadline:
        raise LLMError("AI review deadline reached; no human annotations were changed")
    return result


def _audit_explanations(payload, output, settings, deadline):
    # Fresh context, no scores/verdicts and no repair history to anchor the factual check.
    statements = {d.item_id: [
        {"statement_id": f"s{n}", "text": text}
        for n, text in enumerate(part.strip() for field in [d.reason, *d.limitations]
                                 for part in re.split(r"(?<=[。！？!?])\s*|\n+", field) if part.strip())]
        for d in output.decisions}
    audit_input = {**payload, "draft_explanations": [
        {"item_id": d.item_id, "reason": d.reason, "limitations": d.limitations,
         "statements": statements[d.item_id]}
        for d in output.decisions]}
    audit = _complete_before_deadline("review_explanation_check", audit_input, ReviewExplanationChecks,
                                      settings, deadline, "review_explanation_check")
    if trace := current_trace():
        trace.log("review_explanation_check", "receive assertion table", "validate",
                  result=audit.model_dump(mode="json"), why="Retain audit output even when source/coverage validation fails")
    targets = {i["id"]: i for i in payload["items"]}
    source_map = {p["id"]: p for p in payload["papers"]}
    exact_ids([c.item_id for c in audit.checks], targets, "audit item_id")
    issues = []
    for check in audit.checks:
        actual = {a.statement_id for a in check.attributions}
        expected = {s["statement_id"] for s in statements[check.item_id]}
        if actual != expected:
            raise ValueError(f"Explanation audit must check EVERY supplied draft statement_id for {check.item_id}: "
                             f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
        if {a.paper_id for a in check.attributions} != set(targets[check.item_id]["paper_ids"]):
            raise ValueError("Explanation audit must check assertions about both papers")
        errors = list(check.issues)
        for attribution in check.attributions:
            statement = next(s["text"] for s in statements[check.item_id] if s["statement_id"] == attribution.statement_id)
            # A named metric in this paper's supplied source contradicts an unqualified
            # assertion that the abstract supplies no metrics. Detailed definitions,
            # versions or full metric lists may still be missing; do not infer those.
            paper_ids = {a.paper_id for a in check.attributions if a.statement_id == attribution.statement_id}
            if paper_ids == {attribution.paper_id} and _unqualified_metric_omission(statement):
                mentions = _metric_mentions(source_map[attribution.paper_id], settings)
                if mentions:
                    errors.append(f"{attribution.paper_id}: supplied source explicitly mentions metrics "
                                  f"{sorted({m['term'] for m in mentions})}; cannot claim the abstract gives no metrics. "
                                  "State which metrics are reported and qualify which details are missing.")
            if attribution.assessment == "grounded" or attribution.evidence:
                resolve_evidence(attribution.evidence, source_map, {attribution.paper_id}, settings)
            if attribution.assessment in {"unsubstantiated", "contradicted"}:
                errors.append(f"{attribution.paper_id}: {attribution.assertion}: {attribution.explanation}")
        if errors:
            issues.append({"item_id": check.item_id, "issues": errors})
    if trace := current_trace():
        trace.log("review_explanation_check", "check per-paper factual attribution",
                  "repair" if issues else "pass", result=audit.model_dump(mode="json"),
                  why="Claim/test and draft prose are not evidence; original spans are bound to paper identities")
    if issues:
        raise ValueError("Explanation source attribution failed: " + json.dumps(issues, ensure_ascii=False))


def _grade(payload, source_map, settings, deadline):
    targets = {item["id"]: item for item in payload["items"]}
    message = dict(payload)
    for attempt in range(settings.finding_validation_retries + 1):
        output = _complete_before_deadline("review_agent", message, ReviewAgentDecisions,
                                           settings, deadline, "review_agent")
        try:
            exact_ids([d.item_id for d in output.decisions], targets, "item_id")
            assessed = []
            for decision in output.decisions:
                pair = targets[decision.item_id]
                if decision.verdict == "supported" and min(decision.scores.model_dump().values()) < 2:
                    raise ValueError("Supported requires every rubric dimension to be at least 2")
                if decision.verdict != "uncertain":
                    evidence = resolve_evidence(decision.evidence, source_map, set(pair["paper_ids"]), settings)
                else:
                    evidence = []
                    for ref in decision.evidence:
                        if ref.paper_id not in pair["paper_ids"] or ref.paper_id not in source_map:
                            raise ValueError("Evidence must refer to this pair's supplied source")
                        spans = {s["span_id"]: s["quote"] for s in source_map[ref.paper_id]["spans"]}
                        quote = spans.get(ref.span_id, "")
                        if len(quote.strip()) < settings.finding_min_quote_chars:
                            raise ValueError("Evidence must identify an original sufficiently long source span")
                        evidence.append(Evidence(paper_id=ref.paper_id, quote=quote))
                assessed.append(ReviewAgentAssessment(item_id=decision.item_id, status="assessed",
                    verdict=decision.verdict, scores=decision.scores, total=sum(decision.scores.model_dump().values()),
                    reason=decision.reason, limitations=decision.limitations, evidence=evidence))
            _audit_explanations(payload, output, settings, deadline)
            return assessed
        except ValueError as error:
            if attempt == settings.finding_validation_retries:
                raise LLMError(f"AI review source validation failed: {error}") from error
            message.update(validation_error=str(error), rejected_output=output.model_dump(mode="json"))
    raise AssertionError("unreachable")


def run_review(run_id, expected_version_token, settings):
    pack, payload, source_map, key = _inputs(run_id, settings)
    if pack["version_token"] != expected_version_token:
        raise VersionError("Review version is stale; refresh the source map")
    if pack["sample_mode"] != settings.sample_mode:
        raise VersionError("Server and source map modes differ; switch to the matching mode before running AI review")
    with version_lock(settings, "ai-review-" + key[:32]):
        if report := _cached(key, settings):
            return {"report": report, "reused": True}
        report_id = new_run_id()
        with Trace(settings.logs_dir, report_id) as trace, TokenMeter(settings.review_agent_token_budget) as meter:
            trace.log("review_agent", "independent source assessment", "exclude human annotations and earlier model verdicts",
                      result={"source_run_id": run_id, "item_count": len(payload["items"])},
                      why="Model-only suggestions; source map and human review sidecar stay immutable")
            assessed, gradable = [], []
            for item in payload["items"]:
                if not item["claim"]:
                    assessed.append(ReviewAgentAssessment(item_id=item["id"], status="needs_hypothesis",
                        reason="当前只有论文对，尚无具体研究假设。请先明确研究问题、比较方法和实验设置，再评分。"))
                elif not set(item["paper_ids"]) <= set(source_map):
                    assessed.append(ReviewAgentAssessment(item_id=item["id"], status="insufficient_sources",
                        verdict="uncertain", reason="缺少可核验的双方正文，暂不评分。"))
                elif settings.sample_mode:
                    assessed.append(ReviewAgentAssessment(item_id=item["id"], status="sample",
                        reason="样例模式：展示 Agent 评审流程，未调用模型，不提供科学质量评分。"))
                else:
                    gradable.append(item)
            if gradable:
                ids = {pid for item in gradable for pid in item["paper_ids"]}
                clean = {**payload, "items": gradable, "papers": [source_map[pid] for pid in sorted(ids)]}
                assessed += _grade(clean, source_map, settings,
                                   time.monotonic() + min(settings.review_agent_timeout_s, settings.run_timeout_s))
            by_id = {item.item_id: item for item in assessed}
            report = {"report_id": report_id, "run_id": run_id, "version_token": pack["version_token"],
                "input_fingerprint": key, "provenance": "model", "sample_mode": settings.sample_mode,
                "model": settings.llm_model if not settings.sample_mode else None,
                "created_at": datetime.now(timezone.utc).isoformat(), "usage": meter.usage.model_dump(mode="json"),
                "items": [by_id[item["id"]].model_dump(mode="json") for item in payload["items"]],
                "explanation_check": {"method": "separate_context_source_attribution",
                                      "checked_item_ids": [item["id"] for item in gradable]},
                "limitations": ["只评估当前具体结论及已提供原文，不检索外部文献，也不确认全球新颖性。",
                    "分数是模型的分项判断，不是准确率；解释经独立上下文核验，但使用同一配置模型，仍可能存在共同偏差。"]}
            atomic_json(path_for(settings, "review-agent", key), {"checksum": digest(report), "report": report})
            trace.log("review_agent", "save separate AI report", "done",
                      result={"scored": len(gradable), "unscored": len(assessed) - len(gradable)},
                      why="Exact source spans checked; human judgments and accepted map findings were not overwritten")
            return {"report": report, "reused": False}
