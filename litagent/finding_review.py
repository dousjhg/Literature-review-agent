"""Version-bound human review sidecars. Generating a pack never generates human labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from litagent.cluster_audit import corpus_fingerprint
from litagent.cluster_state import VersionError, atomic_json, digest, load_version, path_for, version_lock
from litagent.finding_engine import FindingContext, choose_pairs, sources
from litagent.schemas import Evidence


class ReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_version_token: str = Field(pattern=r"^[a-f0-9]{32}$")
    verdict: Literal["supported", "unsupported", "uncertain"]
    reviewer: str = Field(min_length=1, max_length=80)
    rationale: str = Field(min_length=10, max_length=4000)
    evidence: list[Evidence] = Field(default_factory=list)
    saw_model_assessment: bool = False


def prepare_review(run_id, settings, *, reveal=False):
    result, state, arrays = load_version(run_id, settings)
    context = FindingContext(question=result.question, similarity=arrays.similarity, use_full_text=result.use_full_text)
    src = sources(state.papers, settings, context)
    gap_pairs, _ = choose_pairs("gaps", state.papers, result.clusters, settings, context, set(src))
    by_id = {p.id: p for p in state.papers}
    owner = {pid: c.id for c in result.clusters for pid in c.paper_ids}
    previous = {d["pair_id"]: d for a in (result.analysis.audits if result.analysis else []) if a.stage == "gaps" for d in a.decisions}
    items, selection = [], {}
    for conn in result.connections:
        item_id = "connection-" + digest(sorted(conn.paper_ids))[:16]
        items.append({"id": item_id, "kind": "connection", "scope": conn.kind, "paper_ids": conn.paper_ids,
            "cluster_ids": conn.cluster_ids, "claim": conn.description, "proposed_test": None})
        selection[item_id] = {"decision": "accepted", "finding_id": conn.id}
    for pair in gap_pairs:
        item_id = "gap-" + pair["pair_id"].removeprefix("PAIR-")
        old = previous.get(pair["pair_id"], {})
        proposal = old.get("proposal") or {}
        items.append({"id": item_id, "kind": "gap", "scope": pair.get("scope", "within_cluster"),
            "paper_ids": pair["paper_ids"], "cluster_ids": pair["cluster_ids"],
            "claim": proposal.get("research_question"), "proposed_test": proposal.get("proposed_test")})
        selection[item_id] = {"decision": old.get("decision", "not_assessed"), "reason": old.get("reason")}
    fingerprint = corpus_fingerprint(state.papers)
    path = path_for(settings, "finding-review", run_id)
    stored = json.loads(path.read_text()) if path.exists() else {}
    if stored and (stored.get("version_token") != result.version_token or stored.get("corpus_fingerprint") != fingerprint):
        raise VersionError("Review source version or original text changed; annotations cannot be carried over")
    records = stored.get("records", {})
    for item in items:
        item["papers"] = [{"id": pid, "title": by_id[pid].title, "year": by_id[pid].year,
            "source_field": src.get(pid, {}).get("source_field", "abstract"),
            "source_text": (by_id[pid].full_text if result.use_full_text and by_id[pid].full_text else by_id[pid].abstract) or "",
            "cluster_id": owner[pid]} for pid in item["paper_ids"]]
        item["annotation"] = records.get(item["id"])
        if reveal:
            item["model_assessment"] = selection[item["id"]]
    valid = {item["id"] for item in items}
    completed = [r for key, r in records.items() if key in valid]
    decisive = [r for r in completed if r["verdict"] != "uncertain"]
    return {"run_id": run_id, "version_token": result.version_token, "question": result.question,
        "sample_mode": result.sample_mode, "corpus_fingerprint": fingerprint,
        "instructions": "Assess source support for a relationship or testable provisional hypothesis. Supported does not establish global novelty. Select exact original quotes from BOTH papers for decisive judgments; uncertain is valid. Model judgments are hidden by default. Reviewer identity is self-declared, not authenticated.",
        "items": items, "summary": {"total": len(items), "reviewed": len(completed), "pending": len(items)-len(completed),
            "supported": sum(r["verdict"] == "supported" for r in completed),
            "unsupported": sum(r["verdict"] == "unsupported" for r in completed),
            "uncertain": sum(r["verdict"] == "uncertain" for r in completed),
            "blind_reviewed": sum(not r.get("saw_model_assessment", False) for r in completed),
            "support_rate": sum(r["verdict"] == "supported" for r in decisive)/len(decisive) if decisive else None,
            "semantic_accuracy": None}}


def save_review(run_id, item_id, submission: ReviewSubmission, settings):
    with version_lock(settings, "review-" + digest(run_id)[:32]):
        pack = prepare_review(run_id, settings)
        if submission.expected_version_token != pack["version_token"]:
            raise VersionError("Review version is stale; refresh the source map")
        item = next((item for item in pack["items"] if item["id"] == item_id), None)
        if item is None:
            raise VersionError("Unknown review item", 404)
        by_id = {p["id"]: p["source_text"] for p in item["papers"]}
        if any(e.paper_id not in by_id or len(e.quote.strip()) < settings.finding_min_quote_chars or e.quote not in by_id[e.paper_id] for e in submission.evidence):
            raise VersionError("Review quotes must be exact text from this item's original paper fields", 422)
        if submission.verdict != "uncertain" and {e.paper_id for e in submission.evidence} != set(item["paper_ids"]):
            raise VersionError("Decisive review requires exact quotes from BOTH papers", 422)
        path = path_for(settings, "finding-review", run_id)
        stored = json.loads(path.read_text()) if path.exists() else {"version_token": pack["version_token"],
            "corpus_fingerprint": pack["corpus_fingerprint"], "records": {}, "history": []}
        record = {**submission.model_dump(mode="json", exclude={"expected_version_token"}), "provenance": "human",
                  "source": "user_submission", "submitted_at": datetime.now(timezone.utc).isoformat()}
        if item_id in stored["records"]:
            stored["history"].append({"item_id": item_id, "annotation": stored["records"][item_id]})
        stored["records"][item_id] = record
        atomic_json(path, stored)
        return {"annotation": record, "summary": prepare_review(run_id, settings)["summary"]}
