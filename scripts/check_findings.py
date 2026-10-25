"""Analyze a cached /cluster map through the actual API without retrieval or re-embedding."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from litagent import api  # noqa: E402
from litagent.cluster_state import load_version  # noqa: E402
from litagent.config import Settings  # noqa: E402
from litagent.trace import mask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("logs/grounded-findings-report.json"))
    args = parser.parse_args()
    cfg = Settings(sample_mode=args.sample_mode)
    source, _, source_arrays = load_version(args.run_id, cfg)
    request_id = args.request_id or uuid.uuid4().hex
    print(f"Source: {source.run_id}; request ID: {request_id}", flush=True)
    api.app.dependency_overrides[api.get_settings] = lambda: cfg
    try:
        with TestClient(api.app) as client:
            response = client.post(f"/cluster/{source.run_id}/analyze", json={
                "expected_version_token": source.version_token, "request_id": request_id})
            response.raise_for_status()
            output = response.json()["output"]
        result, state, arrays = load_version(output["run_id"], cfg)
        by_id = {p.id: p for p in state.papers}
        grounded = all({e.paper_id for e in f.evidence} == set(getattr(f, "paper_ids", getattr(f, "between_paper_ids", [])))
                       and all(any(e.quote in text for text in (by_id[e.paper_id].abstract, by_id[e.paper_id].full_text) if text)
                               for e in f.evidence) for f in [*result.connections, *result.gaps])
        report = {"sample_mode": result.sample_mode, "source_run_id": source.run_id, "finding_run_id": result.run_id,
            "request_id": request_id, "status": result.status, "paper_count": len(state.papers),
            "main_cluster_id": result.main_cluster_id, "main_paper_ids": [pid for c in result.clusters if c.id == result.main_cluster_id for pid in c.paper_ids],
            "connection_count": len(result.connections), "gap_count": len(result.gaps),
            "exact_original_quotes_checked": grounded, "reused_numeric_cache": all(np.array_equal(a, b) for a, b in (
                (source_arrays.paper_vectors, arrays.paper_vectors), (source_arrays.question_vector, arrays.question_vector),
                (source_arrays.similarity, arrays.similarity))),
            "analysis": result.analysis.model_dump(mode="json"), "usage": result.usage.model_dump(),
            "warnings": result.warnings, "connections": [c.model_dump(mode="json") for c in result.connections],
            "gaps": [g.model_dump(mode="json") for g in result.gaps],
            "evaluation_note": "Real source-reference and workflow audit; semantic support and grades are model judgments, not expert accuracy. Gaps do not establish global novelty."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Saved: {result.run_id}; {len(result.connections)} connections, {len(result.gaps)} provisional gaps; {result.status}")
        print(f"Calls: {result.usage.llm_calls}; tokens: {result.usage.total}; original quotes: {grounded}; report: {args.output}")
        return 0 if result.status == "done" and grounded else 2
    except Exception as error:
        print(mask(f"FAILED: {type(error).__name__}: {error}"), file=sys.stderr)
        return 1
    finally:
        api.app.dependency_overrides.clear()


if __name__ == "__main__":
    raise SystemExit(main())
