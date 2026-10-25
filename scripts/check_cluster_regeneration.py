"""Offline API smoke: create a sample map, preview/apply a move, retry and roll back.

Uses the real API and sample-mode agents, never a mocked model or fabricated quality score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from litagent import api  # noqa: E402
from litagent.cluster_state import path_for  # noqa: E402
from litagent.config import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/cluster-regeneration-smoke.json"))
    args = parser.parse_args()
    settings = Settings(sample_mode=True)
    papers = json.loads(Path("data/sample/low_resource_languages_llm.json").read_text())[:6]
    api.app.dependency_overrides[api.get_settings] = lambda: settings
    def checked(response):
        response.raise_for_status()
        body = response.json()
        if body["status"] != "success":
            raise RuntimeError(body["error"])
        return body["output"]
    try:
        with TestClient(api.app) as client:
            root = checked(client.post("/cluster", json={"input": {
                "query": "How are LLMs adapted for low-resource languages?", "papers": papers}}))
            run_id = root["run_id"]
            result_path = path_for(settings, "cluster-map", run_id)
            original_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
            choices = checked(client.get(f"/cluster/{run_id}/regenerate/candidates"))
            selected = next(c for c in choices["candidates"] if c["eligible"])
            plan = checked(client.post(f"/cluster/{run_id}/regenerate/preview", json={"moves": [{
                "paper_id": selected["paper_id"], "target_cluster_id": selected["target_cluster_id"]}]}))
            child = checked(client.post(f"/cluster/{run_id}/regenerate", json={"preview_id": plan["preview_id"]}))
            replay = checked(client.post(f"/cluster/{run_id}/regenerate", json={"preview_id": plan["preview_id"]}))
            history = checked(client.get(f"/cluster/{run_id}/versions"))
            restored = checked(client.post(f"/cluster/{run_id}/activate", json={"target_run_id": run_id,
                "expected_head_run_id": child["run_id"], "expected_head_version_token": child["version_token"]}))
            checks = {"preview_applicable": plan["applicable"], "source_snapshot_unchanged":
                hashlib.sha256(result_path.read_bytes()).hexdigest() == original_hash,
                "one_membership_per_paper": sorted(pid for c in child["clusters"] for pid in c["paper_ids"]) == sorted(p["id"] for p in papers),
                "idempotent_replay": replay == child, "child_became_active": history["active_head"]["run_id"] == child["run_id"],
                "rollback_restored_source": restored["run_id"] == run_id,
                "zero_llm_calls": root["usage"]["llm_calls"] == child["usage"]["llm_calls"] == 0,
                "reused_embeddings": child["regeneration"]["reused_embeddings"],
                "reused_similarity_matrix": child["regeneration"]["reused_similarity_matrix"]}
            report = {"sample_mode": True, "quality_claim": None,
                "note": "API/versioning smoke only. Synthetic sample scores and placeholder topics; not a semantic quality evaluation.",
                "root_run_id": run_id, "child_run_id": child["run_id"], "paper_count": len(papers),
                "checks": checks, "preview": plan, "regeneration": child["regeneration"],
                "child_usage": child["usage"]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"Checks: {sum(checks.values())}/{len(checks)}; report: {args.output}")
            return 0 if all(checks.values()) else 1
    finally:
        api.app.dependency_overrides.clear()


if __name__ == "__main__":
    raise SystemExit(main())
