"""Read-only map catalog/source access plus explicitly submitted human review."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends
from fastapi.staticfiles import StaticFiles

from litagent.cluster_state import ClusterState, VersionError, digest, head, load_result, load_version, path_for
from litagent.finding_review import ReviewSubmission, prepare_review, save_review
from litagent.agents import review_agent
from litagent.schemas import ReviewAgentRequest
from litagent.trace import mask


def map_catalog(settings):
    rows, warnings = [], []
    for path in Path(settings.logs_dir).glob("cluster-state-*.json"):
        try:
            envelope = json.loads(path.read_text())
            if envelope["checksum"] != digest(envelope["state"]):
                raise ValueError("Invalid manifest checksum")
            state = ClusterState.model_validate(envelope["state"])
            if path != path_for(settings, "cluster-state", state.run_id):
                raise ValueError("Invalid manifest identity")
            result = load_result(state.run_id, settings)
            raw = json.loads(path_for(settings, "cluster-map", state.run_id).read_text())
            if digest(raw) != state.result_hash:
                raise ValueError("Invalid result checksum")
            active = head(state.root_run_id, settings)
            rows.append({"run_id": result.run_id, "root_run_id": state.root_run_id, "parent_run_id": result.parent_run_id,
                "question": result.question, "status": result.status, "stage": result.stage,
                "revision": result.revision, "sample_mode": result.sample_mode, "paper_count": len(state.papers),
                "cluster_count": len(result.clusters), "active": active["run_id"] == result.run_id,
                "connection_count": len(result.connections), "gap_count": len(result.gaps)})
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"Skipped {path.name}: {mask(str(error))}")
    rows.sort(key=lambda r: (not r["active"], r["sample_mode"], r["run_id"]), reverse=False)
    # Prefer the most recent active real map; history stays available in the selector.
    rows.sort(key=lambda r: r["run_id"], reverse=True)
    rows.sort(key=lambda r: (not r["active"], r["sample_mode"]))
    return {"maps": rows, "warnings": warnings}


def install_map_routes(app, get_settings, respond):
    @app.get("/maps")
    def maps(settings=Depends(get_settings)):
        return respond(None, lambda: map_catalog(settings))

    @app.get("/ui/config")
    def config(settings=Depends(get_settings)):
        return {"status": "success", "output": {"sample_mode": settings.sample_mode, "run_timeout_s": settings.run_timeout_s}}

    @app.get("/cluster/{run_id}/papers")
    def papers(run_id: str, include_full_text: bool = False, settings=Depends(get_settings)):
        return respond(run_id, lambda: {"papers": [p.model_dump(mode="json", exclude=set() if include_full_text else {"full_text"})
            for p in load_version(run_id, settings)[1].papers]})

    @app.get("/cluster/{run_id}/papers/{paper_id}")
    def paper(run_id: str, paper_id: str, settings=Depends(get_settings)):
        def read():
            found = next((p for p in load_version(run_id, settings)[1].papers if p.id == paper_id), None)
            if found is None:
                raise VersionError("Unknown paper in this snapshot", 404)
            return found
        return respond(run_id, read)

    @app.get("/cluster/{run_id}/review")
    def review(run_id: str, reveal: bool = False, settings=Depends(get_settings)):
        return respond(run_id, lambda: prepare_review(run_id, settings, reveal=reveal))

    @app.post("/cluster/{run_id}/review/{item_id}")
    def submit_review(run_id: str, item_id: str, body: ReviewSubmission, settings=Depends(get_settings)):
        return respond(run_id, lambda: save_review(run_id, item_id, body, settings))

    @app.get("/cluster/{run_id}/review-agent")
    def cached_agent_review(run_id: str, settings=Depends(get_settings)):
        return respond(run_id, lambda: review_agent.read_review(run_id, settings))

    @app.post("/cluster/{run_id}/review-agent")
    def agent_review(run_id: str, body: ReviewAgentRequest, settings=Depends(get_settings)):
        return respond(run_id, lambda: review_agent.run_review(run_id, body.expected_version_token, settings))

    directory = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if directory.exists():
        app.mount("/", StaticFiles(directory=directory, html=True), name="literature-map")
