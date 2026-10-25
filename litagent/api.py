"""Programmatic interface: POST /run takes JSON and returns JSON.

Request:  {"run_id": optional str, "input": {"query": str, "effort": "normal|medium|high|ultra",
           "feedback": optional str, "signed_in": optional bool}}
Success:  {"run_id": ..., "status": "success", "output": {...RunResult fields...}}
Failure:  {"run_id": ..., "status": "error", "error": {"type": ..., "message": ...}}

GET /world/{run_id} returns a run's saved Meterless World Model: entities, relationships, facts,
the merge review queue and the audit log.
POST /cluster accepts a query and saved papers for a standalone cluster-stage preview;
GET /cluster/{run_id} reads its saved result without running later agents.

Human reviews never wait for input here: both are approved, unless `feedback` is given,
in which case Human review 1 sends it back to the Planner once (see RequestReviewer).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from litagent.config import EFFORT_PROFILES, Effort, Settings
from litagent.cluster_preview import run_cluster_map
from litagent import cluster_regeneration
from litagent.finding_analysis import analyze_cluster_map
from litagent.cluster_state import VersionError, path_for, version_lock
from litagent.human_review import RequestReviewer
from litagent.integrations.meterless import load_world
from litagent.orchestrator import run_pipeline
from litagent.schemas import ClusterMapResult, ClusterMove, Paper
from litagent.trace import mask, new_run_id, setup_logging

setup_logging()
app = FastAPI(title="Literature review agent")


class RunInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    effort: Effort = Effort.NORMAL
    feedback: str | None = None  # optional: triggers one re-plan at Human review 1
    signed_in: bool = False


RUN_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"  # run ids name files in logs/, so keep them to safe characters


class RunRequest(BaseModel):
    run_id: str | None = Field(default=None, pattern=RUN_ID_PATTERN)
    input: RunInput


class ClusterInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1)
    effort: Effort = Effort.NORMAL
    papers: list[Paper] = Field(min_length=1, max_length=EFFORT_PROFILES[Effort.ULTRA].max_papers)

    @model_validator(mode="after")
    def bounded_unique_papers(self):
        if len({p.id for p in self.papers}) != len(self.papers):
            raise ValueError("Paper IDs must be unique")
        if len(self.papers) > EFFORT_PROFILES[self.effort].max_papers:
            raise ValueError("Paper count exceeds the selected effort limit")
        return self


class ClusterRequest(BaseModel):
    run_id: str | None = Field(default=None, pattern=RUN_ID_PATTERN)
    input: ClusterInput


class MovePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    moves: list[ClusterMove] = Field(min_length=1)


class RegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class ActivateVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_run_id: str = Field(pattern=RUN_ID_PATTERN)
    expected_head_run_id: str = Field(pattern=RUN_ID_PATTERN)
    expected_head_version_token: str = Field(pattern=r"^[a-f0-9]{32}$")


class AnalyzeMapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version_token: str = Field(pattern=r"^[a-f0-9]{32}$")
    request_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


def get_settings() -> Settings:
    return Settings()


def _error(run_id: str | None, status_code: int, err_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"run_id": run_id, "status": "error", "error": {"type": err_type, "message": mask(message)}},
    )


@app.exception_handler(RequestValidationError)
def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = exc.body if isinstance(exc.body, dict) else {}
    run_id = body.get("run_id") if isinstance(body.get("run_id"), str) else None
    problems = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return _error(run_id, 422, "InvalidInput", problems)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest, settings: Settings = Depends(get_settings)) -> Any:
    run_id = req.run_id or new_run_id()
    try:
        with version_lock(settings, run_id):
            if any(path_for(settings, prefix, run_id).exists() for prefix in ("cluster-map", "cluster-state")):
                raise VersionError("Run ID names an immutable cluster version; use a new run ID")
            result = run_pipeline(
                req.input.query,
                req.input.effort,
                RequestReviewer(req.input.feedback),
                settings,
                signed_in=req.input.signed_in,
                run_id=run_id,
            )
    except VersionError as error:
        return _error(run_id, error.status, "VersionConflict", str(error))
    except Exception as e:  # the trace already has the error and the failing stage
        return _error(run_id, 500, type(e).__name__, str(e))
    return {"run_id": run_id, "status": "success", "output": result.model_dump(mode="json")}


@app.get("/world/{run_id}")
def world(run_id: str, settings: Settings = Depends(get_settings)) -> Any:
    if not re.fullmatch(RUN_ID_PATTERN, run_id):
        return _error(None, 422, "InvalidInput", "run_id may only contain letters, digits, '.', '_' and '-'")
    model = load_world(run_id, settings.logs_dir)
    if model is None:
        return _error(run_id, 404, "NotFound", f"No World Model saved for run {run_id}")
    shape = model.state_shape()
    return {"run_id": run_id, "status": "success", "output": {
        "entities": shape["entities"], "contexts": shape["contexts"], "relationships": shape["relationships"],
        "facts": shape["facts"], "merge_queue": model.merge_queue(), "audit": model.audit(), "events": len(model.events()),
    }}


@app.post("/cluster")
def cluster_preview(req: ClusterRequest, settings: Settings = Depends(get_settings)) -> Any:
    run_id = req.run_id or new_run_id()
    try:
        result = run_cluster_map(req.input.papers, req.input.query, req.input.effort, settings, run_id)
    except VersionError as error:
        return _error(run_id, error.status, {409: "VersionConflict", 404: "NotFound"}.get(error.status, "InvalidInput"), str(error))
    except Exception as error:
        return _error(run_id, 500, type(error).__name__, str(error))
    return {"run_id": run_id, "status": "success", "output": result.model_dump(mode="json")}


def _version_response(run_id: str, operation):
    try:
        result = operation()
        return {"run_id": getattr(result, "run_id", run_id), "status": "success",
                "output": result.model_dump(mode="json") if isinstance(result, BaseModel) else result}
    except VersionError as error:
        return _error(run_id, error.status, {409: "VersionConflict", 404: "NotFound"}.get(error.status, "InvalidInput"), str(error))
    except Exception as error:
        return _error(run_id, 500, type(error).__name__, str(error))


@app.get("/cluster/{run_id}/regenerate/candidates")
def regeneration_candidates(run_id: str, near_only: bool = False, paper_id: str | None = None, settings: Settings = Depends(get_settings)):
    return _version_response(run_id, lambda: cluster_regeneration.candidates(run_id, settings, near_only=near_only, paper_id=paper_id))


@app.post("/cluster/{run_id}/regenerate/preview")
def preview_regeneration(run_id: str, req: MovePreviewRequest, settings: Settings = Depends(get_settings)):
    return _version_response(run_id, lambda: cluster_regeneration.preview_moves(run_id, req.moves, settings))


@app.post("/cluster/{run_id}/regenerate")
def regenerate(run_id: str, req: RegenerateRequest, settings: Settings = Depends(get_settings)):
    return _version_response(run_id, lambda: cluster_regeneration.apply_moves(run_id, req.preview_id, settings))


@app.post("/cluster/{root_id}/activate")
def activate_cluster_version(root_id: str, req: ActivateVersionRequest, settings: Settings = Depends(get_settings)):
    return _version_response(root_id, lambda: cluster_regeneration.activate_version(root_id, req.target_run_id,
        req.expected_head_run_id, req.expected_head_version_token, settings))


@app.get("/cluster/{run_id}/versions")
def cluster_versions(run_id: str, settings: Settings = Depends(get_settings)):
    return _version_response(run_id, lambda: cluster_regeneration.version_history(run_id, settings))


@app.post("/cluster/{run_id}/analyze")
def analyze_saved_map(run_id: str, req: AnalyzeMapRequest, settings: Settings = Depends(get_settings)):
    return _version_response(run_id, lambda: analyze_cluster_map(run_id, req.expected_version_token, settings, req.request_id))


@app.get("/cluster/{run_id}")
def saved_cluster_preview(run_id: str, settings: Settings = Depends(get_settings)) -> Any:
    if not re.fullmatch(RUN_ID_PATTERN, run_id):
        return _error(None, 422, "InvalidInput", "Invalid run ID")
    path = Path(settings.logs_dir) / f"cluster-map-{run_id}.json"
    if not path.exists():
        return _error(run_id, 404, "NotFound", "No cluster preview saved for this run")
    try:
        result = ClusterMapResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        return _error(run_id, 500, type(error).__name__, str(error))
    return {"run_id": run_id, "status": "success", "output": result.model_dump(mode="json")}


from litagent.map_routes import install_map_routes  # noqa: E402

install_map_routes(app, get_settings, _version_response)
