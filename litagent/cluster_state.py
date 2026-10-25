"""Local, immutable clustering snapshots. A ready manifest is written last.

Arrays are loaded without pickle; source text and scoring policy stay bound to a lineage.
Locks use nonblocking flock so a competing API call returns a conflict immediately.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from litagent.cluster_audit import partition_indices
from litagent.cluster_labels import contrastive_payload, label_contexts
from litagent.config import EMBEDDING_PROFILES, Effort, Settings
from litagent.integrations.llm import load_prompt
from litagent.schemas import (ClusterAssessmentProposals, ClusterLabel, ClusterMapResult,
                             FindingCacheRecord, FindingDependency, Paper, QuestionCluster)


class VersionError(ValueError):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


@dataclass
class ClusteringArtifacts:
    paper_vectors: np.ndarray | None = None
    question_vector: np.ndarray | None = None
    similarity: np.ndarray | None = None


class ClusterState(BaseModel):
    format_version: int = 1
    run_id: str
    root_run_id: str
    effort: Effort
    papers: list[Paper]
    frozen_settings: dict
    encoder: dict
    label_keys: dict[str, str]
    finding_dependencies: dict[str, FindingDependency] = Field(default_factory=dict)
    finding_cache: dict[str, FindingCacheRecord] = Field(default_factory=dict)
    array_hash: str
    result_hash: str


FROZEN_SETTINGS = (
    "sample_mode", "cluster_map_mode", "main_similarity_threshold", "side_linkage", "side_similarity_threshold",
    "related_similarity_threshold", "related_cluster_limit", "related_boundary_margin", "paper_keyword_limit",
    "embedding_profile", "embedding_chunk_tokens", "embedding_max_chunks_per_paper",
    "cluster_dense_weight", "cluster_lexical_title_weight", "cluster_lexical_max_df",
    "cluster_lexical_max_features", "cluster_lexical_ngram_range",
    "regeneration_score_tolerance",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_for(settings: Settings, prefix: str, ident: str, suffix: str = "json") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", ident):
        raise VersionError("Invalid version identifier", 422)
    directory = Path(settings.logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}-{ident}.{suffix}"


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def version_lock(settings: Settings, root: str):
    with path_for(settings, "cluster-lock", root, "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise VersionError("Another operation is updating this cluster lineage; retry later") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_result(run_id: str, settings: Settings) -> ClusterMapResult:
    path = path_for(settings, "cluster-map", run_id)
    if not path.exists():
        raise VersionError("No cluster map saved for this run", 404)
    return ClusterMapResult.model_validate_json(path.read_text(encoding="utf-8"))


def side_inputs(papers: list[Paper], clusters: list[QuestionCluster], use_full_text: bool):
    from litagent.agents.clusterer import _text_for
    indices = {p.id: i for i, p in enumerate(papers)}
    sides = [c for c in clusters if c.kind == "side"]
    return ([_text_for(p, use_full_text) for p in papers],
            [[indices[pid] for pid in c.paper_ids] for c in sides], [c.id for c in sides])


def label_policy(settings: Settings) -> dict:
    if settings.cluster_label_strategy not in {"contrastive", "independent"}:
        raise VersionError("cluster_label_strategy must be contrastive or independent", 422)
    keys = [k for k in vars(settings) if k.startswith(("cluster_label_", "cluster_neighbor_", "cluster_evidence_"))]
    keys += ["cluster_representative_relevance_weight", "sample_mode", "llm_model"]
    return {"settings": {k: getattr(settings, k) for k in keys},
            "contrastive_prompt": load_prompt("cluster_labeler_contrastive"),
            "independent_prompt": load_prompt("cluster_labeler"),
            "contrastive_schema": ClusterAssessmentProposals.model_json_schema(),
            "independent_schema": ClusterLabel.model_json_schema(),
            "implementation": {name: file_hash(Path(__file__).with_name(name))
                               for name in ("cluster_state.py", "cluster_labels.py", "question_clusters.py")}}


def label_keys(papers: list[Paper], clusters: list[QuestionCluster], similarity: np.ndarray,
               use_full_text: bool, settings: Settings) -> dict[str, str]:
    """Batch keys include every target, representative and retrieved neighboring context."""
    from litagent.question_clusters import independent_label_payload
    texts, groups, ids = side_inputs(papers, clusters, use_full_text)
    keys, bodies = {}, []
    policy = label_policy(settings)
    for context in label_contexts(papers, texts, groups, similarity, settings, ids):
        cid = context["cluster_id"]
        if settings.sample_mode or context["total_papers"] == 1:
            keys[cid] = digest({"implementation": policy["implementation"], "sample": settings.sample_mode,
                                "members": context["members"]})
        elif settings.cluster_label_strategy == "independent":
            group = groups[ids.index(cid)]
            keys[cid] = digest({"policy": policy, "payload": independent_label_payload(papers, texts, group, settings)})
        elif not any(p["has_body"] for p in context["papers"]):
            keys[cid] = digest({"implementation": policy["implementation"], "kind": "title-only", "members": context["members"]})
        else:
            bodies.append(context)
    for start in range(0, len(bodies), settings.cluster_label_batch_size):
        batch = bodies[start:start + settings.cluster_label_batch_size]
        key = digest({"policy": policy, "payload": contrastive_payload(batch, settings)})
        keys.update({c["cluster_id"]: key for c in batch})
    return keys


def effective_settings(state: ClusterState, settings: Settings) -> Settings:
    from dataclasses import replace
    values = dict(state.frozen_settings)
    values["cluster_lexical_ngram_range"] = tuple(values["cluster_lexical_ngram_range"])
    if settings.sample_mode != values["sample_mode"]:
        raise VersionError("Sample and real versions cannot be mixed; select the original sample-mode setting")
    return replace(settings, **values)


def validate_artifacts(papers: list[Paper], artifacts: ClusteringArtifacts) -> None:
    v, q, s = artifacts.paper_vectors, artifacts.question_vector, artifacts.similarity
    n = len(papers)
    if (v is None or q is None or s is None or v.ndim != 2 or len(v) != n or not v.shape[1]
            or q.shape != (v.shape[1],) or s.shape != (n, n)
            or not all(np.isfinite(x).all() for x in (v, q, s))
            or (np.linalg.norm(v, axis=1) == 0).any() or np.linalg.norm(q) == 0
            or not np.allclose(s, s.T, atol=1e-8) or not np.allclose(np.diag(s), 1)
            or (s < -1 - 1e-8).any() or (s > 1 + 1e-8).any()):
        raise VersionError("Cached vectors or similarity matrix are incomplete or invalid")


def validate_snapshot(result: ClusterMapResult, papers: list[Paper], arrays: ClusteringArtifacts, tolerance: float) -> None:
    partition_indices(papers, result.clusters)
    if not result.clusters or not all(isinstance(c, QuestionCluster) for c in result.clusters):
        raise VersionError("Only main/side question maps support regeneration")
    mains = [c.id for c in result.clusters if c.kind == "main"]
    if len(mains) > 1 or result.main_cluster_id != (mains[0] if mains else None):
        raise VersionError("Snapshot main-cluster identity is inconsistent")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise VersionError("Regeneration score tolerance must be finite and between 0 and 1", 422)
    query = arrays.question_vector / np.linalg.norm(arrays.question_vector)
    vectors = arrays.paper_vectors / np.linalg.norm(arrays.paper_vectors, axis=1, keepdims=True)
    scores = np.clip(vectors @ query, -1, 1)
    indices = {p.id: i for i, p in enumerate(papers)}
    cluster_ids = {c.id for c in result.clusters}
    for cluster in result.clusters:
        QuestionCluster.model_validate(cluster.model_dump())
        if cluster.question != result.question:
            raise VersionError("Cluster question differs from its source map")
        for detail in cluster.paper_details:
            if abs(detail.question_similarity - scores[indices[detail.paper_id]]) > tolerance:
                raise VersionError("Cached query scores do not match the original paper-vector rows")
            if cluster.kind == "main" and detail.question_similarity < result.main_similarity_threshold:
                raise VersionError("Main member is below the original question gate")
            paper = papers[indices[detail.paper_id]]
            if any(not e.quote or not any(e.quote in source for source in (paper.abstract, paper.full_text) if source)
                   for e in detail.evidence):
                raise VersionError("Relevance evidence does not match an original source field")
            if (any(a.cluster_id not in cluster_ids for a in detail.also_related)
                    or detail.runner_up_cluster_id is not None and detail.runner_up_cluster_id not in cluster_ids - {cluster.id}):
                raise VersionError("Alternative references an absent or assigned cluster")


def save_version(result: ClusterMapResult, papers: list[Paper], effort: Effort, artifacts: ClusteringArtifacts,
                 settings: Settings, *, dependencies: dict[str, FindingDependency] | None = None,
                 keys: dict[str, str] | None = None,
                 finding_cache: dict[str, FindingCacheRecord] | None = None) -> ClusterState:
    if path_for(settings, "cluster-state", result.run_id).exists():
        raise VersionError("Cluster versions are immutable; use a new run ID")
    validate_artifacts(papers, artifacts)
    validate_snapshot(result, papers, artifacts, settings.regeneration_score_tolerance)
    result.root_run_id = result.root_run_id or result.run_id
    result.version_token = result.version_token or uuid.uuid4().hex
    array_path = path_for(settings, "cluster-arrays", result.run_id, "npz")
    tmp = array_path.with_name(array_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, paper_vectors=artifacts.paper_vectors,
                                question_vector=artifacts.question_vector, similarity=artifacts.similarity)
        tmp.replace(array_path)
    finally:
        tmp.unlink(missing_ok=True)
    payload = result.model_dump(mode="json")
    state = ClusterState(run_id=result.run_id, root_run_id=result.root_run_id, effort=effort, papers=papers,
        frozen_settings={k: getattr(settings, k) for k in FROZEN_SETTINGS},
        encoder={"kind": "synthetic_sample", "dimensions": artifacts.paper_vectors.shape[1]} if settings.sample_mode
                else EMBEDDING_PROFILES[settings.embedding_profile],
        label_keys=keys if keys is not None else label_keys(papers, result.clusters, artifacts.similarity,
                                                         result.use_full_text, settings),
        finding_dependencies=dependencies or {}, finding_cache=finding_cache or {},
        array_hash=file_hash(array_path), result_hash=digest(payload))
    atomic_json(path_for(settings, "cluster-map", result.run_id), payload)
    # The state file is the ready marker, bound to both result and numeric cache.
    body = state.model_dump(mode="json")
    atomic_json(path_for(settings, "cluster-state", result.run_id), {"checksum": digest(body), "state": body})
    return state


def load_version(run_id: str, settings: Settings) -> tuple[ClusterMapResult, ClusterState, ClusteringArtifacts]:
    result = load_result(run_id, settings)
    ready = path_for(settings, "cluster-state", run_id)
    if not ready.exists():
        raise VersionError("This checkpoint has no reusable vector cache. Create a new POST /cluster map to enable regeneration")
    try:
        envelope = json.loads(ready.read_text(encoding="utf-8"))
        if envelope["checksum"] != digest(envelope["state"]):
            raise ValueError("state checksum mismatch")
        state = ClusterState.model_validate(envelope["state"])
        arrays_path = path_for(settings, "cluster-arrays", run_id, "npz")
        if (state.run_id != run_id or result.run_id != run_id or state.root_run_id != result.root_run_id
                or not result.version_token or state.result_hash != digest(json.loads(
                    path_for(settings, "cluster-map", run_id).read_text(encoding="utf-8")))
                or state.array_hash != file_hash(arrays_path)):
            raise ValueError("snapshot checksum/identity mismatch")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            artifacts = ClusteringArtifacts(**{k: arrays[k].copy() for k in (
                "paper_vectors", "question_vector", "similarity")})
        validate_artifacts(state.papers, artifacts)
        validate_snapshot(result, state.papers, artifacts, state.frozen_settings["regeneration_score_tolerance"])
    except (OSError, ValueError, KeyError) as error:
        raise VersionError(f"Cluster cache integrity check failed: {error}") from error
    return result, state, artifacts


def head(root_id: str, settings: Settings) -> dict:
    path = path_for(settings, "cluster-head", root_id)
    if not path.exists():
        raise VersionError("No active version recorded for this lineage")
    return json.loads(path.read_text(encoding="utf-8"))


def set_head(result: ClusterMapResult, settings: Settings) -> None:
    atomic_json(path_for(settings, "cluster-head", result.root_run_id),
                {"root_run_id": result.root_run_id, "run_id": result.run_id, "version_token": result.version_token})
