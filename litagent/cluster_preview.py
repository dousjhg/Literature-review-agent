"""A noninteractive, single-pass clustering checkpoint independent of later TODO agents."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from litagent.agents import clusterer, grader
from litagent.cluster_audit import partition_indices
from litagent.cluster_state import ClusteringArtifacts, VersionError, path_for, save_version, set_head, version_lock
from litagent.config import EFFORT_PROFILES, Effort, Settings
from litagent.integrations.llm import TokenBudgetExceeded, TokenMeter
from litagent.integrations.meterless import ResearchWorld
from litagent.schemas import ClusterMapResult, Paper, PaperClusterDetail, QuestionCluster
from litagent.question_clusters import attach_alternatives, cluster_id, label_side_groups
from litagent.trace import Trace, new_run_id


class _PreviewTimeLimit(Exception):
    pass


def run_cluster_map(papers: list[Paper], question: str, effort: Effort, settings: Settings,
                    run_id: str | None = None) -> ClusterMapResult:
    question = question.strip()
    run_id = run_id or new_run_id()
    with version_lock(settings, run_id):
        if any(path_for(settings, prefix, run_id, suffix).exists() for prefix, suffix in (
                ("cluster-map", "json"), ("cluster-state", "json"), ("world", "json"), ("trace", "jsonl"))):
            raise VersionError("Run ID already exists; create a new immutable version")
        return _run_cluster_map(papers, question, effort, settings, run_id)


def _run_cluster_map(papers: list[Paper], question: str, effort: Effort, settings: Settings,
                     run_id: str) -> ClusterMapResult:
    if not question.strip() or not papers or len({p.id for p in papers}) != len(papers):
        raise ValueError("Clustering requires a question and unique paper IDs")
    profile = EFFORT_PROFILES[effort]
    if len(papers) > profile.max_papers:
        raise ValueError(f"{effort.value} permits at most {profile.max_papers} papers")
    started = time.monotonic()
    with Trace(settings.logs_dir, run_id) as trace, TokenMeter(profile.token_budget) as meter:
        world = ResearchWorld(trace.run_id, settings.logs_dir)
        world.start_attempt(1, question)
        world.add_papers(papers)
        result = ClusterMapResult(run_id=trace.run_id, question=question,
                                  main_similarity_threshold=settings.main_similarity_threshold,
                                  sample_mode=settings.sample_mode, map_mode=settings.cluster_map_mode,
                                  embedding_profile=settings.embedding_profile, use_full_text=profile.use_full_text,
                                  side_linkage=settings.side_linkage, side_similarity_threshold=settings.side_similarity_threshold,
                                  similarity_settings={k: getattr(settings, k) for k in (
                                      "cluster_dense_weight", "cluster_lexical_title_weight", "cluster_lexical_max_df",
                                      "cluster_lexical_max_features", "cluster_lexical_ngram_range")})
        result.similarity_settings["cluster_lexical_ngram_range"] = list(settings.cluster_lexical_ngram_range)
        artifacts = ClusteringArtifacts()
        try:
            if time.monotonic() - started > settings.run_timeout_s:
                raise _PreviewTimeLimit
            result.clusters = clusterer.cluster(papers, effort, settings, question=question, artifacts=artifacts)
            if settings.sample_mode and settings.cluster_map_mode == "question":
                # Synthetic, explicitly marked sample scores exercise the same API with no models.
                artifacts.paper_vectors = np.tile([1., 0.], (len(papers), 1))
                artifacts.question_vector = np.array([0., 1.])
                artifacts.similarity = np.ones((len(papers), len(papers)))
                positions = {p.id: i for i, p in enumerate(papers)}
                ids = [cluster_id(c.paper_ids) for c in result.clusters]
                if set(ids) & set(positions):
                    raise VersionError("Paper IDs conflict with generated cluster IDs", 422)
                groups = [[positions[pid] for pid in c.paper_ids] for c in result.clusters]
                labels = label_side_groups(papers, [p.title for p in papers], groups, artifacts.similarity, ids, settings)
                result.clusters = [QuestionCluster(**{**c.model_dump(), **labels[cid].model_dump(), "id": cid}, kind="side", question=question,
                    paper_details=[PaperClusterDetail(paper_id=pid, question_similarity=0., relevance="uncertain",
                        reason="Sample placeholder: relevance was not assessed.") for pid in c.paper_ids]) for c, cid in zip(result.clusters, ids)]
                attach_alternatives(result.clusters, papers, artifacts.similarity, settings)
            partition_indices(papers, result.clusters)
            mains = [c for c in result.clusters if isinstance(c, QuestionCluster) and c.kind == "main"]
            if len(mains) > 1:
                raise ValueError("Question map may contain at most one main cluster")
            result.main_cluster_id = mains[0].id if mains else None
            world.write_clusters(result.clusters)
            if settings.sample_mode:
                result.warnings.append("Sample mode: placeholder topics, synthetic similarity=1 and question scores=0; no relevance assessment.")
            elif settings.cluster_map_mode == "question" and not mains:
                result.warnings.append("No paper passed both the question-similarity gate and source-backed direct-relevance assessment.")
            if time.monotonic() - started > settings.run_timeout_s:
                raise _PreviewTimeLimit
            result.grade = grader.grade("clusters", result.clusters, papers, settings)
            world.record_grade("clusters", result.grade)
            ok = grader.passed(result.grade, settings)
            trace.log("grader", "grade question map", f"{result.grade.total}/9 {'pass' if ok else 'fail'}",
                      result=result.grade, why=result.grade.fix or result.grade.reasoning)
            if not ok:
                result.warnings.append("Cluster map did not pass grading; retained for review without automatic regeneration.")
        except (TokenBudgetExceeded, _PreviewTimeLimit) as error:
            result.status = "partial"
            reason = str(error) if isinstance(error, TokenBudgetExceeded) else "Cluster-preview time limit reached"
            result.warnings.append(reason)
            trace.log("clusterer", "preview limit", "partial", why=reason)
        except Exception as error:
            trace.error("clusterer", "cluster preview", error, next_step="return error")
            raise
        result.paper_map = world.to_graph_json()
        result.usage = meter.usage
        if (result.clusters and all(isinstance(c, QuestionCluster) for c in result.clusters)
                and artifacts.similarity is not None):
            save_version(result, papers, effort, artifacts, settings)
            set_head(result, settings)
            return result
        path = Path(settings.logs_dir) / f"cluster-map-{trace.run_id}.json"
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return result
