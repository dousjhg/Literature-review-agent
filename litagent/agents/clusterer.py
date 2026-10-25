"""3. Clusterer agent. Spec: docs/agents/clusterer.md

Embeds papers, groups those with cosine similarity >= threshold, and labels each group.
Normal / Medium use metadata (title + abstract); High / Ultra use full text.
"""

from __future__ import annotations

import json
import numpy as np

from litagent.config import EFFORT_PROFILES, Effort, Settings
from litagent.cluster_labels import contrastive_labels
from litagent.agents.grader import check_cluster_sizes
from litagent.integrations.embeddings import embed_documents
from litagent.integrations.llm import complete_json, load_prompt
from litagent.schemas import Cluster, ClusterLabel, Paper
from litagent.semantic_groups import semantic_clusters
from litagent.similarity import cosine_matrix, paper_similarity, partition_metrics, representative_indices, select_partition
from litagent.trace import current_trace


def cluster(papers: list[Paper], effort: Effort, settings: Settings, fix: str | None = None,
            *, defer_labels_on_size_failure: bool = False) -> list[Cluster]:
    """`fix` is the grading agent's one suggested fix when the previous attempt failed."""
    if settings.sample_mode:
        return _fake_clusters(papers)
    if not papers:
        return []
    if len({p.id for p in papers}) != len(papers):
        raise ValueError("Clusterer requires unique paper ids")
    if not 0 <= settings.cluster_similarity_threshold <= 1:
        raise ValueError("cluster_similarity_threshold must be between 0 and 1")
    if settings.cluster_label_strategy not in {"independent", "contrastive"}:
        raise ValueError("cluster_label_strategy must be independent or contrastive")
    if settings.cluster_grouping_strategy not in {"complete", "semantic"}:
        raise ValueError("cluster_grouping_strategy must be complete or semantic")
    if settings.cluster_label_text_chars < 1 or settings.cluster_label_max_papers < 1 or settings.cluster_redo_threshold_step < 0:
        raise ValueError("Cluster label limit must be positive and threshold step non-negative")

    use_full_text = EFFORT_PROFILES[effort].use_full_text
    texts = [_text_for(p, use_full_text) for p in papers]
    embeddings = np.asarray(embed(texts, settings), dtype=float)
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(papers) or embeddings.shape[1] == 0
            or not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings, axis=1) == 0).any()):
        raise ValueError("Embeddings must contain one finite non-zero vector per paper")
    scores = paper_similarity(papers, embeddings, settings)
    if settings.cluster_grouping_strategy == "semantic":
        return semantic_clusters(papers, texts, scores.combined, settings, fix)
    # A stable paper-id order makes linkage ties independent of API result order.
    order = sorted(range(len(papers)), key=lambda i: papers[i].id)
    canonical = scores.combined[np.ix_(order, order)]
    canonical_groups, threshold, candidates = select_partition(canonical, settings, fix)
    groups = [[order[i] for i in group] for group in canonical_groups]
    if trace := current_trace():
        trace.log("clusterer", "group papers", f"{len(groups)} groups", next_step="label groups",
                  threshold=threshold, sizes={"counts": [len(g) for g in groups]},
                  metrics=partition_metrics(scores.combined, groups),
                  dense_metrics=partition_metrics(scores.dense, groups),
                  calibration={"candidates": candidates}, lexical_missing=scores.lexical_missing,
                  why=f"Complete linkage on semantic/TF-IDF similarity; dense weight={settings.cluster_dense_weight}; "
                      f"mode={settings.cluster_threshold_mode}; singletons retained; grader fix: {fix or 'none'}")
        if use_full_text and any(not p.full_text for p in papers):
            trace.log("clusterer", "select text", "metadata fallback", why=
                      "Some High/Ultra papers have no full text; PDF retrieval and Markovian reading are pending")
    # In the graded pipeline the first size failure is guaranteed to trigger a redo.
    # Return its partition for that check without spending tokens on discarded names.
    if defer_labels_on_size_failure and fix is None and settings.max_redo_per_stage > 0:
        pending = [Cluster(id=f"C{i + 1}", label="Pending size review", paper_ids=[papers[j].id for j in group],
                           rationale="Label deferred until the deterministic size check.") for i, group in enumerate(groups)]
        if problem := check_cluster_sizes(pending, len(papers)):
            if trace := current_trace():
                trace.log("clusterer", "defer labels", "size check will retry", why=problem)
            return pending
    assessed = (contrastive_labels(papers, texts, groups, scores.combined, settings, fix)
                if settings.cluster_label_strategy == "contrastive" else {})
    clusters = []
    for index, group in enumerate(groups, 1):
        members = [papers[i] for i in group]
        if len(members) == 1:
            label = ClusterLabel(label=f"Unclustered: {members[0].title}",
                                 rationale="No compatible group met the pairwise similarity threshold; retained as an outlier.")
        elif settings.cluster_label_strategy == "contrastive":
            assessment = assessed[f"C{index}"]
            label = ClusterLabel(label=assessment.label,
                                 rationale=f"[{assessment.coherence}] {assessment.rationale} {assessment.contrast}")
        else:
            representatives = representative_indices(scores.combined, group, settings.cluster_label_max_papers,
                                                       settings.cluster_representative_relevance_weight)
            payload = {
                "papers": [{"id": papers[i].id, "title": papers[i].title,
                            "text": texts[i][:settings.cluster_label_text_chars]} for i in representatives],
                "total_papers": len(group),
                "group_paper_ids": [p.id for p in members],
                "fix": fix,
            }
            if trace := current_trace():
                trace.log("clusterer", "choose label examples", f"{len(representatives)} of {len(group)} papers",
                          result={"paper_ids": [papers[i].id for i in representatives]},
                          why="MMR balances group centrality and non-redundant evidence; group membership is unchanged")
            label = complete_json(load_prompt("cluster_labeler"), json.dumps(payload, ensure_ascii=False),
                                  ClusterLabel, settings=settings, agent_name="clusterer")
        clusters.append(Cluster(id=f"C{index}", label=label.label,
                                paper_ids=[p.id for p in members], rationale=label.rationale))
    return clusters


def embed(texts: list[str], settings: Settings | None = None) -> np.ndarray:
    return embed_documents(texts, settings or Settings())


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return cosine_matrix(embeddings)


def group_by_threshold(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """Legacy connected-components baseline, retained only for comparison experiments."""
    n = similarity.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _text_for(paper: Paper, use_full_text: bool) -> str:
    if use_full_text and paper.full_text:
        return paper.full_text  # TODO: long texts may need Meterless read_in_chunks first
    return f"{paper.title}. {paper.abstract or ''}"


def _fake_clusters(papers: list[Paper]) -> list[Cluster]:
    """Sample mode: split papers round-robin into 3 placeholder groups (no embeddings, no LLM)."""
    k = 3
    return [
        Cluster(
            id=f"C{i + 1}", label=f"Sample theme {i + 1}", paper_ids=[p.id for p in papers[i::k]],
            rationale="Sample mode: placeholder grouping, not real clustering",
        )
        for i in range(k)
    ]
