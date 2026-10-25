"""Content similarity and diameter-bounded clustering for small literature-review batches.

Similarity is not query relevance, citation popularity, or a calibrated probability.
The legacy connected-components algorithm remains in agents/clusterer.py for comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from litagent.config import CLUSTER_MAX_PAPER_FRACTION, CLUSTER_MAX_SINGLETON_FRACTION, Settings
from litagent.schemas import Paper


@dataclass(frozen=True)
class SimilarityMatrices:
    dense: np.ndarray
    lexical: np.ndarray
    combined: np.ndarray
    lexical_missing: int


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.shape[1] == 0 or (norms <= 0).any():
        raise ValueError("Embeddings must have finite non-zero vectors")
    normalized = vectors / norms
    result = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def paper_similarity(papers: list[Paper], vectors: np.ndarray, settings: Settings) -> SimilarityMatrices:
    """Semantic cosine plus IDF-weighted content agreement; no popularity/citation features."""
    if not 0 <= settings.cluster_dense_weight <= 1 or not 0 <= settings.cluster_lexical_title_weight <= 1:
        raise ValueError("Similarity weights must be between 0 and 1")
    if not 0 < settings.cluster_lexical_max_df <= 1 or settings.cluster_lexical_max_features < 1:
        raise ValueError("Invalid lexical feature limits")
    dense = cosine_matrix(vectors)
    if dense.shape != (len(papers), len(papers)):
        raise ValueError("Embeddings must contain one vector per paper")
    lexical = np.zeros_like(dense)
    available = np.zeros(len(papers), dtype=bool)
    if papers and settings.cluster_dense_weight < 1:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        titles = [p.title for p in papers]
        bodies = [p.abstract or p.title for p in papers]
        vectorizer = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, norm="l2",
            max_df=settings.cluster_lexical_max_df, max_features=settings.cluster_lexical_max_features,
            ngram_range=settings.cluster_lexical_ngram_range,
            token_pattern=r"(?u)\b[^\W\d_][\w-]+\b",
        )
        try:
            vectorizer.fit([f"{t} {b}" for t, b in zip(titles, bodies)])
        except ValueError as e:
            if not any(term in str(e).lower() for term in ("empty vocabulary", "no terms remain", "after pruning")):
                raise
        else:
            weight = settings.cluster_lexical_title_weight
            features = normalize(weight * vectorizer.transform(titles) + (1 - weight) * vectorizer.transform(bodies))
            available = np.asarray(features.getnnz(axis=1) > 0).ravel()
            lexical = np.clip((features @ features.T).toarray(), 0.0, 1.0)
    # A missing lexical vector is unknown, not evidence of dissimilarity.
    usable = available[:, None] & available[None, :]
    weight = np.where(usable, settings.cluster_dense_weight, 1.0)
    combined = weight * dense + (1 - weight) * lexical
    combined = np.clip((combined + combined.T) / 2, -1.0, 1.0)
    np.fill_diagonal(combined, 1.0)
    return SimilarityMatrices(dense, lexical, combined,
                              int((~available).sum()) if settings.cluster_dense_weight < 1 else 0)


def _validated_similarity(similarity: np.ndarray) -> np.ndarray:
    matrix = np.asarray(similarity, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T) or (matrix < -1).any() or (matrix > 1).any()):
        raise ValueError("Similarity must be a finite symmetric square matrix in [-1, 1]")
    return matrix


def complete_linkage_groups(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """Every pair inside each returned group meets the threshold, including boundary ties."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    matrix = _validated_similarity(similarity)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1")
    if len(matrix) < 2:
        return [[0]] if len(matrix) else []
    distances = np.maximum(0, 1 - matrix)
    np.fill_diagonal(distances, 0)
    tree = linkage(squareform(distances, checks=False), method="complete")
    labels = fcluster(tree, t=1 - threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def partition_metrics(similarity: np.ndarray, groups: list[list[int]]) -> dict:
    """Internal diagnostics only; silhouette is not a ground-truth quality assessment."""
    from sklearn.metrics import silhouette_score

    matrix = _validated_similarity(similarity)
    n = len(matrix)
    if sorted(i for group in groups for i in group) != list(range(n)) or any(not g for g in groups):
        raise ValueError("Groups must partition all papers exactly once")
    sizes = [len(group) for group in groups]
    within = [matrix[i, j] for group in groups for a, i in enumerate(group) for j in group[a + 1:]]
    silhouette = None
    if 1 < len(groups) < n:
        labels = np.empty(n, dtype=int)
        for label, group in enumerate(groups):
            labels[group] = label
        distance = np.maximum(0, 1 - matrix)
        np.fill_diagonal(distance, 0)
        silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
    singles = sum(size == 1 for size in sizes)
    return {"clusters": len(groups), "sizes": sizes,
            "largest_fraction": max(sizes, default=0) / max(n, 1),
            "singleton_paper_fraction": singles / max(n, 1),
            "coverage": 1 - singles / n if n else 0,
            "min_within_similarity": float(min(within)) if within else None,
            "mean_within_similarity": float(np.mean(within)) if within else None,
            "silhouette": silhouette}


def select_partition(similarity: np.ndarray, settings: Settings, fix: str | None = None) -> tuple[list[list[int]], float, list[dict]]:
    """Unsupervised cut selection, with an explicit floor and bounded grader-driven retries."""
    if settings.cluster_threshold_mode not in {"fixed", "adaptive"}:
        raise ValueError("cluster_threshold_mode must be fixed or adaptive")
    base = settings.cluster_similarity_threshold
    candidates = sorted(set(settings.cluster_threshold_candidates) | {base}) if settings.cluster_threshold_mode == "adaptive" else [base]
    if not candidates or any(not np.isfinite(t) or not 0 <= t <= 1 for t in candidates):
        raise ValueError("Threshold candidates must be finite values between 0 and 1")
    evaluated = []
    partitions = {}
    for threshold in candidates:
        groups = complete_linkage_groups(similarity, threshold)
        partitions[threshold] = groups
        metrics = partition_metrics(similarity, groups)
        # Degenerate all-in-one/all-singleton partitions are not evidence of a good cut.
        objective = metrics["silhouette"] * metrics["coverage"] if metrics["silhouette"] is not None else None
        evaluated.append({"threshold": threshold, "objective": objective, **metrics})
    valid = [row for row in evaluated if row["objective"] is not None]
    chosen = max(valid, key=lambda row: (row["objective"], row["threshold"])) if valid else next(
        row for row in evaluated if row["threshold"] == base)
    if fix and len(similarity):
        old = partitions[chosen["threshold"]]
        if max(map(len, old), default=0) > 1 and chosen["largest_fraction"] > CLUSTER_MAX_PAPER_FRACTION:
            target = chosen["threshold"] + settings.cluster_redo_threshold_step
        elif sum(len(g) == 1 for g in old) / max(len(old), 1) > CLUSTER_MAX_SINGLETON_FRACTION:
            target = chosen["threshold"] - settings.cluster_redo_threshold_step
        else:
            target = chosen["threshold"]
        # Never relax below the configured candidate floor merely to pass a size check.
        threshold = float(np.clip(target, min(candidates) if settings.cluster_threshold_mode == "adaptive" else 0,
                                  max(candidates) if settings.cluster_threshold_mode == "adaptive" else 1))
        if threshold != chosen["threshold"]:
            groups = complete_linkage_groups(similarity, threshold)
            metrics = partition_metrics(similarity, groups)
            evaluated.append({"threshold": threshold, "objective": None, "grader_redo": True, **metrics})
            return groups, threshold, evaluated
    return partitions[chosen["threshold"]], chosen["threshold"], evaluated


def representative_indices(similarity: np.ndarray, group: list[int], limit: int, relevance_weight: float) -> list[int]:
    """MMR chooses central but non-redundant label examples; never changes membership."""
    if limit < 1 or not 0 <= relevance_weight <= 1:
        raise ValueError("Invalid representative limit or MMR weight")
    if len(group) <= limit:
        return list(group)
    block = similarity[np.ix_(group, group)]
    centrality = (block.sum(axis=1) - np.diag(block)) / (len(group) - 1)
    selected = [int(np.argmax(centrality))]
    while len(selected) < limit:
        remaining = [i for i in range(len(group)) if i not in selected]
        best = max(remaining, key=lambda i: (
            relevance_weight * centrality[i] - (1 - relevance_weight) * max(block[i, j] for j in selected), -i))
        selected.append(best)
    return [group[i] for i in selected]
