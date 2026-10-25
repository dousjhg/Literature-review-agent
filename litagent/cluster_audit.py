"""Deterministic cluster diagnostics; none of these checks are semantic ground truth."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations

import numpy as np

from litagent.schemas import Cluster, Paper
from litagent.similarity import partition_metrics


def normalized_title(title: str) -> str:
    return " ".join(re.findall(r"\w+", title.casefold()))


def duplicate_components(papers: list[Paper]) -> list[list[str]]:
    """Potential duplicate records by exact normalized title or DOI, never deleted here."""
    parent = {p.id: p.id for p in papers}

    def root(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    seen = {}
    for p in sorted(papers, key=lambda p: p.id):
        doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", (p.doi or "").strip().casefold())
        keys = [("title", normalized_title(p.title)), ("doi", doi)]
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                parent[root(p.id)] = root(seen[key])
            seen[key] = p.id
    groups = {}
    for p in sorted(papers, key=lambda p: p.id):
        groups.setdefault(root(p.id), []).append(p.id)
    return sorted(groups.values())


def corpus_fingerprint(papers: list[Paper]) -> str:
    sources = [{k: getattr(p, k) for k in ("id", "title", "abstract", "full_text", "doi")}
               for p in sorted(papers, key=lambda p: p.id)]
    return hashlib.sha256(json.dumps(sources, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def partition_indices(papers: list[Paper], clusters: list[Cluster]) -> list[list[int]]:
    ids = {p.id: i for i, p in enumerate(papers)}
    members = [pid for c in clusters for pid in c.paper_ids]
    if (len(ids) != len(papers) or len({c.id for c in clusters}) != len(clusters)
            or len(members) != len(ids) or set(members) != set(ids)
            or any(not c.paper_ids for c in clusters)):
        raise ValueError("Clusters must partition unique input papers exactly once with unique cluster ids")
    return [[ids[pid] for pid in c.paper_ids] for c in clusters]


def cluster_diagnostics(papers: list[Paper], clusters: list[Cluster], similarity: np.ndarray,
                        threshold: float) -> dict:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid diagnostic threshold")
    groups = partition_indices(papers, clusters)
    metrics = partition_metrics(similarity, groups)
    if similarity.shape != (len(papers), len(papers)):
        raise ValueError("Similarity must match input papers")
    within = []
    for cluster, group in zip(clusters, groups):
        pairs = list(combinations(group, 2))
        if pairs:
            a, b = min(pairs, key=lambda pair: (similarity[pair], papers[pair[0]].id, papers[pair[1]].id))
            within.append({"cluster_id": cluster.id, "paper_ids": [papers[a].id, papers[b].id],
                           "similarity": float(similarity[a, b])})
    boundaries = []
    for left, right in combinations(range(len(groups)), 2):
        block = similarity[np.ix_(groups[left], groups[right])]
        a, b = np.unravel_index(np.argmax(block), block.shape)
        boundaries.append({"cluster_ids": [clusters[left].id, clusters[right].id],
                           "closest_paper_ids": [papers[groups[left][a]].id, papers[groups[right][b]].id],
                           "min_similarity": float(block.min()), "max_similarity": float(block.max()),
                           "merge_meets_threshold": bool(block.min() >= threshold)})
    label_groups = {}
    for c in clusters:
        label_groups.setdefault(normalized_title(c.label), []).append(c.id)
    return {"corpus_fingerprint": corpus_fingerprint(papers), "threshold": threshold,
            "metrics": metrics, "weakest_within_pairs": within,
            "boundaries": sorted(boundaries, key=lambda row: (-row["max_similarity"], row["cluster_ids"])),
            "duplicate_labels": [ids for ids in label_groups.values() if len(ids) > 1],
            "possible_duplicate_records": [g for g in duplicate_components(papers) if len(g) > 1],
            "missing_abstract_ids": [p.id for p in papers if not (p.abstract or "").strip()],
            "interpretation": "Numerical diagnostics only; high similarity and exact quote matches do not prove semantic coherence."}
