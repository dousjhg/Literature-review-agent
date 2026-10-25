"""Blind, provenance-aware human pair review. No model-produced labels become human gold."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from litagent.cluster_audit import corpus_fingerprint, duplicate_components, partition_indices
from litagent.schemas import Cluster, Evidence, Paper
from litagent.similarity import partition_metrics


class PairReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_ids: tuple[str, str]
    judgment: Literal["same_subtopic", "different_subtopic", "uncertain"] | None = None
    provenance: Literal["pending", "human", "model"] = "pending"
    reviewer: str = ""
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


def paper_splits(papers: list[Paper]) -> dict[str, str]:
    """Keep aliases of a study in one fold; these are future review folds, not unseen data."""
    return {pid: ("development" if int(hashlib.sha256(group[0].encode()).hexdigest(), 16) % 2 == 0 else "holdout")
            for group in duplicate_components(papers) for pid in group}


def prepare_review(papers: list[Paper], clusters: list[Cluster], similarity: np.ndarray,
                   per_stratum: int, split: str = "development") -> tuple[dict, dict]:
    if per_stratum < 1 or split not in {"development", "holdout"}:
        raise ValueError("Invalid review sampling configuration")
    groups = partition_indices(papers, clusters)
    partition_metrics(similarity, groups)  # validate matrix and partition before sampling
    if similarity.shape != (len(papers), len(papers)):
        raise ValueError("Similarity must match input papers")
    membership = {i: g for g, members in enumerate(groups) for i in members}
    splits = paper_splits(papers)
    eligible = sorted((i for i, p in enumerate(papers) if splits[p.id] == split), key=lambda i: papers[i].id)
    pairs = list(combinations(eligible, 2))
    key = lambda pair: tuple(papers[i].id for i in pair)
    strata = {
        "weak_within": sorted((p for p in pairs if membership[p[0]] == membership[p[1]]),
                              key=lambda p: (similarity[p], key(p))),
        "strong_between": sorted((p for p in pairs if membership[p[0]] != membership[p[1]]),
                                 key=lambda p: (-similarity[p], key(p))),
        "background": sorted(pairs, key=lambda p: hashlib.sha256("|".join(key(p)).encode()).hexdigest()),
    }
    selected = {}
    for stratum, ranked in strata.items():
        used = 0
        for pair in ranked:
            if pair not in selected:
                selected[pair] = stratum
                used += 1
                if used == per_stratum:
                    break
    # Blind ordering and no predictions/stratum names in the annotation document.
    ordered = sorted(selected, key=lambda p: hashlib.sha256(("blind|" + "|".join(key(p))).encode()).hexdigest())
    pack = {"version": 1, "corpus_fingerprint": corpus_fingerprint(papers), "split": split,
            "protocol": "Same research subtopic: shared concrete task/method/evaluation setting, not just a broad field. "
                        "Use uncertain when metadata cannot decide. Human decisions need a reason and exact evidence "
                        "from both papers. Diagnostic sample; not representative population accuracy. "
                        "These saved papers have already been inspected; holdout is not an untouched final test set.",
            "papers": [p.model_dump(include={"id", "title", "abstract", "full_text"}) for p in
                       sorted(papers, key=lambda p: p.id) if any(p.id in key(pair) for pair in ordered)],
            "reviews": [PairReview(paper_ids=key(pair)).model_dump() for pair in ordered]}
    manifest = {"corpus_fingerprint": corpus_fingerprint(papers), "split": split,
                "selection": [{"paper_ids": key(pair), "stratum": selected[pair],
                               "similarity": float(similarity[pair]),
                               "same_cluster": membership[pair[0]] == membership[pair[1]]} for pair in ordered]}
    return pack, manifest


def score_reviews(pack: dict, papers: list[Paper], clusters: list[Cluster]) -> dict:
    if pack.get("version") != 1 or pack.get("corpus_fingerprint") != corpus_fingerprint(papers):
        raise ValueError("Review corpus fingerprint/version does not match the original paper sources")
    if pack.get("split") not in {"development", "holdout"}:
        raise ValueError("Unknown review split")
    partition_indices(papers, clusters)
    membership = {pid: c.id for c in clusters for pid in c.paper_ids}
    sources = {p.id: (p.title, p.abstract or "", p.full_text or "") for p in papers}
    splits = paper_splits(papers)
    seen = set()
    tp = fp = fn = tn = uncertain = excluded = 0
    for row in pack["reviews"]:
        review = PairReview.model_validate(row)
        a, b = review.paper_ids
        pair = tuple(sorted((a, b)))
        if a == b or a not in sources or b not in sources or pair in seen:
            raise ValueError("Review pairs must be unique, distinct and reference known papers")
        if splits[a] != pack["split"] or splits[b] != pack["split"]:
            raise ValueError("Review pair leaks papers across development/holdout folds")
        seen.add(pair)
        if review.provenance != "human" or review.judgment is None:
            excluded += 1
            continue
        if not review.reviewer.strip() or not review.rationale.strip():
            raise ValueError("Human judgments require reviewer attribution and rationale")
        if review.judgment == "uncertain":
            uncertain += 1
            continue
        evidence_ids = set()
        for evidence in review.evidence:
            if (evidence.paper_id not in pair or not evidence.quote.strip()
                    or not any(evidence.quote in source for source in sources[evidence.paper_id])):
                raise ValueError("Human pair evidence must be verbatim from the referenced original paper")
            evidence_ids.add(evidence.paper_id)
        if evidence_ids != set(pair):
            raise ValueError("Decisive human judgments require evidence from both papers")
        predicted = membership[a] == membership[b]
        positive = review.judgment == "same_subtopic"
        tp += int(predicted and positive)
        fp += int(predicted and not positive)
        fn += int(not predicted and positive)
        tn += int(not predicted and not positive)
    scored = tp + fp + fn + tn
    return {"status": "scored_diagnostic_pairs" if scored else "awaiting_human_labels",
            "scored": scored, "uncertain": uncertain, "pending_or_model_excluded": excluded,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pair_precision": tp / (tp + fp) if tp + fp else None,
            "pair_recall": tp / (tp + fn) if tp + fn else None,
            "pair_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "accuracy": (tp + tn) / scored if scored else None,
            "interpretation": "Co-membership on reviewed diagnostic pairs only; model/pending judgments excluded. "
                              "An evidence substring check does not verify that the judgment is semantically correct."}
