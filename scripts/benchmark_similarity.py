"""Offline ablations on saved papers: representation, linkage and adaptive/hybrid scoring.

    python scripts/benchmark_similarity.py --profiles minilm semcse

Source-file topics are weak labels for broad topic separation, not expert subtopic labels.
All threshold selection is unsupervised; topic labels are used only in reporting metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent.agents.clusterer import embed, group_by_threshold  # noqa: E402
from litagent.config import EMBEDDING_PROFILES, SIMILARITY_HYBRID_DENSE_WEIGHT, Settings  # noqa: E402
from litagent.schemas import Paper  # noqa: E402
from litagent.similarity import paper_similarity, partition_metrics, select_partition  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def corpus() -> tuple[list[Paper], list[str]]:
    papers, labels, seen = [], [], set()
    for path in sorted((ROOT / "data/sample").glob("*.json")):
        for row in json.loads(path.read_text()):
            paper = Paper.model_validate(row)
            if not paper.abstract:
                continue
            keys = {"id:" + paper.id, "title:" + re.sub(r"\W+", "", paper.title.casefold())}
            if paper.doi:
                keys.add("doi:" + paper.doi.casefold())
            if keys & seen:
                continue
            seen |= keys
            papers.append(paper)
            labels.append(path.stem)
    return papers, labels


def external_metrics(similarity: np.ndarray, groups: list[list[int]], topics: list[str]) -> dict:
    from sklearn.metrics import adjusted_rand_score, average_precision_score, normalized_mutual_info_score, roc_auc_score
    predicted = np.empty(len(topics), dtype=int)
    for index, group in enumerate(groups):
        predicted[group] = index
    rows, cols = np.triu_indices(len(topics), k=1)
    actual = np.array([topics[i] == topics[j] for i, j in zip(rows, cols)])
    pairs = predicted[rows] == predicted[cols]
    tp = int(np.sum(actual & pairs))
    return {"coarse_ARI": float(adjusted_rand_score(topics, predicted)),
            "coarse_NMI": float(normalized_mutual_info_score(topics, predicted)),
            "coarse_pair_precision": tp / int(pairs.sum()) if pairs.any() else None,
            "coarse_pair_recall": tp / int(actual.sum()) if actual.any() else None,
            "coarse_pair_AUC": float(roc_auc_score(actual, similarity[rows, cols])) if len(set(actual)) == 2 else None,
            "coarse_pair_AP": float(average_precision_score(actual, similarity[rows, cols])) if actual.any() else None}


def evaluate(papers: list[Paper], vectors: np.ndarray, settings: Settings, topics: list[str] | None) -> list[dict]:
    scores = paper_similarity(papers, vectors, replace(settings, cluster_dense_weight=SIMILARITY_HYBRID_DENSE_WEIGHT))
    variants = [
        ("legacy_connected_0.7", scores.dense, replace(settings, cluster_dense_weight=1, cluster_threshold_mode="fixed")),
        ("dense_complete_0.7", scores.dense, replace(settings, cluster_dense_weight=1, cluster_threshold_mode="fixed")),
        ("dense_complete_adaptive", scores.dense, replace(settings, cluster_threshold_mode="adaptive")),
        ("hybrid_complete_adaptive", scores.combined, replace(settings, cluster_threshold_mode="adaptive")),
    ]
    results = []
    for name, matrix, variant in variants:
        if name == "legacy_connected_0.7":
            groups, threshold = group_by_threshold(matrix, 0.7), 0.7
        else:
            groups, threshold, _ = select_partition(matrix, variant)
        metrics = partition_metrics(matrix, groups)
        results.append({"strategy": name, "papers": len(papers), "threshold": threshold, **metrics,
                        "dense_diagnostics": partition_metrics(scores.dense, groups),
                        "weak_label_metrics": external_metrics(matrix, groups, topics) if topics else None,
                        "groups": [[papers[i].id for i in group] for group in groups]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", choices=EMBEDDING_PROFILES, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/similarity-comparison.json")
    args = parser.parse_args()
    papers, topics = corpus()
    settings = Settings(sample_mode=False)
    report = {"limitations": "Source topics are weak broad labels, not fine-grained semantic ground truth. "
              "Internal silhouette/coverage are cut-selection heuristics, not validation accuracy.", "profiles": {}}
    # Hash-based disjoint batches; no labels enter selection and no paper occurs in both batches.
    split = [int(hashlib.sha256(p.id.encode()).hexdigest(), 16) % 2 for p in papers]
    for profile in args.profiles or [settings.embedding_profile]:
        config = replace(settings, embedding_profile=profile, embedding_model_dir=None)
        key = hashlib.sha256(json.dumps({"profile": EMBEDDING_PROFILES[profile],
                                       "papers": [p.model_dump() for p in papers]}, sort_keys=True).encode()).hexdigest()
        cache = ROOT / f"data/cache/similarity-benchmark/{key}.npz"
        start = time.perf_counter()
        hit = cache.is_file()
        if hit:
            with np.load(cache, allow_pickle=False) as saved:
                vectors = saved["vectors"]
        else:
            vectors = embed([f"{p.title}. {p.abstract}" for p in papers], config)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, vectors=vectors)
        batches = {"dev_mixed": [i for i in range(len(papers)) if split[i] == 0],
                   "validation_mixed": [i for i in range(len(papers)) if split[i] == 1]}
        batches.update({topic: [i for i, t in enumerate(topics) if t == topic] for topic in sorted(set(topics))})
        profile_report = {"model": EMBEDDING_PROFILES[profile], "embedding_cache_hit": hit,
                          "embedding_seconds": time.perf_counter() - start, "batches": {}}
        for name, indices in batches.items():
            rows = evaluate([papers[i] for i in indices], vectors[indices], config,
                            [topics[i] for i in indices] if name.endswith("mixed") else None)
            profile_report["batches"][name] = rows
            for row in rows:
                coarse = row["weak_label_metrics"]
                print(f"{profile} {name} {row['strategy']}: t={row['threshold']:.2f} "
                      f"groups={row['clusters']} largest={row['largest_fraction']:.2%} "
                      f"coverage={row['coverage']:.2%} "
                      f"ARI={coarse['coarse_ARI'] if coarse else 'n/a'}")
        report["profiles"][profile] = profile_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
