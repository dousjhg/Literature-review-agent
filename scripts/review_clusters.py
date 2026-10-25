"""Prepare blind diagnostic pairs, or score human judgments against a cluster report (offline).

python scripts/review_clusters.py --report logs/clusters-<run>.json --output evals/cluster-review.json
python scripts/review_clusters.py --report logs/clusters-<run>.json --annotations evals/cluster-review.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cluster_pairs import prepare_review, score_reviews  # noqa: E402
from litagent.agents.clusterer import _text_for, embed  # noqa: E402
from litagent.config import Settings  # noqa: E402
from litagent.schemas import Cluster, Paper, QuestionCluster  # noqa: E402
from litagent.similarity import paper_similarity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--papers", type=Path, default=Path("data/sample/low_resource_languages_llm.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/cluster-review.json"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--per-stratum", type=int, default=Settings().cluster_review_pairs_per_stratum)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    clusters = [QuestionCluster.model_validate(c) if "kind" in c else Cluster.model_validate(c) for c in report["clusters"]]
    required = {pid for c in clusters for pid in c.paper_ids}
    papers = [Paper.model_validate(p) for p in json.loads(args.papers.read_text()) if p["id"] in required]
    if args.annotations:
        result = score_reviews(json.loads(args.annotations.read_text()), papers, clusters)
        print(json.dumps(result, indent=2))
        return 0
    manifest_path = args.output.with_suffix(".selection.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Review/selection file exists; choose another --output to preserve annotations")
    allowed_settings = {"cluster_dense_weight", "cluster_lexical_title_weight", "cluster_lexical_max_df",
                        "cluster_lexical_max_features", "cluster_lexical_ngram_range"}
    overrides = {k: v for k, v in report.get("similarity_settings", {}).items() if k in allowed_settings}
    if "cluster_lexical_ngram_range" in overrides:
        overrides["cluster_lexical_ngram_range"] = tuple(overrides["cluster_lexical_ngram_range"])
    settings = Settings(embedding_profile=report.get("embedding_profile", "semcse"), **overrides)
    texts = [_text_for(p, report.get("use_full_text", False)) for p in papers]
    scores = paper_similarity(papers, embed(texts, settings), settings)
    pack, manifest = prepare_review(papers, clusters, scores.combined, args.per_stratum, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {len(pack['reviews'])} pending pairs: {args.output}")
    print(f"Selection diagnostics (keep hidden during review): {manifest_path}")
    print(json.dumps(score_reviews(pack, papers, clusters), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
