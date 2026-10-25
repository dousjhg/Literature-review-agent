"""Compare cosine thresholds on saved real papers, with no LLM or network calls.

    python scripts/check_threshold.py
    python scripts/check_threshold.py --limit 30 --output logs/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent.agents.clusterer import cosine_similarity_matrix, embed, group_by_threshold  # noqa: E402
from litagent.config import CLUSTER_THRESHOLD_SWEEP, EMBEDDING_PROFILES, Settings  # noqa: E402
from litagent.schemas import Paper  # noqa: E402
from litagent.similarity import complete_linkage_groups, paper_similarity, partition_metrics  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "data/sample"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=EMBEDDING_PROFILES)
    parser.add_argument("--legacy", action="store_true", help="Compare the original cosine connected-components baseline")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    settings = Settings()
    if args.profile:
        settings.embedding_profile = args.profile
    report = []
    for path in sorted(SAMPLES.glob("*.json")):
        papers = [Paper.model_validate(p) for p in json.loads(path.read_text())][:args.limit]
        vectors = embed([f"{p.title}. {p.abstract or ''}" for p in papers], settings)
        similarity = cosine_similarity_matrix(vectors) if args.legacy else paper_similarity(papers, vectors, settings).combined
        for threshold in CLUSTER_THRESHOLD_SWEEP:
            groups = group_by_threshold(similarity, threshold) if args.legacy else complete_linkage_groups(similarity, threshold)
            sizes = sorted((len(g) for g in groups), reverse=True)
            row = {"sample": path.stem, "papers": len(papers), "threshold": threshold,
                   "clusters": len(groups), "sizes": sizes, "profile": settings.embedding_profile,
                   "linkage": "connected" if args.legacy else "complete",
                   "metrics": partition_metrics(similarity, groups)}
            report.append(row)
            print(f"{path.stem} threshold={threshold:.1f} clusters={len(groups)} sizes={sizes}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
