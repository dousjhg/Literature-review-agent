"""Run the question-aware cluster stage on saved papers; no retrieval or later TODO stages.

python scripts/check_question_map.py --question "How does LoRA improve low-resource machine translation?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent.cluster_preview import run_cluster_map  # noqa: E402
from litagent.config import EFFORT_PROFILES, Effort, Settings  # noqa: E402
from litagent.schemas import Paper, QuestionCluster  # noqa: E402
from litagent.trace import mask, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--papers", type=Path, default=Path("data/sample/low_resource_languages_llm.json"))
    parser.add_argument("--effort", choices=[e.value for e in Effort], default="normal")
    parser.add_argument("--main-threshold", type=float, default=Settings().main_similarity_threshold)
    parser.add_argument("--side-linkage", choices=["complete", "average"], default=Settings().side_linkage)
    parser.add_argument("--sample-mode", action="store_true")
    args = parser.parse_args()
    setup_logging()
    effort = Effort(args.effort)
    settings = Settings(sample_mode=args.sample_mode, main_similarity_threshold=args.main_threshold,
                        side_linkage=args.side_linkage)
    papers = [Paper.model_validate(p) for p in json.loads(args.papers.read_text())][:EFFORT_PROFILES[effort].max_papers]
    try:
        result = run_cluster_map(papers, args.question, effort, settings)
    except Exception as error:
        print(mask(f"FAILED: {type(error).__name__}: {error}"), file=sys.stderr)
        return 1
    for cluster in result.clusters:
        print(f"{cluster.id} [{cluster.kind if isinstance(cluster, QuestionCluster) else 'sample'}] "
              f"({len(cluster.paper_ids)} papers): {cluster.label}")
    print(f"Grade: {result.grade.total if result.grade else 'not completed'}/9; stage status: {result.status}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    print(f"Report: logs/cluster-map-{result.run_id}.json")
    print(f"LLM calls: {result.usage.llm_calls}; tokens: {result.usage.total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
