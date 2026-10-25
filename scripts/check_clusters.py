"""Exercise real embeddings, Nightshift labels, grading/redo and World Model on saved papers.

    python scripts/check_clusters.py

No OpenAlex search or Zetaris write is needed. Later pipeline stages are not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent.agents import clusterer  # noqa: E402
from litagent.cluster_audit import cluster_diagnostics  # noqa: E402
from litagent.config import EFFORT_PROFILES, Effort, Settings  # noqa: E402
from litagent.integrations.llm import TokenMeter  # noqa: E402
from litagent.integrations.meterless import ResearchWorld  # noqa: E402
from litagent.orchestrator import _graded_stage  # noqa: E402
from litagent.schemas import Paper, RunResult  # noqa: E402
from litagent.similarity import paper_similarity  # noqa: E402
from litagent.trace import Trace, mask, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, default=Path(__file__).resolve().parents[1] /
                        "data/sample/low_resource_languages_llm.json")
    parser.add_argument("--effort", choices=[e.value for e in Effort], default="normal")
    parser.add_argument("--label-strategy", choices=["independent", "contrastive"], default="contrastive")
    parser.add_argument("--grouping-strategy", choices=["complete", "semantic"], default="complete")
    parser.add_argument("--max-redos", type=int, default=Settings().max_redo_per_stage)
    parser.add_argument("--threshold", type=float, default=Settings().cluster_similarity_threshold)
    args = parser.parse_args()
    setup_logging()
    if args.max_redos < 0:
        parser.error("--max-redos must be non-negative")
    settings = Settings(sample_mode=False, cluster_label_strategy=args.label_strategy,
                        cluster_grouping_strategy=args.grouping_strategy, max_redo_per_stage=args.max_redos,
                        cluster_similarity_threshold=args.threshold)
    effort = Effort(args.effort)
    profile = EFFORT_PROFILES[effort]
    try:
        papers = [Paper.model_validate(p) for p in json.loads(args.papers.read_text())][:profile.max_papers]
        with Trace(settings.logs_dir) as trace, TokenMeter(profile.token_budget) as meter:
            world = ResearchWorld(trace.run_id, settings.logs_dir)
            world.start_attempt(1, args.papers.stem)
            world.add_papers(papers)
            result = RunResult(run_id=trace.run_id, status="partial", papers=papers)
            clusters = _graded_stage("clusters", lambda fix: clusterer.cluster(
                papers, effort, settings, fix, defer_labels_on_size_failure=True),
                                     papers, settings, trace, result, world)
            world.write_clusters(clusters)
            scores = paper_similarity(papers, clusterer.embed(
                [clusterer._text_for(p, profile.use_full_text) for p in papers], settings), settings)
            threshold = next(e["threshold"] for e in reversed(trace.events) if e["action"] == "group papers")
            audit = cluster_diagnostics(papers, clusters, scores.combined, threshold)
            # Last attempt replaces earlier assessments for clusters with the same IDs.
            assessments = {}
            for event in trace.events:
                if event["action"] == "group papers":
                    assessments = {}
                elif event["action"] == "assess cluster":
                    assessments[event["result"]["cluster_id"]] = event["result"]
            report = {"run_id": trace.run_id, "stage": "clusters", "papers": len(papers),
                      "paper_ids": [p.id for p in papers], "source_file": str(args.papers),
                      "label_strategy": settings.cluster_label_strategy,
                      "grouping_strategy": settings.cluster_grouping_strategy,
                      "embedding_profile": settings.embedding_profile,
                      "use_full_text": profile.use_full_text,
                      "similarity_settings": {key: getattr(settings, key) for key in (
                          "cluster_dense_weight", "cluster_lexical_title_weight", "cluster_lexical_max_df",
                          "cluster_lexical_max_features", "cluster_lexical_ngram_range")},
                      "audit": audit, "assessments": list(assessments.values()),
                      "clusters": [c.model_dump() for c in clusters], "warnings": result.warnings,
                      "usage": meter.usage.model_dump(), "paper_map": world.to_graph_json()}
            out = Path(settings.logs_dir) / f"clusters-{trace.run_id}.json"
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            for c in clusters:
                print(f"{c.id} ({len(c.paper_ids)} papers): {c.label}")
            for warning in result.warnings:
                print(f"Warning: {warning}")
            print(f"Report: {out}; LLM calls: {meter.usage.llm_calls}; tokens: {meter.usage.total}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {mask(str(e))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
