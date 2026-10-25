"""Regrade two fixed partitions with the SAME rubric, hiding variant names and rationales.

This is an LLM diagnostic, not expert ground truth or a membership-quality improvement.
Each grading call has an isolated trace/cache partition to avoid cross-variant replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent.agents.grader import grade  # noqa: E402
from litagent.cluster_audit import corpus_fingerprint, normalized_title, partition_indices  # noqa: E402
from litagent.config import Settings  # noqa: E402
from litagent.integrations.llm import TokenMeter, load_prompt  # noqa: E402
from litagent.schemas import Cluster, Paper  # noqa: E402
from litagent.trace import Trace, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--papers", type=Path, default=Path("data/sample/low_resource_languages_llm.json"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-partitions", action="store_true", help="Allow changed memberships; compare end-to-end quality, not naming alone")
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Use positive repeats and a new output path")
    setup_logging()
    reports = {"baseline": json.loads(args.baseline.read_text()), "candidate": json.loads(args.candidate.read_text())}
    clusters = {variant: [Cluster.model_validate(c) for c in report["clusters"]] for variant, report in reports.items()}
    required = {pid for c in clusters["baseline"] for pid in c.paper_ids}
    papers = [Paper.model_validate(p) for p in json.loads(args.papers.read_text()) if p["id"] in required]
    for values in clusters.values():
        partition_indices(papers, values)
    members = lambda values: sorted(tuple(sorted(c.paper_ids)) for c in values)
    identical = members(clusters["baseline"]) == members(clusters["candidate"])
    if not identical and not args.compare_partitions:
        parser.error("Partitions differ: a label-only comparison requires identical memberships")
    # Under a partition comparison BOTH variants receive source-based review; size stays diagnostic.
    settings = Settings(sample_mode=False, cluster_grouping_strategy="semantic" if args.compare_partitions else "complete")
    results = {v: [] for v in clusters}
    with TokenMeter() as meter:
        for repeat in range(args.repeats):
            for variant in (list(clusters) if repeat % 2 == 0 else list(reversed(clusters))):
                with Trace(settings.logs_dir) as trace:
                    result = grade("clusters", clusters[variant], papers, settings)
                    memories = [e.get("nightshift_memory") for e in trace.events if "llm call" in e["action"]]
                    results[variant].append({"run_id": trace.run_id, "grade": result.model_dump(), "total": result.total,
                                             "memory_receipts": memories})
                    print(f"{variant}, repeat {repeat + 1}: {result.total}/9")
        duplicates = {v: len(cs) - len({normalized_title(c.label) for c in cs}) for v, cs in clusters.items()}
        output = {"protocol": ("Memberships may change; size is diagnostic for BOTH variants; " if args.compare_partitions else "Fixed memberships; ") +
                              "same grader prompt; no producer rationale or variant identity in judge input; "
                              "alternating order; isolated cache app and per-request H-MEM scope. See memory receipts. "
                              "LLM diagnostic only; not human gold. "
                              "Exact duplicate labels is a lexical metric, not semantic separability.",
                  "corpus_fingerprint": corpus_fingerprint(papers),
                  "grader_prompt_sha256": hashlib.sha256(load_prompt("grader").encode()).hexdigest(),
                  "model": settings.llm_model, "repeats": args.repeats,
                  "source_reports": {"baseline": str(args.baseline), "candidate": str(args.candidate)},
                  "identical_memberships": identical, "exact_duplicate_label_excess": duplicates,
                  "results": results, "usage": meter.usage.model_dump()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
