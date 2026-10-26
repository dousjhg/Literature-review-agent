"""Interactive terminal annotation of a blind cluster review pack; a human types every judgment.

    python scripts/annotate_cluster_pairs.py evals/cluster-review-low-resource.json --reviewer "Your name"

Shows one pending pair at a time with both abstracts split into numbered sentences. You choose
same / different / uncertain, type a rationale, and pick sentence numbers as evidence; the tool
copies those sentences verbatim, so quotes always match the original text. Progress is saved after
every pair (resume by running it again). Nothing is filled in without your keystrokes, and only
`provenance: human` rows written here count in scripts/review_clusters.py scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cluster_pairs import PairReview  # noqa: E402
from litagent.cluster_state import atomic_json  # noqa: E402

CHOICES = {"s": "same_subtopic", "d": "different_subtopic", "u": "uncertain"}


def sentences(text: str) -> list[str]:
    """Contiguous substrings of the original text, split after sentence punctuation."""
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]


def show(paper: dict, tag: str) -> list[str]:
    body = sentences(paper.get("abstract") or paper.get("full_text") or "")
    print(f"\n[{tag}] {paper['id']}  {paper['title']}")
    if not body:
        print("    (no abstract; only the title is available, so 'uncertain' is usually right)")
    for n, sentence in enumerate(body, 1):
        print(textwrap.fill(sentence, 100, initial_indent=f"  {tag}{n}. ", subsequent_indent="      "))
    return body


def pick(prompt: str, body: list[str], tag: str) -> list[str]:
    while True:
        raw = input(f"{prompt} (sentence numbers like 2 or 1,3; blank for none): ").strip()
        numbers = [int(x) for x in re.findall(r"\d+", raw)]
        if numbers and all(1 <= n <= len(body) for n in numbers):
            return [body[n - 1] for n in dict.fromkeys(numbers)]
        if not raw:
            return []
        print(f"  Use {tag} sentence numbers between 1 and {len(body)}.")


def annotate(path: Path, reviewer: str) -> int:
    pack = json.loads(path.read_text(encoding="utf-8"))
    papers = {p["id"]: p for p in pack["papers"]}
    pending = [i for i, r in enumerate(pack["reviews"]) if r["provenance"] == "pending"]
    print(textwrap.fill(pack["protocol"], 100))
    print(f"\n{len(pending)} of {len(pack['reviews'])} pairs pending. Type q at the judgment prompt to stop.")
    for done, index in enumerate(pending, 1):
        row = pack["reviews"][index]
        a, b = (papers[pid] for pid in row["paper_ids"])
        print("\n" + "=" * 100 + f"\nPair {done}/{len(pending)}")
        body_a, body_b = show(a, "A"), show(b, "B")
        while (choice := input("\nSame research subtopic? [s]ame / [d]ifferent / [u]ncertain / [q]uit: ").strip().lower()) not in {*CHOICES, "q"}:
            pass
        if choice == "q":
            break
        judgment = CHOICES[choice]
        while not (rationale := input("Rationale (task / method / data / evaluation): ").strip()):
            print("  A rationale is required.")
        evidence = []
        if judgment != "uncertain":
            while True:
                quotes_a, quotes_b = pick("Evidence from A", body_a, "A"), pick("Evidence from B", body_b, "B")
                if quotes_a and quotes_b:
                    break
                print("  A decisive judgment needs at least one sentence from each paper (or choose uncertain).")
            evidence = [{"paper_id": a["id"], "quote": q} for q in quotes_a] + [{"paper_id": b["id"], "quote": q} for q in quotes_b]
        review = PairReview(paper_ids=tuple(row["paper_ids"]), judgment=judgment, provenance="human",
                            reviewer=reviewer, rationale=rationale, evidence=evidence)
        pack["reviews"][index] = review.model_dump(mode="json")
        atomic_json(path, pack)
        print(f"  Saved ({judgment}).")
    left = sum(r["provenance"] == "pending" for r in pack["reviews"])
    print(f"\n{left} pairs still pending. Score with: python scripts/review_clusters.py --report <clusters report> --annotations {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--reviewer", required=True, help="Your name or initials, recorded with every judgment")
    args = parser.parse_args()
    if not args.reviewer.strip():
        parser.error("--reviewer must not be blank")
    if not sys.stdin.isatty():
        parser.error("Run this in an interactive terminal; judgments must be typed by a human reviewer")
    return annotate(args.pack, args.reviewer.strip())


if __name__ == "__main__":
    raise SystemExit(main())
