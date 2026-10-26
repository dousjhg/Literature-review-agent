"""The terminal annotator records only typed human judgments, with verbatim sentence evidence."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from evals.cluster_pairs import PairReview

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("annotate", ROOT / "scripts/annotate_cluster_pairs.py")
annotate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotate)


@pytest.fixture
def pack(tmp_path):
    path = tmp_path / "pack.json"
    shutil.copy(ROOT / "evals/cluster-review-low-resource.json", path)
    return path


def typed(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))


def test_decisive_judgment_copies_verbatim_sentences_and_quit_keeps_the_rest_pending(pack, monkeypatch):
    # Pair 1: an invalid choice, an out-of-range number, then a valid "same"; pair 2: quit.
    typed(monkeypatch, ["x", "s", "Both adapt LLMs to one task.", "99", "1", "1,2", "q"])
    annotate.annotate(pack, "Tester")
    data = json.loads(pack.read_text())
    first = PairReview.model_validate(data["reviews"][0])
    papers = {p["id"]: p for p in data["papers"]}
    assert first.provenance == "human" and first.reviewer == "Tester" and first.judgment == "same_subtopic"
    assert {e.paper_id for e in first.evidence} == set(first.paper_ids)
    assert all(e.quote in papers[e.paper_id]["abstract"] for e in first.evidence)
    assert all(r["provenance"] == "pending" for r in data["reviews"][1:])


def test_uncertain_needs_no_evidence_and_rationale_is_required(pack, monkeypatch):
    typed(monkeypatch, ["u", "", "Abstracts do not state the task.", "q"])
    annotate.annotate(pack, "Tester")
    first = json.loads(pack.read_text())["reviews"][0]
    assert first["judgment"] == "uncertain" and first["evidence"] == [] and first["rationale"]


def test_sentences_are_original_substrings():
    text = "First claim. Second claim! Third: 3.5 points?  Last"
    assert annotate.sentences(text) == ["First claim.", "Second claim!", "Third: 3.5 points?", "Last"]
    assert all(s in text for s in annotate.sentences(text))
