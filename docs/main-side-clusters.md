# Question-aware main and side clusters

Implemented the main/side map and previewed regeneration from the 25 Oct 2026 proposal. `main` is the map's one pool of
papers directly answering the research question; `side` groups organize the rest by
topic. Each paper ID belongs to exactly one group. The main pool may be absent.
Implementation and validation: 26 Oct 2026, Australia/Sydney.

## Current behavior

1. The pipeline passes `SearchPlan.question` to the Clusterer. The question and papers
   use the same pinned, local encoder. Query scores use original normalized vectors;
   there is no batch centering or lexical-fusion score in the query gate.
2. `main_similarity_threshold` screens candidates. It is an engineering default,
   currently 0.5, not a calibrated probability or an expert-approved relevance cutoff.
   A high score alone does not admit a paper to main.
3. Candidates receive a source-based `direct` / `supporting` / `uncertain` assessment.
   Direct/supporting decisions require an original abstract/full-text span. Title-only
   records stay uncertain. Below-gate records are unassessed, not declared irrelevant.
4. Proposed direct members undergo an independent verification using their selected
   original source spans, without the producing agent's verdict or reason. Only papers
   still marked direct enter `MAIN`. Verification checks the question's specific
   method/task/setting constraints. Quotes prove provenance; semantic support remains
   a model judgment requiring review.
5. Remaining papers use fixed `side_similarity_threshold` (default 0.7). Complete
   linkage is the default; `side_linkage="average"` is available for explicit comparison.
   Complete linkage bounds every member pair; average linkage does not. Main is a
   question-relevance pool and has no pairwise topic-diameter requirement.
6. Side labels use bounded, evidence-grounded neighbor contexts. Initial side IDs hash their
   sorted member IDs and remain stable when retrieval order changes; regeneration preserves
   logical IDs across membership edits. Labels can still
   change with source text, question-selected remainder, neighboring groups or prompts;
   stable IDs alone are not sufficient cache keys.

`cluster_map_mode="question"` is the default for callers providing a question. Use
`cluster_map_mode="topic"` for the earlier topic-only flow. Calls without a question
remain compatible with the previous Clusterer interface. Normal `/run` sample output keeps
its existing offline placeholders. Standalone `/cluster` sample maps now use typed side
groups, synthetic content similarities of 1 and query scores of 0, explicitly marked as
sample data so the full regeneration API can be exercised without models.

## Paper details and alternate clusters

Each question cluster extends the legacy contract with `kind`, `question` and
`paper_details`. Details contain:

| Field | Meaning |
| --- | --- |
| `question_similarity` | Raw question-to-paper cosine; not a relevance probability |
| `relevance`, `reason`, `evidence` | Source-based relevance decision and its original quotes |
| `keywords` | Ranked TF-IDF terms found within an original title or abstract field |
| `assigned_similarity` | Mean content similarity to other members of the assigned group, excluding self |
| `runner_up_cluster_id`, `runner_up_similarity` | Highest-mean alternative among final groups, regardless of admission eligibility |
| `margin` | Assigned mean minus runner-up mean; null for singleton primary groups |
| `near_boundary` | Both a minimum related score and an absolute margin test pass; not uncertainty calibration |
| `also_related` | Bounded high-scoring alternatives with mean/minimum similarity, eligibility and a numerical explanation |

A singleton has no measured within-group mean, so it gets null rather than a fake
self-similarity of 1. Alternative main admission still requires the query gate and
verified direct relevance. Side eligibility uses its actual linkage criterion; main
membership takes priority. Runner-up is a diagnostic, not a permission to move.
Content scores are semantic cosine by default; enabling the existing optional lexical
fusion changes the side/alternative score to that configured mixture. Question scores
always remain the separate raw semantic cosine.

World Model `member-of` remains single-valued. `also-related-to` is a separate,
multi-valued hint edge and cannot create additional membership. Each paper node exposes
its current attempt's `cluster_detail`; paper entities are not duplicated. Hint scores
are not `connects-to` scientific findings. The UI can use these fields in paper details;
the [map UI](../frontend/README.md) now renders these details and keeps hints visually
separate from scientific findings.

## API and saved checkpoints

`POST /cluster` accepts already retrieved/provided records. It runs only the clustering
stage and one independent grade. Real Connection / Gap finders can then be run through
[`POST /cluster/{run_id}/analyze`](grounded-findings.md). Creating the map itself performs
no OpenAlex search or Zetaris write. Normal `/run`
still uses the existing retrieval/storage adapters and passes its question into clustering.

```bash
cd Literature-review-agent
python run.py
curl -s -X POST http://localhost:8000/cluster \
  -H 'Content-Type: application/json' \
  -d @input_examples/cluster-map/lora-translation.json
```

Request structure:

```json
{
  "input": {
    "query": "How does LoRA fine-tuning improve machine translation for low-resource languages?",
    "effort": "normal",
    "papers": []
  }
}
```

`papers` must actually contain 1–30 records for normal effort, at most the configured
effort limit for other levels, with unique IDs. Use the supplied JSON file for a complete
three-paper request. `run_id` is optional; normally let the server generate a new ID for
each preview. A preview is synchronous and bounded by model timeouts, paper limits and
the existing token meter, and never asks for keyboard input.

Responses use the existing `{run_id, status, output}` wrapper. Output explicitly says
`stage="clusters"`, includes `sample_mode`, and contains `main_cluster_id` (possibly
null), groups, the paper map, grade, warnings and usage. `status="done"` means this
stage completed, not that the grade passed or the full literature review finished.
Time/token limits return `status="partial"` and retain any finished map. A grading
failure is returned with a warning and does not automatically move papers or regenerate.

`GET /cluster/{run_id}` reads the saved checkpoint; `GET /world/{run_id}` exposes its
World Model provenance. Checkpoints are atomically saved to ignored
`logs/cluster-map-<run_id>.json`. The existing `/run` partial-result path also retains
completed clustering output and graph if grading runs out of tokens.

## Reproduce a real saved-paper run

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_question_map.py \
  --question "How does LoRA fine-tuning improve machine translation for low-resource languages?"
python scripts/check_question_map.py --sample-mode --question "How are LLMs adapted?"
pytest -q
```

The first command uses real Nightshift calls but only locally saved papers and model
weights. All generative calls remain in `integrations/llm.py`, with fresh per-call H-MEM
scopes and logged memory receipts.

## Evaluation and remaining scope

Real 30-record run `20261025-125211-381a8d` selected one main paper, W4408280927, whose
abstract explicitly discusses LSFTL / low-rank adaptation for low-resource MT. The other
29 records formed 18 side groups. All 30 membership edges and 60 separate hint edges
were retained. Eight reported model calls used 39,741 tokens. The grader accepted the
main choice in its explanation but scored the whole map **3/9**, criticizing side-topic
fragmentation/mixing and duplicate records. The warning is preserved. This is an
exploratory model judgment, not an expert accuracy benchmark.

An earlier run without the admission verifier put a generic SFT paper and a
keyword-only LoRA record into main. Its 6/9 score did not make those admissions correct.
Both outcomes are retained in the benchmark report; there is no claim that aggregate
grading improved. See [recorded results](benchmarks/main-side-map.json).

User-selected regeneration, zero-model movement preview, dependency tracking, numeric/label
reuse and rollback are now implemented; see [regeneration API and limitations](cluster-regeneration.md).
Real Connection/Gap finders and the interactive map UI are now implemented. Gap candidates
include main–side boundaries without changing exclusive paper membership. Robust duplicate
study identities remain outstanding.
Side-topic quality still requires the existing blind human review and further calibration.


## Validation

The initial native suite and offline CPU container passed 243 tests for the main/side phase.
The regeneration phase extends those checks; see [its validation](cluster-regeneration.md#reproduction-and-validation).
Container checks mount the updated application, prompts, tests and eval code read-only
into the existing `litagent-similarity:local` dependency/model image; the image itself
was not rebuilt for this change. The complete three-paper request is schema-validated.

Subtopic pair review retains cluster roles and excludes pairs involving main from both
sampling and co-membership scoring. It reports the excluded count instead of penalizing
a main relevance pool for containing different subtopics. Main relevance needs its own
question-specific human review; no human labels or accuracy figures were manufactured.

Saved previews include encoder profile, the intended full-text mode and side scoring
settings for reproducible review. High/Ultra use the full text supplied with the records
(the `/run` and `/search` Retrievers fetch open-access PDFs); missing text falls back to metadata
with a trace event. The `use_full_text` flag describes effort intent, not that every PDF was read.
