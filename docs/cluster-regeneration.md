# Previewed side-cluster regeneration

Implemented 26 Oct 2026 (Australia/Sydney), after the initial main/side map.
This API lets a user select side-to-side moves, inspect the exact membership changes,
then generate a separate version. It does not automatically move boundary papers.

## Workflow

Start with a **new** `POST /cluster` question map (see [the complete request](../input_examples/cluster-map/lora-translation.json)).
Its response now includes `root_run_id`, `revision=0`, and an opaque `version_token`.
The root remains available at its original URL after every edit.

| Endpoint | Operation |
| --- | --- |
| `GET /cluster/{run_id}/regenerate/candidates` | Numerical side-to-side candidates, with mean/minimum target scores and current admission eligibility |
| `GET /cluster/{run_id}/regenerate/candidates?near_only=true` | Only candidates within the configured absolute margin; singleton sources have no measured margin |
| `GET /cluster/{run_id}/regenerate/candidates?paper_id=PAPER_ID` | Filter a selected paper before applying the result limit |
| `POST /cluster/{run_id}/regenerate/preview` | Simulate the selected simultaneous moves; zero model/embedding calls |
| `POST /cluster/{run_id}/regenerate` | Apply a saved, applicable preview to a new immutable child version |
| `GET /cluster/{run_id}/versions` | Lineage history and current active head |
| `POST /cluster/{root_id}/activate` | Select a completed saved version, including rollback |

Use actual IDs returned by the candidates endpoint in the preview request:

```json
{
  "moves": [
    {"paper_id": "PAPER_ID", "target_cluster_id": "TARGET_SIDE_ID"}
  ]
}
```

The preview returns `preview_id`, `sample_mode`, the source version token, a proposed child
run ID, `changes` with each affected group's `before`/`after` paper IDs, labels to regenerate
or reuse, dependent findings to invalidate, and `applicable`/`errors`. It creates no new map
and does not change the active head. A failed numerical check returns an inspectable
`applicable=false` plan; invalid IDs, duplicate moves or attempts to move main papers return 422.

Apply the chosen preview against the same source run:

```json
{"preview_id": "PREVIEW_ID_FROM_RESPONSE"}
```

The returned `ClusterMapResult` has a new `run_id`, `parent_run_id`, incremented `revision`,
the original `root_run_id`, and a `regeneration` summary. Its own `/cluster/{child_id}` and
`/world/{child_id}` URLs expose its result and provenance. A stage that completes becomes
the active head even if the independent grade fails; the failed grade and warning remain
visible. A partial child stays inspectable and leaves the source active.

Candidate scores are diagnostics, not a recommendation that moving a paper improves
scientific topic quality. The list is bounded and reports truncation; preview accepts any
explicit valid target, including one outside the displayed list. The
[map UI](../frontend/README.md) now exposes candidates, preview/apply and version rollback.

## Membership rules

- Paper IDs remain exclusive: every input paper appears exactly once.
- Main membership, question, paper sources and original relevance decisions are frozen.
  A numerical side similarity cannot admit a paper to main or remove a main member.
  Start a new map to re-assess the question or its evidence.
- Side IDs represent logical groups across this lineage, even when their members change.
  Initial IDs hash their initial members; later content hashes are cache keys, not new IDs.
  A source group that becomes empty is removed.
- All moves are applied simultaneously before validating affected final groups. In complete
  mode every final member pair must meet the original side cutoff. In average mode the
  final group pair mean and each incoming paper's mean against its final peers must meet
  the cutoff. A batch of individually eligible candidates can therefore be rejected.
- Assigned scores, runner-up groups, margins and hint edges are refreshed for **every**
  paper. Deleted groups and a paper's new primary group cannot remain as stale alternatives.

## Reuse and invalidation

New question maps persist the original papers, raw query/document vectors, combined
similarity matrix and scoring policy in ignored `logs/cluster-*` files. Regeneration loads
those numeric results and never calls the encoder or changes the lineage's metric/cutoff.
Each version has its own immutable copy of the numeric arrays. The ready manifest is
written last and checksums bind it to its JSON result and `.npz` arrays. Loading uses
`allow_pickle=False` and validates shapes, finite/nonzero vectors, matrix bounds/symmetry,
paper coverage, query scores, main identity and original-field relevance quotes.

Label keys cover member titles, selected source spans, retrieved neighboring contexts,
the **whole fixed model batch**, model, prompts, schemas, validation settings and labeling
implementation hashes. If any target context changes, the affected batch is regenerated.
An unchanged member list alone does not establish a reusable label. Singletons and
title-only abstentions have deterministic keys; the independent label strategy has its
own request keys. Main's label/relevance decisions are reused. The whole-map grader runs
again independently, through the existing isolated Nightshift wrapper.

The snapshot also supports finding dependencies on clusters, papers and other findings.
Dependencies on cluster membership/labels or moved paper assignments invalidate their
findings; invalidation propagates to dependent findings. Unknown/global dependency scope
is conservative. Unaffected explicitly local findings can be copied into the child World
Model. Invalidated findings are removed from current output and listed in the summary.

**Real Connection / Gap finders are now available** through
[`POST /cluster/{run_id}/analyze`](grounded-findings.md). Regenerating an analyzed map
reselects candidates, reuses exact matching positive/negative tasks and runs changed tasks,
including newly possible findings. Grades and coverage checks remain explicit. Budget or
provider interruptions produce a partial child with completed tasks retained for resume.
Older manually attached findings without analysis provenance retain the conservative
invalidation behavior: pending stages are listed and the child is not activated. Undeclared
dependency scope is global. New standalone cluster-only maps do not start finding stages
until explicitly analyzed.

## Conflicts, retries and rollback

Apply rechecks the active source/version token and the generation policy under nonblocking
lineage/child locks. Concurrent operations, stale previews or a policy change return 409.
Create a fresh preview after refreshing the active version. Current scoring settings do
not silently override the frozen source policy. Default bounds are 20 moves, 40 displayed
candidates and 30 saved regenerations per lineage, including branches after rollback.

A preview names one fixed child ID. Retrying an already committed apply returns that
same child without repeating model calls or changing a later active-head selection.
If an apply started but failed or was interrupted before commit, it records an attempt
receipt and will not repeat paid calls automatically. Inspect its trace and submit a new
preview for a deliberate retry. A crash after a ready snapshot but before head promotion
can leave an inspectable child; explicitly activate it if appropriate.

Rollback is an explicit compare-and-swap operation against the head returned by `/versions`:

```json
{
  "target_run_id": "OLDER_COMPLETED_RUN_ID",
  "expected_head_run_id": "CURRENT_HEAD_RUN_ID",
  "expected_head_version_token": "CURRENT_HEAD_TOKEN"
}
```

The target must be a completed version in the same lineage. The selection changes only
the head pointer and is logged; no saved map is overwritten. Reusing a cluster version ID
for `/cluster`, `/run` or a fresh `ResearchWorld` is rejected. Locks use Unix `flock`,
available on the project's macOS host and Linux Docker environment; they coordinate one
shared local filesystem, not a distributed database.

Older pre-cache checkpoints, legacy topic-only results and normal `/run` results cannot
be regenerated through these endpoints. They return 409 with instructions to create a new
`POST /cluster` map from the original papers. Sources/vectors are never reconstructed by
guessing. Sample and real lineages cannot be mixed.

## Reproduction and validation

```bash
python scripts/check_cluster_regeneration.py
python -m pytest -q
```

The smoke script calls the actual FastAPI routes in explicit sample mode using six saved
OpenAlex papers: create, select, preview, apply, idempotent retry, history and rollback.
It makes no model/network calls. The checked [sample report](benchmarks/cluster-regeneration-smoke.json)
records all nine workflow checks passing, zero LLM calls and cached numeric reuse.
Synthetic sample similarities and placeholder grades are **not semantic quality evidence**.

The automated suite additionally checks main freezing, exact coverage, simultaneous-move
rejection, average-mode admission, empty-group removal, full-batch and neighbor invalidation,
unaffected-label reuse, dependent finding invalidation, frozen scoring, corruption,
timeouts/budgets, failed-call receipts, conflicting locks, stale previews and rollback.
The full native suite and a network-disabled Linux CPU container each pass **270 tests**.
Container verification mounts the updated application, prompts, tests and evals read-only
into the existing `litagent-similarity:local` dependency/model image; that image was not rebuilt.
Real side-topic quality still requires the pending blind human review described in
[the first-phase report](main-side-clusters.md).
