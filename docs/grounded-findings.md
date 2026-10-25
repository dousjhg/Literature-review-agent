# Real Connection / Gap finders on cached question maps

Implemented 26 Oct 2026 (Australia/Sydney). A new `/cluster` question map can now run
actual Nemotron scientific finding stages using its saved papers, embeddings and matrix.
The normal `/run` pipeline calls the same agents. Offline sample mode remains explicit.

## Analyze the existing map

```bash
python scripts/check_findings.py --run-id YOUR_MAP_RUN_ID
```

Or read `version_token` from `GET /cluster/{run_id}` and submit:

```http
POST /cluster/{run_id}/analyze
Content-Type: application/json
```

```json
{
  "expected_version_token": "SOURCE_VERSION_TOKEN",
  "request_id": "OPTIONAL_32_CHARACTER_HEX_ID"
}
```

The source must be the active version or a partial descendant of the active version.
The response is a new immutable `ClusterMapResult` with `stage="findings"`, the same
question/membership/labels, a new run ID, parent/root IDs, `connections`, `gaps` and
`analysis`. The latter contains completed/pending stages, candidate/acceptance/abstention
counts, rejection reasons, cache task counts and independent grades. The original map
and World Model remain unchanged. A complete stage can have zero findings.

Completed versions become active even when a model grade fails, with the failed grade
and warning retained. Partial versions remain inspectable and leave the original head
active. Pass the partial version's ID/token with a NEW request ID to resume. Completed
positive and negative tasks are reused; an interrupted task can be retried deliberately.
Repeating a committed request ID returns its original child without more model calls or
changing a later rollback. The head/child locks and source tokens prevent concurrent
writes. Version selection/rollback uses the existing [regeneration API](cluster-regeneration.md).

## Sources and claims

The connection stage compares bounded within/cross-group paper pairs. The gap stage uses
bounded within-group and main–side pairs and produces testable, provisional research hypotheses.
Default limits are three within-group pairs and three main–side pairs, with one per group
or cluster pair. Main–side ranking blends the cached similarity (0.7) with local question
TF-IDF relevance (0.3); this is an engineering retrieval policy, not calibrated scientific
accuracy. Side–side gaps remain outside this selection. Supporting evidence does not
move a paper into main.
Both select original `abstract`/supplied `full_text` spans; the model references span IDs
and code copies their exact quotes. Each accepted finding requires evidence from BOTH
boundary papers. Invalid IDs, foreign papers, invented spans, short or missing evidence
trigger at most one source-validation repair by default and then fail safely.

Title-only records cannot substantiate real findings. Duplicate DOI/normalized-title
records are not independent corroboration. Main is still a question-relevance pool and
is not assumed to be a uniform research subtopic. Cosine scores and cluster labels only
retrieve candidate pairs; they cannot prove scientific relationships.

Separate verification calls receive original excerpts, selected quotes and the claim,
without producer reasoning/verdict/scores. Every model call/JSON repair uses the existing
`integrations/llm.py` Nightshift wrapper with fresh H-MEM scope and child-run cache partition.
Quotes establish provenance; semantic support and grades remain model judgments requiring
human review. A grade failure is visible and does not force more connections or gaps.

Real output extends the legacy `Connection` / `Gap` fields, so source quotes and roles
survive JSON round trips and appear on World Model edges/nodes. Hypotheses include their
basis, proposed test and coverage metadata. Sample-mode examples retain plain models and
are never promoted into real findings.

## Coverage checks

A gap must arise from explicit source limitations or a meaningful supported cross-paper
comparison, rather than a missing mention in two abstracts. Independent verification
can reject even a plausible question whose supplied quotes do not support it.

The checker searches for studies that would answer each accepted question, checks local
corpus excerpts plus top OpenAlex results, and suppresses a hypothesis if an addressing
study has exact source evidence. Queries include older and non-OA studies and use no
citation threshold. Original local records remain frozen; external studies are coverage
provenance, not automatically added to clusters or counted as new original members.

The bounded states are:

| Status | Meaning |
| --- | --- |
| `addressed` | An evidenced answering study was found; the hypothesis is removed and its counterevidence retained in the audit |
| `no_address_found_in_checked_sources` | No answer in these specific checked excerpts/top results; global novelty remains unknown |
| `inconclusive` | Ambiguous/incomplete coverage or records without usable body text |
| `unavailable` | At least one required search failed; never used as negative novelty evidence |

Each retained hypothesis is always `hypothesis_status="provisional"`. Even successful
searches are not systematic reviews: they are limited by query phrasing, top-result caps,
index coverage and excerpts. High/Ultra use only supplied full text; automatic PDF reading
remains outstanding. An abstract cannot establish the entirety of a paper's experiments.

The API and search semantics follow the primary [OpenAlex search documentation](https://help.openalex.org/api/searching/)
and [authentication documentation](https://help.openalex.org/api/authentication/). Existing
environment credentials are used without being printed. No account/sign-up actions occur.

## Reuse and regrouping

Each selected connection group is a cache task; all selected gap hypotheses share one
bounded batch. Keys include actual sources, membership/labels, question, selected pairs,
full batch, relevant finding context, model, prompts, schemas, settings and implementation.
Gaps also bind to the complete local corpus and an expiring coverage check. An unchanged
member list alone is insufficient. Negative outcomes are cached, so repeated analysis
does not pay again for unchanged abstentions. Coverage failure is not a cacheable absence.

After an analyzed map is regenerated, both finders reselect candidates for its final
memberships and reuse only exact matching tasks. Changed cross-cluster relationships and
new possible relationships are considered even when there were no old positive findings.
All numerical hints are refreshed separately. Candidate/batch changes can invalidate
more than the clusters whose members moved. Pending finding stages remain explicit if a
budget/provider/deadline interrupts computation. Older manually attached findings without
analysis provenance retain the conservative invalidation behavior.

All requests have per-call timeouts; stage deadlines and effort token budgets are enforced.
A partial result retains completed cache tasks and accepted results for an explicit resume.
There are no keyboard prompts, unbounded searches or model loops chasing a passing score.

## Recorded real run and validation

The first completed finding map was **`20261025-144502-ad0809`**. Its original clustering map is
`20261025-142442-4ae347`: the first 30 saved low-resource LLM papers, one main paper
(W4408280927), 18 side groups, SemCSE vectors and a 30×30 cached matrix. Medium effort
was used with the 30-record input, so each job had the existing 100,000-token budget.
The original map's 7/9 model grade is a diagnostic, not expert acceptance or an improvement
claim over earlier stochastic grades. See [map metadata](benchmarks/new-question-map.json).

The finding run compared eight connection pairs: one accepted, five abstained and two
rejected by independent source verification. Rejections included an unsupported claim
that a generic SFT paper used LoRA. The retained connection relates two low-resource SFT
studies and preserves exact original quotes from both. The independent connection grade
was 9/9; it remains a model judgment.

The first Gap request needed source-reference repair and the provider timed out. The
[partial report](benchmarks/grounded-findings-partial.json) preserves that failure and
completed Connection tasks. An explicit resume reused **all six** completed Connection
cache tasks without rerunning their producer/verifier calls. Its three Gap candidates
produced one abstention and two independent rejections, leaving **zero** reported gaps.
The gap grade was **3/9**, and its warning remains in the
[completed report](benchmarks/grounded-findings.json). No unsupported hypothesis was added
to satisfy a quantity or grading target.

That run exposed a grading-scope omission: the grader saw the corpus without the original
question and precise reviewed-pair scope, and demanded a hypothesis outside the bounded
analysis. Future grading requests now include the question and only the selected pair IDs,
while omitting producer/verifier reasoning, verdicts and similarity scores. The recorded
3/9 score was neither rerun nor replaced. Scientific quality and recall still require
human review. That version's within-group-only selection missed main–side hypotheses;
the expansion recorded below addresses this candidate-scope limitation without claiming
an exhaustive gap search.

Because no hypothesis passed initial source verification, the actual finding run correctly
made no novelty-search claims. Separately, the real
[coverage integration probe](benchmarks/gap-coverage-probe.json) used a deliberately answered
question: one OpenAlex request returned four studies, and the independent checker cited
W4408280927 as already answering it. This probe created **no research gap** and did not
change the active map. It checked 28 available local body excerpts among 30 considered
records, explicitly retaining the two missing-body records as a coverage limitation.

The [workflow audit](benchmarks/finding-workflow-checks.json) passes ten integrity/component
checks: frozen membership, byte-equivalent numerical arrays, original quote validity,
resume cache reuse, completed stages, active-head selection, actual OpenAlex success and
coverage counterevidence. All 23 returned model receipts across clustering, partial run,
resume and probe report zero prior-answer-memory injection. They report 94,529 tokens;
the timed-out request supplied no token receipt. These are workflow measurements, not
semantic accuracy scores.

That phase's native suite and a network-disabled Linux CPU container both passed **288 tests**.
Tests cover source-reference repair, false semantic claims, title-only/duplicate exclusion,
positive and negative caches, coverage counterevidence, API idempotency, partial resume,
regrouping with new/invalidated findings, matrix reuse, graded scope and World Model
conformance. Sample API outputs were regenerated after correcting their obsolete
"real Gap finder is not implemented" placeholder message. Docker verification mounts
updated code, prompts, tests, evals and sample outputs read-only into the existing
`litagent-similarity:local` dependency/model image; it does not rebuild or download models.

Resumable task snapshots apply to `/cluster/{run_id}/analyze` and analyzed-map regeneration.
Normal `/run` retains its existing stage-level partial-result contract; it does not
create resumable `/cluster` snapshots automatically. Automatic PDF retrieval/full-text reading,
duplicate-study disambiguation beyond the current heuristics, and expert semantic quality
review remain outside this implementation.

## Main–side expansion and interactive review

The new completed active map is **`20261025-230430-c3fbbf`**, from the same frozen
30-paper corpus, one main paper and 18 side groups. The
[expansion receipt](benchmarks/main-side-gap-expansion.json) records 15 returned model calls
and 52,596 tokens, reusing the numerical cache. Eight Connection candidates yielded one
retained connection. Six Gap candidates (three within-group and three main–side) yielded
five abstentions and one independently rejected proposal, leaving **zero** gaps.
No novelty search was represented as having succeeded for a rejected hypothesis.

The connection grade was **5/9** with a retained warning; the gap grade was **9/9** for
honest abstention within the declared scope. These model judgments differ from earlier
ones and establish neither expert accuracy nor an improvement attributable to this policy.
The new proposal fields and both source records are available for human inspection.

The [interactive map](../frontend/README.md) reads saved versions without model calls.
It displays source quotes, numerical hints, scientific edges, move previews and version
rollback. `GET /maps`, `GET /ui/config`, and `GET /cluster/{id}/papers` provide catalog,
server mode and original metadata; individual `/papers/{paper_id}` reads supplied full
text on demand. `GET /cluster/{id}/review` hides model assessments by default;
`?reveal=true` explicitly reveals them.

`POST /cluster/{id}/review/{item_id}` accepts a version token, self-declared reviewer,
reason, verdict (`supported`, `unsupported`, `uncertain`), original evidence and whether
the reviewer saw the model assessment. A decisive judgment requires exact quotes of at
least 20 characters from both sources. Annotations live in a separate version/corpus-bound
sidecar with overwrite history; the original map is immutable. Reviewer identity is not
authenticated, and quote matching cannot establish that a judgment was made by a human.

The frozen [review pack](../evals/finding-review-main-side.json) has **seven pending items**
(one retained connection and six gap candidate pairs), all annotations blank. See
[review instructions](../evals/README.md). Its queue includes abstentions so recall can be
inspected, although it is still a bounded selected sample. Support rate is null until there
are decisive valid submissions; semantic accuracy is deliberately null.

The expanded native Python suite and the network-disabled Linux CPU container each pass
**295 tests**. The frontend production build passes TypeScript checks, and three fresh
Chrome browser tests exercise actual sample-mode HTTP routes, including JSON import,
move/apply/rollback, explicit temporary review submissions, stale-preview conflicts,
mobile touch panning and overflow checks. A separate read-only browser pass inspected
the real 30-paper map and seven-item queue at 1536×1024 and 390×844, with no console errors
or horizontal overflow. The container check reused the existing dependency image with
updated code and assets mounted read-only; the full new Docker image was not rebuilt.
