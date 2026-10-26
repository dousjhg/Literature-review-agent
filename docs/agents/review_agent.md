# Independent finding Review Agent

Code: `litagent/agents/review_agent.py`; prompt: `prompts/review_agent.md`.
The map's **人工验收** page includes **AI 自动评分** and **查看已有评分**.
The former assesses the current version's queue; the latter only reads a saved report.
Opening the review page does not call the model or reveal its judgments automatically.

The existing pipeline grader scores an entire stage. This Agent separately reviews each
explicit Connection claim or Gap hypothesis with three 0–3 dimensions:

| Dimension | What it assesses |
| --- | --- |
| Original-source support | Whether both original sources support the material premises |
| Research-question relevance | Whether the relationship addresses the stated task/method |
| Testability | Whether the connection is checkable or the hypothesis has a concrete comparison/test |

The total is an ordinal 0–9 diagnostic, not accuracy or confidence. A `supported` verdict
requires at least 2 in every dimension. `unsupported` identifies contradicted or unsupported
material assertions; `uncertain` records insufficient source information. The Agent assesses
the relationship/hypothesis, not the quality of each paper in isolation. Different existing
experimental settings do not by themselves invalidate a proposed future controlled comparison.

Pairs without a specific claim receive `needs_hypothesis`, with **null scores** and no model
call. Missing body sources receive `insufficient_sources`. Sample mode labels its placeholder
workflow explicitly and does not manufacture numerical grades.

## Independence and provenance

The prompt includes only the question, specific claim/test, paper identities and bounded
original-source spans. It excludes human reviewers, labels, reasoning, old model verdicts,
scores, similarities and cluster labels. Calls use the existing Nightshift wrapper, with
fresh H-MEM scope per call/repair and a separate review-run cache partition. This is a separate
context using the currently configured model; it is not independent model-family validation.

The model returns span IDs. Code copies their exact original text and requires both papers
for decisive judgments. Wrong/foreign span IDs, incomplete output IDs and inconsistent
supported scores receive at most the configured bounded validation repair.

Before saving, a separate-context call (`prompts/review_explanation_check.md`) audits every
reason and limitation against each paper's own source. It receives neither scores/verdicts
nor repair history. The server assigns IDs to every draft sentence/limitation; the audit must
cover every ID and both papers. Each attribution includes an assertion, its paper, source
references and a grounded/proposal/scope_limit/unsubstantiated/contradicted assessment. Grounded rows
require valid evidence from that same paper. Unsubstantiated/contradicted rows trigger repair
even if the auditor's summary issues list is empty. Claims and future tests are explicitly separated from evidence. It checks
cross-paper attribution, proposed versus reported metrics, and absence claims based only on
abstracts/excerpts. Any reported error triggers the same bounded repair loop; missing audit
IDs, deadline expiry, provider failure or persistent attribution errors prevent report saving.
The trace records each audit and the report identifies checked items. This normally adds one
call to a grading batch; the same token/deadline budget covers grading, auditing and repairs.
It is still a model check using the same configured model, not a guarantee of factual accuracy.

Sources additionally expose literal metric-term matches with span IDs (configurable in
`review_agent_metric_terms`). These are text-presence hints, not proof of measured performance.
A narrow deterministic guard rejects an unqualified Chinese assertion that an abstract gives
no evaluation metrics when that paper's supplied text names a configured metric. It applies
to statements attributed to one paper; it preserves caveats about missing definitions,
versions, detailed settings or complete metric lists. It does not cover every paraphrase,
language, metric or attribution error. A regression exercises a falsely approving auditor
to ensure this guard still rejects an erroneous omission of COMET.

This Agent performs no external coverage search, establishes no global novelty, and does
not promote a hypothesis into the map or override earlier finding verification. Its records
have `provenance="model"`; existing human annotations and immutable map files are unchanged.
The UI preserves unsaved human form fields while loading suggestions. If the user subsequently
submits a judgment after viewing suggestions, that submission records exposure to model advice.

## API and cache

- `GET /cluster/{run_id}/review-agent`: return `report` or null without model calls.
- `POST /cluster/{run_id}/review-agent`: body `{"expected_version_token":"..."}`.
  Return `report` and `reused`; a stale token or server/source mode mismatch returns 409.

Reports are separate checksum-protected `logs/review-agent-<fingerprint>.json` files.
The fingerprint binds to source version, corpus, exact claims/text, model, rubric prompt,
schemas (including the explanation auditor), limits and implementation. Human edits do not invalidate it. Identical requests
reuse the report; concurrent generation is locked. Invalid cached checksums fail closed.
Failed generation leaves human judgments intact and requires an explicit retry.

Default limits in `config.py`: 24 queue items, 6,000 source characters per paper, a 30,000
token accounting budget and a 180-second review deadline (also bounded by run timeout).
Token metering uses the provider receipt; an individual completion can exceed the remaining
budget before it is accounted for. Sample mode and test fixtures need no external services.

Tests cover source/ID validation, rubric consistency, source/mode/version binding, cache
reuse and corruption, exclusion of human labels, unchanged human/map bytes, null scores
for absent hypotheses, and the browser flow for viewing AI suggestions without auto-submitting
human judgments.

The attribution regression uses a NusaMT/spBLEU versus LSFTL/COMET fixture with valid quotes
but swapped prose. Stubbed auditor responses test bounded repair and failure handling, not
the semantic accuracy of the model. See [the source audit](../nusamt-source-audit.md) for the
separate original-paper check and the scope of the refreshed live review.
