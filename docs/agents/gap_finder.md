# 5. Gap finder agent

Code: `litagent/agents/gap_finder.py`, `litagent/finding_engine.py`.
Prompts: `prompts/gap_finder.md`, `prompts/gaps_verify.md`, `prompts/gap_coverage.md`.

**Status:** Real, provisional research hypotheses with bounded coverage checks; offline sample branch retained.

1. Reuse the original matrix. Select at most three within-group pairs (one per group)
   and three main–side pairs (one per cluster pair), with similarity in the configured
   range (default 0.5–0.9). Exclude detected duplicates and records lacking body evidence.
   Cross-boundary ranking blends matrix similarity (0.7) and local question TF-IDF
   relevance (0.3); the latter is a retrieval hint, not evidence of scientific support.
   These bounded defaults are not calibrated expert-quality thresholds.
2. Propose an author-stated open problem or a clearly speculative cross-paper hypothesis,
   with a testable question, baseline/evaluation plan, importance and original span
   references from BOTH papers. Omitted mentions in a few abstracts are insufficient.
3. Independently verify source support and scope. Reject unsupported questions instead of
   filling a gap quota. Zero accepted gaps is a valid possible result.
4. Search for the positive study that would ANSWER the question: at most two short queries
   per hypothesis and four top OpenAlex results per query. Coverage uses no OA, year or
   citation-popularity restriction. Each request has a deadline and one HTTP attempt.
5. An independent checker receives bounded original-source excerpts from the entire local
   corpus and returned search hits. Suppress a hypothesis when an evidenced study addresses
   it, including a local paper outside the proposed boundary pair.
6. Keep remaining results explicitly `hypothesis_status="provisional"`, with status,
   timestamps, queries, retrieved studies, evidence and the coverage scope. Failed searches
   are `unavailable`; incomplete sources/checks can be `inconclusive`. No absence of a match
   establishes global novelty.
7. Independently grade and store surviving hypotheses in the World Model.

Gap tasks bind to source context, the complete local corpus, relevant connections,
model/prompts/schemas and their whole batch. Completed negative decisions are cached too.
Coverage records expire after the configured lifetime (default one day); unavailable
coverage is retried only in a new explicit analysis request, never reused as absence proof.
Dependencies include boundary membership and the connection findings supplied as context.

The output extends `Gap` with `basis`, `proposed_test`, `similarity`, `coverage`,
`scope` and `cluster_ids`. A main–side hypothesis links both clusters in the World Model;
its legacy `cluster_id` anchor is the main group. Legacy within-group output remains compatible.
Findings remain reviewable model hypotheses, not confirmed research absences.
See [API, reproduction and limitations](../grounded-findings.md).
