# 5. Gap finder agent

Code: `litagent/agents/gap_finder.py`, `litagent/finding_engine.py`.
Prompts: `prompts/gap_finder.md`, `prompts/gaps_verify.md`, `prompts/gap_coverage.md`.

**Status:** Real, provisional research hypotheses with bounded coverage checks; offline sample branch retained.

1. Reuse the original matrix. Select at most three within-group paper pairs, at most one
   per group, with similarity in the configured range (default 0.5–0.9). Exclude detected
   duplicate studies and records lacking body evidence.
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

The output extends `Gap` with `basis`, `proposed_test`, `similarity` and `coverage`.
Findings remain reviewable model hypotheses, not confirmed research absences.
See [API, reproduction and limitations](../grounded-findings.md).
