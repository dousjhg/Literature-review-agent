# 4. Connection finder agent

Code: `litagent/agents/connection_finder.py`, `litagent/finding_engine.py`.
Prompts: `prompts/connection_finder.md`, `prompts/connections_verify.md`.

**Status:** Real source-grounded implementation plus offline sample branch.

1. Reuse the Clusterer's original similarity matrix and original source fields.
2. Exclude title-only/insufficient-text records and detected duplicate studies (DOI/title).
3. Select bounded within-cluster and between-cluster comparisons, prioritizing main-related
   pairs. Defaults: at most four within, four between, two per paper, two per side group,
   and one selected comparison per pair of different groups. This is not exhaustive.
4. Ask Nemotron for a specific supported relationship or an explicit abstention. Relations
   are shared method, shared task, complementary contributions, contrasting comparable
   results, or an explicitly stated extension. Dates/similarity cannot prove influence.
5. Require source-span references from BOTH target papers. Code resolves exact original
   quotes and bounds repair attempts. A neighbor's quote cannot support a target paper.
6. Independently verify proposed claims against original excerpts, without producer
   reasoning, scores or verdicts. Fresh Nightshift scopes isolate every call.
7. Independently grade accepted findings, retain warnings and write typed findings to
   the World Model. A failed grade never obliges the producer to invent a connection.

Completed model tasks, including abstentions, are cached by complete inputs, full model
batch, source contexts, membership, model, prompts and schemas. Dependencies record the
referenced clusters and paper assignments. Output preserves `Connection` fields and adds
`relation`, `similarity` and `verification="source_supported"`. This verification is a model
judgment, not an expert or formal entailment proof.

Use [the cached map analysis API](../grounded-findings.md) or the normal `/run` pipeline.
Numeric controls live in `Settings`. Citation retrieval/graph-based influence inference
is not implemented; the agent makes no such claims without explicit supplied text.
