# Build prompts for Claude Code

Paste these one at a time, in this order. Each step depends on the one before.
`CLAUDE.md` in the repo root already gives Claude Code the project rules, so the prompts stay short.
The hackathon technical-requirements prompt (Nightshift, Zetaris, `/run`, sample mode, logs, Docker) has already been applied.

**Before each prompt:** fill in the "Decisions" lines. If you leave one blank, Claude Code should ask you rather than guess.
**After each prompt:** run `pytest` and a sample-mode run (`python -m litagent "your question" --sample-mode`) yourself, read the diff, then commit.

---

## 0. LLM wrapper (do this first: every agent uses it)

Done: `complete_json` calls Nightshift (Nemotron 3.5 Super VL), counts tokens per run (`TokenMeter`) and enforces the effort level's token budget. Use this prompt only to change or extend it. Check the live connection with `python scripts/check_nightshift.py`.

```
Review and extend complete_json in litagent/integrations/llm.py.

Decisions:
- Provider: Meterless Nightshift (fixed by the hackathon rules; OpenAI-compatible, official OpenAI client)
- Model: Nemotron 3.5 Super VL (NIGHTSHIFT_MODEL in litagent/config.py, no automatic routing; confirm the ID in Nightshift's Providers tab)
- Structured output method: response_format json_object + schema in the system prompt, unless Nightshift supports json_schema for this model

Requirements:
- Read METERLESS_BASE_URL and METERLESS_RUNTIME_TOKEN from environment variables (see .env.example). Never read or store the NVIDIA key.
- Parse the response into output_model; if validation fails, retry once and include the validation error in the retry.
- Keep logging the model name and token usage for every call, and keep counting tokens in the run's TokenMeter (budget enforced before each call).
- Keep the request timeout (settings.llm_timeout_s).
- Keep tests/test_llm.py passing with the client mocked. Don't call the real API in tests.

Plan first and show me the plan before writing code.
```

## 1. Planner

Done (mocked tests). Remaining: tune `prompts/planner.md` on real questions once Nightshift is set up.

```
Implement the real (non-sample-mode) Planner in litagent/agents/planner.py, following docs/agents/planner.md and prompts/planner.md.

Decisions:
- How to decide there is "no clear topic": the LLM decides, with examples in the prompt (empty message caught in code)
- Number of search queries per plan: by effort, Normal 2 / Medium 3 / High-Ultra 4
- Citations: filter only when the user gives a number; otherwise rank by citations per year

Requirements:
- Keep apply_defaults() as the only place defaults are applied.
- When feedback from a human review is passed in, the new plan must change only what the feedback asks for.
- Improve prompts/planner.md with 3 examples: a clear question, a question with no topic, and a re-plan with feedback.
- Add tests with complete_json mocked.

Plan first and show me the plan before writing code.
```

## 2. Retriever

Done for OpenAlex (zero results → `no_results` status). Remaining: Semantic Scholar / arXiv, full-text PDFs, and checking the LLM summary on Nightshift.

```
Implement the real Retriever in litagent/agents/retriever.py and the search tools in litagent/tools/, following docs/agents/retriever.md.

Decisions:
- Sources: OpenAlex only for now (Semantic Scholar / arXiv later)
- Ranking before cutting to max_papers: 70% relevance (rank fusion across queries) / 30% citations per year; papers without an abstract dropped
- Full text at High/Ultra: later (logged as skipped)
- School-account access: leave as a TODO for now (licence check pending)

Requirements:
- Check litagent/tools/openalex.py against the OpenAlex docs and fix anything wrong.
- Merge results from all sources and deduplicate.
- If zero papers are found, return an empty list and log it; the orchestrator should tell the user to broaden the question.
- Implement summarise() with the LLM.
- Keep every HTTP call under a timeout.
- Add tests with HTTP mocked (sample API responses saved under tests/fixtures/).

Plan first and show me the plan before writing code.
```

## 3. Clusterer

Implemented and optimized: pinned CPU SemCSE cosine/CLS embeddings (MiniLM fast baseline),
complete-linkage grouping, retained single-paper outliers, MMR label examples, Nightshift labels,
bounded grading redo, offline model loading and Docker build-time provisioning. Verify with
`scripts/check_threshold.py` and `scripts/check_clusters.py`. Full-text retrieval and the
Meterless Markovian reasoning engine are still separate TODOs.
The latest research/ablation decisions are in `docs/similarity-optimization.md`.

```
Implement the real Clusterer in litagent/agents/clusterer.py, following docs/agents/clusterer.md and prompts/cluster_labeler.md.

Decisions:
- Embedding model: CLAUSE-Bielefeld/SemCSE_cosine, pinned revision in config.py, CPU only; MiniLM optional
- Text to embed: title + abstract; supplied full text at High/Ultra, metadata fallback logged
- Single-paper groups (outliers): keep as explicit "Unclustered" groups, without a labeller call
- Threshold: start at 0.7 (config.cluster_similarity_threshold)

Requirements:
- Use cosine_similarity_matrix and complete-linkage grouping; keep group_by_threshold as an experimental baseline.
- Label each group with the LLM; pass the grading agent's fix into the prompt when given.
- The Docker image must not download models at runtime: download the embedding model in the Dockerfile at build time. No local GPU.
- Write a small script (scripts/check_threshold.py) that runs on data/sample/ and prints the number of clusters and their sizes for thresholds 0.5-0.9, so we can check 0.7 works.
- Add tests with embeddings and the LLM mocked.

Plan first and show me the plan before writing code.
```

## 4. Grading agent

```
Implement the real grading agent in litagent/agents/grader.py, following docs/agents/grading_agent.md and prompts/grader.md.

Decisions:
- Model: ______ (same Nemotron model as the agents, or a different Nightshift model so it doesn't share their blind spots)
- When a stage still fails after one redo: ______ (continue with a warning, or ask the user)

Requirements:
- Keep the rule-based cluster size check running before the LLM grader.
- Finish the TODO rubric items in prompts/grader.md for connections and gaps.
- The grader must not see the producing agent's reasoning, only its output and the papers.
- Add tests: a good output passes, a bad one fails with a fix.

Plan first and show me the plan before writing code.
```

## 5. Connection finder

```
Implement the real Connection finder in litagent/agents/connection_finder.py, following docs/agents/connection_finder.md and prompts/connection_finder.md.

Decisions:
- Which pairs to compare within a cluster: ______ (e.g. top-k most similar per paper, k = 3)
- Max pairs per cluster: ______
- Connection types to show in the UI: ______
- Between clusters: ______ (compare cluster labels + 2-3 representative papers each)
- Use Semantic Scholar citation links as evidence: yes / no

Requirements:
- Every connection needs a verbatim quote from each paper. Add a check in code that each quote actually appears in the paper's abstract or full text, and drop connections that fail it.
- Pass the grading agent's fix into the prompt when given.
- Add tests with the LLM mocked, including one where a fabricated quote gets dropped.

Plan first and show me the plan before writing code.
```

## 6. Gap finder

```
Implement the real Gap finder in litagent/agents/gap_finder.py, following docs/agents/gap_finder.md and prompts/gap_finder.md.

Decisions:
- Similarity range for comparing papers: ______ (draft 0.5-0.8)
- Max gaps per cluster: ______
- Coverage check (search to see if a gap is already filled): yes / no

Requirements:
- Only compare papers within the same cluster (team decision).
- Same quote check as the Connection finder.
- If the coverage check finds a paper that fills the gap, drop the gap and log why.
- Pass the grading agent's fix into the prompt when given.
- Add tests with the LLM and search mocked.

Plan first and show me the plan before writing code.
```

## 7. Sponsor integrations (after reading the sponsor docs)

```
Replace the in-memory parts of litagent/integrations/zetaris.py and litagent/integrations/meterless.py with the real Zetaris and Meterless integrations.

Already in place: ZetarisRestClient.run_sql (documented REST API from github.com/zetaris/getting-started),
and the WorldModel interface + tests/test_meterless_world_model.py.

Here is what we know about each: ______ (paste notes from the Data Day recording, the Resources tab and the Help channels)

Requirements:
- Keep the sample-mode in-memory versions working for tests.
- Keep the public methods the same so the agents don't change.
- Keep timeouts on every call and log each Zetaris query and Meterless Engine call to the trace.
- tests/test_meterless_world_model.py must pass against the real World Model.
- Update docs/sponsor-tech.md with what each integration does.

Plan first and show me the plan before writing code.
```
