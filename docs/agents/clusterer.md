# 3. Clusterer agent

Code: `litagent/agents/clusterer.py`, `litagent/cluster_labels.py` · Default prompt: `prompts/cluster_labeler_contrastive.md`

1. **Goal:** Group papers into labelled research ideas.
2. **Plan:**
   1. Build text per paper: metadata (Normal / Medium) or full text (High / Ultra, via Meterless Markovian engine).
   2. Embed each paper.
   3. Complete-linkage grouping: every pair in a group must meet the selected similarity threshold (default 0.7).
   4. Retrieve nearest groups, then name bounded batches using task/method/data/evaluation contrasts and source-span references.
   5. Write clusters to the world model.
3. **Tools:** pinned `CLAUSE-Bielefeld/SemCSE_cosine` on CPU with CLS pooling; MiniLM fast baseline; SciPy complete linkage; Nightshift/Nemotron labeller.
4. **Memory to keep:** clusters → world model.
5. **Rubric:** see `prompts/grader.md` → clusters. Rule-based size check runs first (no cluster > 40% of papers, ≤ 50% single-paper clusters).
6. **Stop conditions:** grade < 7/9 → redo once with the grader's fix → continue with a warning.
7. **Run log:** threshold used, number of clusters, size distribution.
8. **Guardrails:** runtime loads local model files only; no automatic download, remote model code or GPU. Input ids must be unique, vectors finite and non-zero, labels non-empty. Code assigns group memberships; the LLM can only name groups. Singletons are retained as `Unclustered: <title>` without an LLM call.

**Implementation choices:** title + abstract for Normal/Medium; supplied full text for High/Ultra, with a logged metadata fallback when missing. SemCSE encodes each token window using its documented CLS representation; MiniLM uses its original mean pooling. Window vectors are mean-pooled and normalized, with a logged per-paper cap. This is document embedding, not the pending Meterless Markovian reasoning engine. MMR selects at most `cluster_label_max_papers` central, non-redundant examples for the label prompt without changing group membership. All numeric controls live in `litagent/config.py`.

**Similarity strategies:** the default is semantic cosine and a fixed 0.7 cutoff. Complete linkage removes the previous chaining bug. Optional TF-IDF fusion uses sublinear TF, IDF, and separately weighted title/abstract features; unavailable sparse features fall back to dense similarity. Optional adaptive cuts maximize silhouette × non-singleton coverage over a bounded grid, without topic labels or LLM decisions. Experiments showed adaptive cuts can overmerge subtopics within a narrow field, so they are not the default. TF-IDF did not consistently improve clustering and is also opt-in. On a size-related redo the cutoff moves by one configured step and the fix also goes to the labeller; every result still satisfies its actual logged pairwise cutoff.

**Validation:** `scripts/benchmark_similarity.py` compares two encoders and four scoring/grouping strategies. Saved-file topic labels only evaluate coarse field separation, not expert subtopic quality. See [research, decisions and results](../similarity-optimization.md). `group_by_threshold` remains only as the legacy experimental baseline. Outliers and grading failures remain visible instead of forcing a desired number of clusters.

**Label quality:** `cluster_label_strategy="contrastive"` is the default; `independent` retains the earlier naming baseline.
Labels see all member titles, up to eight MMR source examples, and the two nearest groups (selected by closest cross-group pair).
Context includes both minimum and maximum cross-group similarity; proximity is not permission to merge.
The LLM selects paper/span IDs, and code copies exact source substrings into `Evidence`, validates group ownership and
source coverage, and rejects invalid output after a bounded repair. Coherence is explicitly `coherent`, `mixed`, or
`insufficient_evidence`; a coherent assessment requires body text and evidence for every supplied example. This checks
provenance, not whether a quote semantically entails a claim. Assessments are recorded in trace and the smoke report;
the public `Cluster` shape and sample snapshots remain unchanged. The grader does not see the producing agent's rationale.

**Human evaluation:** `scripts/review_clusters.py` generates blind, unlabelled diagnostic paper pairs and scores attributed
human judgments. No human labels are fabricated and LLM proposals are excluded from human metrics. Duplicate-record
components stay in one review fold. See [evaluation instructions](../../evals/README.md).

**Experimental semantic grouping:** `cluster_grouping_strategy="semantic"` uses `litagent/semantic_groups.py` and
`prompts/cluster_subtopics.md` to propose task/method/domain groups from bounded source spans and candidate neighbors.
Every proposal is projected back onto the original complete-linkage cutoff; no LLM proposal can bypass that bound.
Missing, overlapping or uncited assignments are retained as singletons with a trace event; unknown IDs and invalid
evidence fail validation. Title-only and content-isolated papers are retained without a model assignment. Size rules
become diagnostics for this candidate; the grader still evaluates the papers and the normal pass mark still applies.
The candidate does not lower the cutoff to reduce singletons. Default grouping remains `complete` pending evaluation.
On the inspected 30-paper batch the candidate fragmented the corpus at both 0.7 and 0.65; it is deliberately not the default.

**Independent model calls:** `llm.py` sends a fresh `meterless-scope` for each request and repair, in addition to per-run
`x-title`. Nightshift's H-MEM can reuse answers across app IDs even on cache misses; receipt injection counts are logged.

**Setup / verification:**

```bash
python -m pip install -r requirements.txt
python scripts/setup_embeddings.py  # default scientific model; Docker does this at build time
python scripts/setup_embeddings.py --profile minilm  # optional fast baseline
python scripts/check_threshold.py  # local embeddings only
python scripts/benchmark_similarity.py --profiles minilm semcse
python scripts/check_clusters.py   # real labels, grader/redo and World Model; saved papers, no retrieval
python scripts/check_clusters.py --label-strategy independent  # old naming control
python scripts/check_clusters.py --grouping-strategy semantic --max-redos 0  # one-shot candidate experiment
python scripts/review_clusters.py --report logs/clusters-<run_id>.json --output evals/cluster-review.json
```

**Input:** `list[Paper]`, Effort · **Output:** `list[Cluster]`
**Owner:** TODO · **Status:** Real implementation + offline tests; sample mode preserved. PDF retrieval and Markovian full-text reasoning remain pending.
