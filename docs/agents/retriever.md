# 2. Retriever agent

Code: `litagent/agents/retriever.py` · Tools: `litagent/tools/`

1. **Goal:** Find the papers that match the search plan and store them in Zetaris.
2. **Plan:**
   1. Run each query on every configured source (`retrieval_sources`: OpenAlex and arXiv by default, plus Semantic Scholar when `SEMANTIC_SCHOLAR_API_KEY` is set). A failed search is logged and skipped; if every search fails the run stops with an error.
   2. Open-access only unless the user is signed in with a school account.
   3. Merge and remove duplicates across sources and versions (same DOI, same arXiv ID, including arXiv DOIs `10.48550/arXiv.*`, or same normalised title). The first-seen record keeps its ID; missing abstract/DOI/PDF link are filled from its duplicates and the highest citation count is kept. Then drop papers without an abstract: later agents quote abstracts as evidence. A stated citation minimum also drops records with no citation count (arXiv).
   4. Rank 70% relevance (reciprocal rank fusion across queries) / 30% citations per year (`rank_relevance_weight` in config), then cut to the effort level's paper limit.
   5. High / Ultra: download full-text PDFs (TODO; logged as skipped for now).
3. **Tools:**
   - `search_openalex` → query, criteria → `list[Paper]` → retries network errors, 5xx and short 429s 3×, then raises `OpenAlexError`; fails at once when the daily limit is used up (a free `OPENALEX_API_KEY` raises it 10×)
   - `search_semantic_scholar` → query, criteria → `list[Paper]` (`S2:<id>`) → retries 5xx and short 429s, then raises `SemanticScholarError`; year, citation-minimum and OA filters server-side, paper type filtered from `publicationTypes`
   - `search_arxiv` → query, criteria → `list[Paper]` (`arXiv:<id>`, PDF link always set, no citation count) → requests spaced 3 s apart as arXiv asks; retries 5xx, then raises `ArxivError`; skipped for reviews-only plans
4. **Memory to keep:** writes papers to Zetaris; adds paper nodes to the world model.
5. **Rubric:** no grading agent (team decision). Checked by Human review 1.
6. **Stop conditions:** zero papers found → status `no_results` with a message naming the filters and suggesting broader terms.
7. **Run log:** papers per query, failed queries, duplicates removed, papers without abstract, citation filter, final count; LLM summary fallback.
8. **Guardrails:** respect API rate limits; no paywalled downloads without a licence check.

**Open question:** institutional access via school login. Check the university's licence terms (publishers usually forbid automated bulk downloading). Safer alternative: signed-in users upload PDFs they already have.

**Input:** `SearchPlan` · **Output:** `list[Paper]`, `RetrievalSummary`
**Owner:** TODO · **Status:** OpenAlex, arXiv and Semantic Scholar working (mocked tests; live checks 26 Oct: OpenAlex and arXiv returned results, keyless Semantic Scholar returned 429); LLM summary checked on Nightshift; web Human review 1 via `POST /search`
