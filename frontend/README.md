# Literature Map · 文献地图

React + TypeScript + Vite workspace for saved `/cluster` question maps. The graph renders
the actual paper memberships, scientific connections and provisional gap hypotheses.
It includes original source quotes, alternative-cluster scores, explicit move previews,
immutable versions, rollback, and version-bound human review.

## Run locally

From the repository root (Node 22.12+ and Python dependencies are required):

```bash
cd frontend
npm ci
npm run build
cd ..
python run.py
```

Open <http://127.0.0.1:8000/>. Build before starting the API: it mounts `frontend/dist`
at startup. Docker builds the frontend in a separate Node stage and serves it from the
same Python application. API routes take precedence over static files.

For development, keep the API running and use `npm run dev` in this directory.
Vite proxies `/api` to `http://127.0.0.1:8000` (override with `LITAGENT_API_PROXY`).

Opening the UI reads saved data and does not start a model call. **检索论文**, **新建地图**, **运行分析**,
and applying a move in real mode can call the configured Nightshift model. The UI
shows the server's sample/real mode and the operation progress. Sample mode explicitly
labels its synthetic similarities, groupings and findings.

## Use the map

1. Choose a saved version, or **新建地图**. Two sources are available:
   - **在线检索** (default): type a research question (years, citation counts or paper type
     can be stated in it) and an effort level. `POST /search` runs the Planner and OpenAlex
     retrieval, stores the papers through Zetaris and returns the plan, a summary and the
     paper list (Human review 1). Nothing is approved automatically: untick papers to drop,
     or write feedback and **按反馈重新检索** to re-plan (stated criteria are kept), then
     **用保留的 N 篇建立地图** sends only the kept papers to `POST /cluster`.
   - **导入记录**: import a JSON paper array, a `/cluster` request (`input.query`,
     `input.effort`, `input.papers`), or a response with `output.papers`. A supplied
     query/effort replaces the dialog's prefilled values. Try
     `input_examples/cluster-map/lora-translation.json`; it contains original records.
2. Click a topic or paper. Inspect its source evidence, full abstract, keywords and
   numerical alternatives. Main is a question-relevance pool; paper membership is exclusive.
   Map coordinates are for navigation and do not encode semantic distance. On a phone,
   topic navigation and the inspector show full labels; the overview omits dense labels.
   Drag the map with a finger and use the zoom controls to inspect individual nodes.
3. Click a solid scientific connection to inspect both sources. Dashed similarity hints
   are navigational suggestions. Empty gap output is valid and does not prove no gap exists.
4. Select an eligible side-cluster target, **预览移动**, then explicitly apply the preview.
   Main papers stay fixed. The server validates the final simultaneous membership.
5. **版本记录** lets you inspect old versions or explicitly set a completed version as current.
   A stale preview shows a conflict inside the dialog; it cannot overwrite a changed head.
6. **人工验收** hides model decisions by default. Fill in the reviewer, reason, judgment
   and exact quotes from both original sources, then save. Reviewer identity is self-declared.
   Reading a review queue never creates human labels. Each version has its own annotations.
7. **AI 自动评分** asks the independent Review Agent to grade the version's explicit claims
   on source support, question relevance and testability (0–3 each). **查看已有评分** only
   reads a cached report. Missing hypotheses receive no score. Suggestions and their original
   quotes appear separately from the human form; judgments are saved only by explicit submission.
   Identical Agent requests reuse the report. See [rubric and API](../docs/agents/review_agent.md).

The footer exposes failed grades, partial runs and warnings. Original full text is loaded
on demand only when supplied. High/Ultra searches fetch open-access PDFs into the records; maps
imported from JSON use whatever full text the records already contain.

## Verification

```bash
npm run build
npx playwright install chromium       # once, if no local Chrome is available
npm run test:e2e
# Alternative using an installed Chrome:
PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
```

Tests launch temporary sample-mode APIs and fresh browser profiles. They exercise real
HTTP routes for analysis, evidence inspection, moves, rollback, explicit review saving,
JSON import, stale-version conflicts and mobile navigation. They never label the real
review pack or call a paid model. Screenshots/logs go to the OS temporary directory
(`LITAGENT_QA_OUTPUT` can override it), outside the repository.

The interface uses Chinese navigation with original paper titles and quotes preserved.
White surfaces, a cool pale sidebar, forest-green controls and native SVG nodes keep the
scientific records readable without decorative imagery or fabricated graph edges.
