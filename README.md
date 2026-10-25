# TODO: Project name

> TODO: One-sentence pitch. Example: *An agent that maps a research field for you: it finds the papers, groups them into ideas, shows how they connect, and points out where the gaps are, with every claim traced back to a source.*

**Open Agent Hackathon 2026 · Track 02 — The Agent That Can Explain Why**

| | |
| --- | --- |
| Demo video (1–4 min) | TODO: link |
| Slide deck | TODO: link |
| Live demo (optional) | TODO: link |

## The problem

TODO: 2–3 sentences. Who is the user (a research student starting a literature review?), what do they do today, and why is it slow or error-prone?

## What it does

TODO: Describe the flow from the user's side in 4–6 bullets.

- The user types a research question into a chat box and picks an effort level (Normal / Medium / High / Ultra).
- TODO
- TODO

## Architecture

The full diagram and design decisions are in [ARCHITECTURE.md](ARCHITECTURE.md).
The file inventory, implemented flow and remaining work are in [docs/project-flow.md](docs/project-flow.md).

Five agents run in sequence, each followed by a check:

| Step | Agent | Checked by |
| --- | --- | --- |
| 1 | Planner: turns the question into a search plan | Asks the user for a topic if none is given |
| 2 | Retriever: finds papers through scholarly APIs | Human review 1 |
| 3 | Clusterer: groups papers into labelled ideas | Grading agent |
| 4 | Connection finder: links papers within and between clusters | Grading agent |
| 5 | Gap finder: finds research gaps inside clusters | Grading agent, then Human review 2 |

Per-agent specs are in [docs/agents/](docs/agents/).

## Sponsor technology

See [docs/sponsor-tech.md](docs/sponsor-tech.md) for the full write-up.

| Sponsor | How we use it | Status |
| --- | --- | --- |
| Meterless Nightshift | Primary AI provider: every LLM call goes through it (`litagent/integrations/llm.py`, OpenAI-compatible client) | Wired; confirm model ID in Providers tab |
| NVIDIA | Nemotron (`NIGHTSHIFT_MODEL` in `config.py`), connected through Nightshift (the NVIDIA key lives in Nightshift's Providers tab). The trace logs the model that handled each request. | Configured model: Nemotron 3 Ultra |
| Zetaris | Shared paper data layer, running locally in Docker: the Retriever writes papers to our paper store, which Zetaris serves as the `litagent_db` data source, and every later agent reads them through Zetaris SQL (`litagent/integrations/zetaris.py`) | Working (live round trip) |
| Meterless World Model | Implemented in Python from Meterless's spec and reference (`litagent/integrations/world_model/`). Holds every run's papers, authors, clusters, connections (with evidence) and gaps; later agents read from it, `/run` returns it as `paper_map`, `GET /world/{run_id}` returns the full world | Working: all 8 Meterless conformance checks pass |
| Cursor (with SpaceX XAI) | Optional: development environment, project rules in `.cursor/rules/` | Optional |

Each sponsor integration sits behind its own adapter class, so an outage in one doesn't break the whole project, and sample mode runs without any of them.

## Getting started

### Requirements

- Python 3.11+
- For real runs: Meterless Nightshift with a Nemotron model, and Zetaris running locally in Docker (setup: [docs/setup-zetaris.md](docs/setup-zetaris.md)). Sample mode needs neither.
- Recommended: a free OpenAlex API key (`OPENALEX_API_KEY`). Without one, OpenAlex allows about 100 searches a day; with one, about 1,000.
- No GPU needed.

### Setup

```bash
git clone TODO-repo-url
cd TODO-repo-folder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your values
python scripts/setup_embeddings.py  # real clustering: download the pinned CPU encoder once
```

Real clustering defaults to the scientific-content model `SemCSE_cosine` with complete linkage
and a 0.7 cosine cutoff. To use the faster MiniLM baseline, prepare it with
`python scripts/setup_embeddings.py --profile minilm` and set `EMBEDDING_PROFILE=minilm` in `.env`.
The research, optional hybrid/adaptive strategies and ablation results are in
[docs/similarity-optimization.md](docs/similarity-optimization.md).

Set up the Zetaris paper store once (Zetaris must already be running in Docker):

```bash
docker compose up -d                 # our paper store (Postgres, localhost:5433)
python scripts/setup_zetaris.py      # creates the table, registers it in Zetaris, checks a round trip
```

Confirm the Nemotron model ID in `litagent/config.py` (`NIGHTSHIFT_MODEL`), then check the connection with one tiny call:

```bash
python scripts/check_nightshift.py   # prints the model that answered and the tokens used
```

### Run the API (main entry point)

```bash
SAMPLE_MODE=true python run.py   # serves POST /run on 0.0.0.0:$PORT (default 8000)
```

```bash
curl -X POST localhost:8000/run -H 'Content-Type: application/json' \
  -d '{"input": {"query": "How do large language models handle low-resource languages?", "effort": "normal"}}'
```

Request: `{"run_id": optional, "input": {"query": str, "effort": "normal|medium|high|ultra", "feedback": optional str}}`

- Success: `{"run_id": ..., "status": "success", "output": {plan, papers, clusters, connections, gaps, paper_map, usage, warnings, ...}}`
- Failure: `{"run_id": ..., "status": "error", "error": {"type": ..., "message": ...}}` (HTTP 422 for bad input, 500 for a failed stage)
- No topic in the query: `status: "success"` with `output.status: "need_topic"` and a friendly `message_to_user`.
- Nothing found: `output.status: "no_results"`, with a `message_to_user` naming the filters used and suggesting broader terms.
- Human reviews never wait for input through `/run`. Both are approved unless you pass `feedback`, which sends that comment back to the Planner once (Human review 1).
- `output.status: "partial"` means the run hit `RUN_TIMEOUT_S` (default 600 s) or its token budget, and returned what it had finished (the reason is in `warnings`).
- `output.usage` gives the run's LLM tokens (`prompt_tokens`, `completion_tokens`, `llm_calls`) against the effort level's `budget`.

### Sample mode

`SAMPLE_MODE=true` runs the whole pipeline with no Zetaris, Nightshift or network access. Papers are real OpenAlex records saved in `data/sample/`; clustering, connections, gaps and grades are placeholders labelled "Sample mode".

### Examples

`input_examples/` holds 4 `/run` requests (normal, medium with feedback, high, and no topic). `output_examples/` holds the matching responses, generated in sample mode:

```bash
SAMPLE_MODE=true python run.py &
for f in input_examples/*.json; do
  curl -s -X POST localhost:8000/run -H 'Content-Type: application/json' -d @$f | python -m json.tool --indent 2 > output_examples/$(basename $f)
done
```

`pytest` checks that each input still produces its output.

### Docker

```bash
docker build -t litagent .
# Optional smaller/faster encoder build:
# docker build --build-arg EMBEDDING_PROFILE=minilm -t litagent-minilm .
docker run --rm -p 8000:8000 -e SAMPLE_MODE=true litagent
# Real run: Zetaris and Nightshift run on your machine, outside the container
docker run --rm -p 8000:8000 --env-file .env litagent
```

Inside the container, `localhost` is the container itself: point `ZETARIS_REST_URL`, `METERLESS_BASE_URL` and `PAPERS_DB_URL` at `host.docker.internal` instead (on Linux, add `--add-host=host.docker.internal:host-gateway`). Dependencies are installed at build time; nothing is downloaded at runtime.

### Command line

```bash
python -m litagent "How do large language models handle low-resource languages?" --effort normal --sample-mode
python -m litagent "TODO your question" --effort medium      # real run, human reviews asked in the terminal
```

To try only the Planner, Retriever and summary on a real question (Nightshift + OpenAlex, no Zetaris):

```bash
python scripts/try_search.py "your question" --effort medium --feedback "only review papers"
```

### Logs

Every run also saves its Meterless World Model to `logs/world-<run_id>.json` (served by `GET /world/{run_id}`). Every run writes `logs/trace-<run_id>.jsonl` (one JSON object per line) and prints readable `[INFO]` lines to stdout. Each event has `timestamp, run_id, agent_name, action, decision, result, next, why`. LLM calls add the model that handled the request and token usage. Zetaris queries, Meterless World Model writes, grades, redos, fallbacks and errors are all logged. Tokens and keys are masked.

### Run time

Every external call has a timeout (model requests 60 s, Zetaris SQL 45 s, OpenAlex 30 s), redos and replans are capped in `litagent/config.py`, and the run stops between stages after `RUN_TIMEOUT_S` (default 600 s). The target is 5–10 minutes; Ultra (~150 papers, full text) may hit the cap and return a partial result.

Each effort level also has a token budget (`EFFORT_PROFILES` in `litagent/config.py`). Every LLM call's tokens are counted, retries included, and once the budget is used up the run stops before the next call and returns a partial result. Nemotron 3.5 Super VL's reasoning counts as completion tokens, so tune the budgets after measuring real runs.

### Tests

```bash
pytest
```

To check the newly implemented real Clusterer independently of the unfinished later agents:

```bash
python scripts/check_threshold.py --output logs/thresholds.json  # saved papers, local CPU only
python scripts/check_clusters.py  # saved papers + real Nightshift labels, grading/redo and World Model
python scripts/check_question_map.py --question "How does LoRA improve low-resource machine translation?"
python scripts/check_clusters.py --grouping-strategy semantic --max-redos 0  # experimental source-grounded grouping
python scripts/review_clusters.py --report logs/clusters-<run_id>.json --output evals/cluster-review.json
python scripts/benchmark_similarity.py --profiles minilm semcse  # prepare both profiles first
```

Question-aware runs now return one optional **main** relevance group and **side** topic groups,
with per-paper source evidence, keywords and alternate-cluster hints. `POST /cluster` provides
a standalone clustering checkpoint; `GET /cluster/{run_id}` retrieves it. See
[main/side behavior, API examples and measured limitations](docs/main-side-clusters.md).

New standalone question maps also support user-selected side moves: numerical candidates,
zero-model previews, cached regeneration into immutable child versions, history and rollback.
See [regeneration endpoints and cache rules](docs/cluster-regeneration.md), or exercise the
offline API flow with `python scripts/check_cluster_regeneration.py`.

Real Connection and Gap finders now run in `/run` and through `POST /cluster/{run_id}/analyze`:
bounded candidate selection, original quotes from both papers, independent verification,
OpenAlex coverage checks, model grades and cached task reuse. Gaps remain provisional research
hypotheses. See [API, real-run audit and limits](docs/grounded-findings.md), or run
`python scripts/check_findings.py --run-id YOUR_MAP_ID`. PDF retrieval and Markovian full-text
reasoning remain pending. Sample mode runs all stages using explicitly labelled placeholders;
it needs neither the encoder weights nor external services.

## Evaluation

TODO: Summarise results from [docs/evaluation.md](docs/evaluation.md). Example: *"On N test topics with published survey papers, cluster agreement was X% and gap recall was Y%."*

## Data

- **Real runs:** paper metadata comes from open scholarly APIs (OpenAlex now; Semantic Scholar and arXiv planned) and is stored in Zetaris for the later agents.
- **Sample mode and tests:** `data/sample/` holds about 60 real OpenAlex papers for each of 3 topics (OpenAlex data is CC0). Recreate or add topics with `scripts/download_sample_data.py`.
- See [data/README.md](data/README.md) for sources and licences. No sensitive or restricted data is stored in this repo.

## Pre-existing components

See [docs/pre-existing-components.md](docs/pre-existing-components.md). Everything else was built during the hackathon (22–27 October 2026).

## Limitations and next steps

- TODO

## Team

| Name | Role |
| --- | --- |
| TODO | TODO |

## Licence

TODO: Choose a licence after checking the Official Rules (section 09, Intellectual property).
