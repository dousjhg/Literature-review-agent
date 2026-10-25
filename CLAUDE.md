# Literature review agent

Open Agent Hackathon 2026, Track 02: The Agent That Can Explain Why.
Pipeline (see ARCHITECTURE.md): Planner -> Retriever -> Clusterer -> Connection finder -> Gap finder,
with grading agents after the last three and two human reviews.

## Conventions

- Agents exchange the Pydantic models in `litagent/schemas.py`. Don't pass free text between agents.
- Every connection and gap must carry Evidence (paper id + verbatim quote).
- All LLM calls go through `litagent/integrations/llm.py` (Meterless Nightshift, OpenAI-compatible; the Nemotron model is set in `config.py`).
- Paper data goes through `litagent/integrations/zetaris.py`; agent findings go to the Meterless World Model through `litagent/integrations/meterless.py` (engine in `litagent/integrations/world_model/`; keep `tests/test_world_model_conformance.py` green). Keep each sponsor integration behind its adapter class.
- Tunable numbers live in `litagent/config.py`, not inside agents.
- Log each important decision with `trace.log(...)`, including why. Traces go to `logs/trace-<run_id>.jsonl`; never log secrets.
- Keep the sample-mode branch in each agent working (`SAMPLE_MODE=true`: no Zetaris, Nightshift or network); `pytest` runs the pipeline in sample mode.
- Every external call has a timeout; a judge's run must never wait for keyboard input.
- Nightshift caches answers by user message + app (not the system prompt) and replays similar messages. `llm.py` gives each run its own cache partition (`x-title: litagent-<run_id>`); keep it, or prompt edits and similar requests get stale answers. The trace's `nightshift_cache` field shows hits.
- Nightshift H-MEM can inject prior answers across app IDs within one scope, even on cache misses. Keep `llm.py`'s fresh `meterless-scope` for every request/repair; `nightshift_memory` logs the receipt's injection count. Producer and grader must not share answer memory.
- Never commit keys. Read secrets from `.env` via environment variables.

## Commands

- `pytest`
- `SAMPLE_MODE=true python run.py` then `curl -X POST localhost:8000/run -H 'Content-Type: application/json' -d @input_examples/01_normal_low_resource_languages.json`
- `python -m litagent "your question" --effort normal --sample-mode`
- Real runs need Zetaris + the paper store: `docker compose up -d`, then `python scripts/setup_zetaris.py` once
- After changing sample-mode output, regenerate `output_examples/` (README, "Examples").
