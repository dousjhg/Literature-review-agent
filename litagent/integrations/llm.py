"""Single entry point for LLM calls, so you can swap providers or models in one place.

Primary provider: Meterless Nightshift, which accepts OpenAI-format requests, so we use the
official OpenAI client pointed at METERLESS_BASE_URL. NVIDIA Nemotron models are connected
through Nightshift (the NVIDIA key lives in Nightshift's Providers tab, never in this app).
The model is set explicitly in config.NIGHTSHIFT_MODEL.

Nightshift caches answers, keyed on the user message and the calling app (not the system prompt),
and replays semantically similar messages. A stale or near-duplicate replay would be wrong for us
(an edited prompt ignored, or one paper pair answered with another pair's quotes), so every request
names its app per run (`x-title: litagent-<run_id>`): each run gets its own cache partition.
The cache status Nightshift reports is logged with every call. Nightshift also retrieves
prior answers through H-MEM across app IDs within one scope: every request/repair gets a
fresh meterless-scope so producers, graders, and previous runs cannot share those answers.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from litagent.config import Settings
from litagent.schemas import TokenUsage
from litagent.trace import current_trace

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
CHARS_PER_TOKEN = 4  # rough estimate when the provider reports no usage


class LLMError(RuntimeError):
    """The model could not be reached or did not return valid JSON after one retry."""


class TokenBudgetExceeded(LLMError):
    """The run has used its effort level's token budget; the orchestrator returns a partial result."""


_current_meter: ContextVar[TokenMeter | None] = ContextVar("current_token_meter", default=None)


class TokenMeter:
    """Counts the tokens of every LLM call in a run (retries included) and enforces the budget.

    The orchestrator opens one per run (`with TokenMeter(budget) as meter:`); complete_json finds it
    through current_meter(), so agents don't have to pass it around.
    """

    def __init__(self, budget: int | None = None):
        self.usage = TokenUsage(budget=budget)
        self._token = None

    def __enter__(self) -> TokenMeter:
        self._token = _current_meter.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _current_meter.reset(self._token)

    def add(self, prompt_tokens: int, completion_tokens: int, estimated: bool = False) -> None:
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.llm_calls += 1
        self.usage.estimated_calls += int(estimated)

    def check(self) -> None:
        """Raise before a call once the budget is used up."""
        if self.usage.budget is not None and self.usage.total >= self.usage.budget:
            raise TokenBudgetExceeded(f"Token budget used: {self.usage.total:,} of {self.usage.budget:,} tokens")


def current_meter() -> TokenMeter | None:
    return _current_meter.get()


def load_prompt(name: str) -> str:
    """Load a prompt file from prompts/, e.g. load_prompt("planner")."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def get_client(settings: Settings) -> OpenAI:
    base_url = os.getenv("METERLESS_BASE_URL")
    api_key = os.getenv("METERLESS_RUNTIME_TOKEN")
    if not base_url or not api_key:
        raise LLMError("Set METERLESS_BASE_URL and METERLESS_RUNTIME_TOKEN (see .env.example), or run with SAMPLE_MODE=true.")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=settings.llm_timeout_s, max_retries=1)


def complete_json(
    system_prompt: str,
    user_message: str,
    output_model: type[T],
    model: str | None = None,
    settings: Settings | None = None,
    agent_name: str = "llm",
    client: OpenAI | None = None,
) -> T:
    """Call the LLM and parse its answer into `output_model`.

    Asks for JSON matching output_model's schema, validates it with Pydantic, and retries
    once with the validation error included. Logs the model name and token usage of every call,
    adds the tokens to the run's TokenMeter, and raises TokenBudgetExceeded before a call once
    the run's budget is used up.
    """
    settings = settings or Settings()
    if settings.sample_mode:
        raise LLMError("complete_json was called in sample mode; agents should use their sample branch.")
    model = model or settings.llm_model
    if model.startswith("TODO"):
        raise LLMError("Set NIGHTSHIFT_MODEL in litagent/config.py to a Nemotron model ID from Nightshift's Providers tab.")
    client = client or get_client(settings)

    schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\nReply with one JSON object only, matching this JSON Schema:\n{schema}",
        },
        {"role": "user", "content": user_message},
    ]

    meter = current_meter()
    for attempt in (1, 2):
        if meter:
            meter.check()
        start = time.monotonic()
        request_scope = f"litagent-call-{uuid.uuid4().hex}"
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_headers={"x-title": _app_name(), "meterless-scope": request_scope},
        )
        text = response.choices[0].message.content or ""
        tokens = _tokens(response.usage, messages, text)
        cache = _cache_status(response)
        memory = _memory_status(response)
        if meter:
            meter.add(tokens["prompt"], tokens["completion"], estimated=tokens.get("estimated", False))
        if memory.get("injected", 0):
            _log(agent_name, f"llm call {output_model.__name__} (attempt {attempt})", "unexpected answer memory",
                 "raise", "Prior answers would contaminate independent agents/evaluations", response.model or model,
                 tokens, start, cache, memory)
            raise LLMError("Nightshift injected prior-answer memory despite a fresh request scope")
        if memory.get("scope") is not None and memory["scope"] != request_scope:
            raise LLMError("Nightshift did not honor the requested answer-memory scope")
        try:
            parsed = output_model.model_validate_json(_clean_json_text(text))
        except ValidationError as e:
            _log(agent_name, f"llm call {output_model.__name__} (attempt {attempt})", "invalid JSON",
                 "retry with error" if attempt == 1 else "raise", str(e), response.model or model, tokens, start, cache, memory)
            if attempt == 2:
                raise LLMError(f"{output_model.__name__}: model output failed validation twice") from e
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Your JSON did not validate:\n{e}\nReply again with corrected JSON only."},
            ]
            continue
        _log(agent_name, f"llm call {output_model.__name__} (attempt {attempt})", "valid JSON",
             "return", "", response.model or model, tokens, start, cache, memory)
        return parsed
    raise AssertionError("unreachable")


def _app_name() -> str:
    """Nightshift partitions its answer cache by app; one partition per run means no replays across runs."""
    trace = current_trace()
    return f"litagent-{trace.run_id}" if trace else "litagent"


def _cache_status(response) -> str | None:
    """Nightshift's compute receipt says whether the answer was computed or replayed ("miss", "exact", ...)."""
    receipt = getattr(response, "meterless", None)
    if isinstance(receipt, dict):
        return (receipt.get("cache") or {}).get("status")
    return None


def _memory_status(response) -> dict:
    """Log only isolation metadata; never read or log retrieved memory contents."""
    metadata = getattr(response, "meterless", None)
    if not isinstance(metadata, dict):
        return {"reported": False}
    receipt = metadata.get("receipt")
    if not isinstance(receipt, dict) or "hmem" not in receipt:
        return {"reported": False, "scope": metadata.get("scope")}
    hmem = receipt["hmem"] or {}
    return {"reported": True, "scope": metadata.get("scope"), "injected": hmem.get("injected", 0)}


def _tokens(usage, messages: list[dict], reply: str) -> dict:
    """Token usage as reported by the provider, or a rough estimate (marked) when it reports none."""
    if usage and usage.prompt_tokens is not None:
        return {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens or 0}
    prompt_chars = sum(len(m["content"]) for m in messages)
    return {"prompt": prompt_chars // CHARS_PER_TOKEN, "completion": len(reply) // CHARS_PER_TOKEN, "estimated": True}


def _clean_json_text(text: str) -> str:
    """Remove reasoning and wrappers around the JSON.

    Nemotron 3 reasons by default and may put its thinking in <think>...</think> before the answer
    (sometimes without the opening tag). Some models also wrap JSON in ```json fences.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text


def _log(agent: str, action: str, decision: str, next_step: str, why: str, model: str, tokens, start: float,
         cache: str | None = None, memory: dict | None = None) -> None:
    if trace := current_trace():
        trace.log(agent, action, decision, next_step=next_step, why=why, model=model, tokens=tokens,
                  duration_ms=round((time.monotonic() - start) * 1000), nightshift_cache=cache,
                  nightshift_memory=memory)
