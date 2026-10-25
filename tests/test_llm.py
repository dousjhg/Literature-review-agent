"""complete_json with the OpenAI client mocked: never calls Nightshift in tests."""

import json

import pytest

from litagent.config import Settings
from litagent.integrations.llm import LLMError, TokenBudgetExceeded, TokenMeter, complete_json
from litagent.schemas import HumanFeedback
from litagent.trace import Trace
from tests.fakes import FakeClient


def real_settings() -> Settings:
    s = Settings()
    s.sample_mode = False
    s.llm_model = "nvidia/nemotron-test"
    return s


def test_valid_json_is_parsed_and_logged(tmp_path):
    client = FakeClient(['{"approved": true}'])
    with Trace(str(tmp_path)) as trace:
        out = complete_json("sys", "user", HumanFeedback, settings=real_settings(), agent_name="planner", client=client)
    assert out == HumanFeedback(approved=True)
    assert client.calls[0]["model"] == "nvidia/nemotron-test"
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    [event] = trace.events
    assert event["agent_name"] == "planner"
    assert event["model"] == "nvidia/nemotron-test"
    assert event["tokens"] == {"prompt": 10, "completion": 5}


def test_retries_once_with_validation_error():
    client = FakeClient(['{"approved": "maybe"}', '```json\n{"approved": false, "comment": "x"}\n```'])
    out = complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    assert out == HumanFeedback(approved=False, comment="x")
    retry_messages = client.calls[1]["messages"]
    assert "did not validate" in retry_messages[-1]["content"]


@pytest.mark.parametrize("reply", [
    '<think>The user wants approval. {"approved": false}</think>\n{"approved": true}',
    'Reasoning without an opening tag...</think>{"approved": true}',
])
def test_reasoning_text_is_removed_before_parsing(reply):
    client = FakeClient([reply])
    out = complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    assert out == HumanFeedback(approved=True)
    assert len(client.calls) == 1  # parsed first time, no retry needed


def test_gives_up_after_second_invalid_reply():
    client = FakeClient(["not json", json.dumps({"approved": "maybe"})])
    with pytest.raises(LLMError):
        complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)


def test_refuses_in_sample_mode_and_without_model():
    s = real_settings()
    s.sample_mode = True
    with pytest.raises(LLMError):
        complete_json("sys", "user", HumanFeedback, settings=s, client=FakeClient([]))
    s = real_settings()
    s.llm_model = "TODO-nemotron-model-id"
    with pytest.raises(LLMError):
        complete_json("sys", "user", HumanFeedback, settings=s, client=FakeClient([]))


def test_meter_counts_every_attempt_including_retries():
    client = FakeClient(['{"approved": "maybe"}', '{"approved": true}'])
    with TokenMeter(budget=1000) as meter:
        complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    assert meter.usage.llm_calls == 2
    assert (meter.usage.prompt_tokens, meter.usage.completion_tokens) == (20, 10)
    assert meter.usage.estimated_calls == 0


def test_tokens_are_estimated_when_provider_reports_none(tmp_path):
    client = FakeClient(['{"approved": true}'], report_usage=False)
    with Trace(str(tmp_path)) as trace, TokenMeter() as meter:
        complete_json("sys", "user " * 100, HumanFeedback, settings=real_settings(), client=client)
    assert meter.usage.estimated_calls == 1
    assert meter.usage.prompt_tokens > 100  # schema + 500 characters of user message, at ~4 characters per token
    assert trace.events[-1]["tokens"]["estimated"] is True


def test_no_call_once_budget_is_used_up():
    client = FakeClient(['{"approved": true}'])
    with TokenMeter(budget=100) as meter:
        meter.add(80, 20)
        with pytest.raises(TokenBudgetExceeded):
            complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    assert client.calls == []


def test_each_run_gets_its_own_nightshift_cache_partition(tmp_path):
    client = FakeClient(['{"approved": true}'])
    with Trace(str(tmp_path), run_id="run-42"):
        complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    headers = client.calls[0]["extra_headers"]
    assert headers["x-title"] == "litagent-run-42"
    assert headers["meterless-scope"].startswith("litagent-call-")


def test_every_call_and_json_repair_has_isolated_answer_memory():
    client = FakeClient(['not json', '{"approved": true}', '{"approved": false}'])
    for _ in range(2):
        complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    scopes = [call["extra_headers"]["meterless-scope"] for call in client.calls]
    assert len(scopes) == len(set(scopes)) == 3


def test_unexpected_memory_injection_is_rejected_and_logged(tmp_path):
    client = FakeClient(['{"approved": true}'])
    client.receipt = {"scope": "ignored-scope", "receipt": {"hmem": {"injected": 2}}}
    with Trace(str(tmp_path)) as trace:
        with pytest.raises(LLMError, match="prior-answer memory"):
            complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    assert trace.events[-1]["nightshift_memory"]["injected"] == 2


def test_gateway_scope_mismatch_is_rejected_even_with_no_retrieval():
    client = FakeClient(['{"approved": true}'])
    client.receipt = {"scope": "user", "receipt": {"hmem": None}}
    with pytest.raises(LLMError, match="did not honor"):
        complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)


def test_nightshift_cache_status_is_logged(tmp_path):
    client = FakeClient(['{"approved": true}'])
    client.receipt = {"cache": {"status": "exact"}}
    with Trace(str(tmp_path)) as trace:
        complete_json("sys", "user", HumanFeedback, settings=real_settings(), client=client)
    assert trace.events[-1]["nightshift_cache"] == "exact"
