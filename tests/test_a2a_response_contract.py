import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from google.protobuf.json_format import MessageToDict

from a2a.helpers.proto_helpers import new_text_part
from agentbeats.sync_client import (
    build_send_message_jsonrpc_request,
    create_message_with_parts,
)
from track_2_agent_under_test_codex.car_bench_agent import CARBenchAgentExecutor
from track_2_agent_under_test_codex.codex_client import (
    CodexAppServerClient,
    CodexTokenUsage,
    _parse_usage_limit_retry_at,
    add_token_usage,
)
from track_2_agent_under_test_codex_planner.planner_agent import (
    PlannerExecutorCARBenchAgentExecutor,
)
from track_2_agent_under_test_cerebras.car_bench_agent import (
    CARBenchAgentExecutor as CerebrasCARBenchAgentExecutor,
)
from track_2_agent_under_test_cerebras.litellm_client import (
    CerebrasRateLimitHeaders,
    DEFAULT_CEREBRAS_API_BASE,
    CEREBRAS_INTEGRATION_HEADER,
    LiteLLMCompletionClient,
    LiteLLMSchedulerConfig,
    LiteLLMTemplateError,
    LiteLLMTokenUsage,
    _direct_cerebras_payload,
    _exception_details,
    _extract_cerebras_rate_limit_signal,
    _response_rate_limit_headers,
    estimate_request_tokens,
)
from track_2_agent_under_test_cerebras_planner.planner_agent import (
    PlannerExecutorCARBenchAgentExecutor as CerebrasPlannerExecutor,
)
from evaluator.car_bench_evaluator import _sum_successful_llm_time_seconds
from turn_metrics import (
    AVG_LLM_CALL_TIME_MS,
    COMPLETION_TOKENS,
    COST,
    MODEL,
    NUM_LLM_CALLS,
    NUM_PASSES,
    PROMPT_TOKENS,
    QUOTA_WAIT_TIME_MS,
    THINKING_TOKENS,
)


class A2AResponseContractTest(unittest.TestCase):
    def test_sync_client_serializes_a2a_1_json_field_names(self) -> None:
        message = create_message_with_parts(
            parts=[new_text_part("hello")],
            context_id="ctx-1",
        )

        payload = build_send_message_jsonrpc_request(message)
        serialized_message = payload["params"]["message"]

        self.assertEqual(payload["method"], "SendMessage")
        self.assertIn("messageId", serialized_message)
        self.assertIn("contextId", serialized_message)
        self.assertNotIn("message_id", serialized_message)
        self.assertNotIn("context_id", serialized_message)
        self.assertEqual(serialized_message["parts"], [{"text": "hello"}])

    def test_codex_token_usage_parses_app_server_notification_payload(self) -> None:
        usage = CodexTokenUsage.from_app_server(
            {
                "last": {
                    "inputTokens": 100,
                    "cachedInputTokens": 30,
                    "outputTokens": 12,
                    "reasoningOutputTokens": 4,
                    "totalTokens": 116,
                },
                "total": {
                    "inputTokens": 999,
                    "cachedInputTokens": 333,
                    "outputTokens": 222,
                    "reasoningOutputTokens": 111,
                    "totalTokens": 1665,
                },
            }
        )

        self.assertIsNotNone(usage)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.cached_input_tokens, 30)
        self.assertEqual(usage.output_tokens, 12)
        self.assertEqual(usage.reasoning_output_tokens, 4)
        self.assertEqual(usage.total_tokens, 116)

    def test_codex_token_usage_can_be_aggregated_across_internal_calls(self) -> None:
        usage = add_token_usage(
            CodexTokenUsage(input_tokens=10, output_tokens=3, total_tokens=13),
            CodexTokenUsage(
                input_tokens=20,
                output_tokens=4,
                reasoning_output_tokens=2,
                total_tokens=26,
            ),
        )

        self.assertEqual(usage.input_tokens, 30)
        self.assertEqual(usage.output_tokens, 7)
        self.assertEqual(usage.reasoning_output_tokens, 2)
        self.assertEqual(usage.total_tokens, 39)

    def test_cerebras_litellm_usage_parses_reasoning_tokens(self) -> None:
        usage = LiteLLMTokenUsage.from_litellm(
            SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details=SimpleNamespace(cached_tokens=30),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
            )
        )

        self.assertIsNotNone(usage)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.cached_input_tokens, 30)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.reasoning_output_tokens, 7)
        self.assertEqual(usage.total_tokens, 120)

    def test_cerebras_completion_kwargs_include_provider_settings(self) -> None:
        client = LiteLLMCompletionClient(
            api_base=DEFAULT_CEREBRAS_API_BASE,
            service_tier="priority",
        )

        kwargs = client._completion_kwargs(
            model="cerebras/gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            response_schema={"type": "object", "additionalProperties": False},
            response_schema_name="next_action",
            max_completion_tokens=1024,
            temperature=0.0,
        )

        self.assertEqual(kwargs["custom_llm_provider"], "cerebras")
        self.assertEqual(kwargs["api_base"], DEFAULT_CEREBRAS_API_BASE)
        self.assertEqual(kwargs["service_tier"], "priority")
        self.assertEqual(
            kwargs["extra_headers"][CEREBRAS_INTEGRATION_HEADER],
            "litellm",
        )
        self.assertEqual(kwargs["max_completion_tokens"], 1024)
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])

    def test_cerebras_raw_error_probe_payload_strips_litellm_fields(self) -> None:
        payload = _direct_cerebras_payload(
            {
                "model": "cerebras/gpt-oss-120b",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 1024,
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "custom_llm_provider": "cerebras",
                "api_base": DEFAULT_CEREBRAS_API_BASE,
                "extra_headers": {CEREBRAS_INTEGRATION_HEADER: "litellm"},
            }
        )

        self.assertEqual(payload["model"], "gpt-oss-120b")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertNotIn("custom_llm_provider", payload)
        self.assertNotIn("api_base", payload)
        self.assertNotIn("extra_headers", payload)

    def test_cerebras_error_details_promote_provider_error_body(self) -> None:
        mapped_error = RuntimeError("mapped")
        provider_error = RuntimeError("provider")
        provider_error.body = {
            "message": "Tokens per minute limit exceeded",
            "type": "too_many_tokens_error",
            "param": "quota",
            "code": "token_quota_exceeded",
        }
        mapped_error.__context__ = provider_error

        details = _exception_details(mapped_error)

        self.assertEqual(
            details["provider_error_body"]["code"],
            "token_quota_exceeded",
        )

    def test_cerebras_rate_limit_headers_parse_from_litellm_response(self) -> None:
        response = SimpleNamespace(
            _hidden_params={
                "response_headers": {
                    "x-ratelimit-limit-requests-day": "10",
                    "x-ratelimit-limit-tokens-minute": "30000",
                    "x-ratelimit-remaining-requests-day": "9",
                    "x-ratelimit-remaining-tokens-minute": "1200",
                    "x-ratelimit-reset-requests-day": "3600",
                    "x-ratelimit-reset-tokens-minute": "12.5",
                }
            }
        )

        headers = _response_rate_limit_headers(response)

        self.assertIsNotNone(headers)
        self.assertEqual(headers.limit_requests_day, 10.0)
        self.assertEqual(headers.limit_tokens_minute, 30000.0)
        self.assertEqual(headers.remaining_requests_day, 9.0)
        self.assertEqual(headers.remaining_tokens_minute, 1200.0)
        self.assertEqual(headers.reset_requests_day_seconds, 3600.0)
        self.assertEqual(headers.reset_tokens_minute_seconds, 12.5)

    def test_cerebras_scheduler_estimates_prompt_plus_completion_budget(self) -> None:
        estimated = estimate_request_tokens(
            messages=[{"role": "user", "content": "abcd"}],
            max_completion_tokens=10,
            chars_per_token=1000,
            safety_factor=1.0,
        )

        self.assertEqual(estimated, 11)

    def test_cerebras_scheduler_rejects_request_larger_than_token_bucket(self) -> None:
        client = LiteLLMCompletionClient(
            scheduler_config=LiteLLMSchedulerConfig(
                tokens_per_minute=5,
                token_estimate_chars_per_token=1000,
                token_safety_factor=1.0,
            )
        )

        with self.assertRaises(LiteLLMTemplateError):
            client._scheduler.wait_before_call(
                messages=[{"role": "user", "content": "abcd"}],
                max_completion_tokens=10,
            )

    def test_cerebras_scheduler_can_fail_fast_on_budget_wait(self) -> None:
        client = LiteLLMCompletionClient(
            scheduler_config=LiteLLMSchedulerConfig(
                tokens_per_minute=20,
                token_estimate_chars_per_token=1000,
                token_safety_factor=1.0,
                max_schedule_wait_seconds=0,
            )
        )
        request = [{"role": "user", "content": "abcd"}]

        client._scheduler.wait_before_call(
            messages=request,
            max_completion_tokens=10,
        )

        with self.assertRaises(LiteLLMTemplateError):
            client._scheduler.wait_before_call(
                messages=request,
                max_completion_tokens=10,
            )

    def test_cerebras_header_wait_hint_uses_remaining_tokens(self) -> None:
        client = LiteLLMCompletionClient()
        client._last_rate_limit_headers_by_model["cerebras/gpt-oss-120b"] = (
            CerebrasRateLimitHeaders(
                remaining_tokens_minute=100.0,
                reset_tokens_minute_seconds=12.0,
            ),
            time.perf_counter(),
        )

        hint = client._rate_limit_header_wait_hint(
            model="cerebras/gpt-oss-120b",
            estimated_tokens=150,
        )

        self.assertIsNotNone(hint)
        wait_seconds, wait_reason, _ = hint
        self.assertGreater(wait_seconds, 12.0)
        self.assertLess(wait_seconds, 13.5)
        self.assertEqual(wait_reason, "cerebras_headers_tokens_minute")

    def test_cerebras_record_attempt_reports_previous_estimate_delta(self) -> None:
        client = LiteLLMCompletionClient()
        client._record_attempt(model="cerebras/gpt-oss-120b", estimated_tokens=100)

        state = client._record_attempt(
            model="cerebras/gpt-oss-120b",
            estimated_tokens=140,
        )

        self.assertEqual(state["previous_estimated_request_tokens"], 100)
        self.assertEqual(
            state["estimated_request_token_delta_since_previous"],
            40,
        )

    def test_cerebras_queue_exceeded_signal_uses_configured_backoff(self) -> None:
        signal = _extract_cerebras_rate_limit_signal(
            {
                "status_code": 429,
                "provider_error_body": {
                    "message": "We're experiencing high traffic right now!",
                    "type": "too_many_requests_error",
                    "param": "queue",
                    "code": "queue_exceeded",
                },
                "provider_error_headers": {"x-should-retry": "false"},
            },
            queue_backoff_seconds=45.0,
            retry_buffer_seconds=1.0,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.code, "queue_exceeded")
        self.assertEqual(signal.source, "provider")
        self.assertEqual(signal.schedule_wait_seconds, 45.0)
        self.assertEqual(signal.schedule_reason, "provider_queue_exceeded_backoff")
        self.assertEqual(signal.x_should_retry, "false")

    def test_cerebras_probe_token_quota_signal_uses_retry_after(self) -> None:
        signal = _extract_cerebras_rate_limit_signal(
            {
                "status_code": 429,
                "provider_error_body": {
                    "message": "We're experiencing high traffic right now!",
                    "type": "too_many_requests_error",
                    "param": "queue",
                    "code": "queue_exceeded",
                },
                "provider_error_headers": {"x-should-retry": "false"},
                "direct_cerebras_error": {
                    "status_code": 429,
                    "headers": {"retry-after": "60"},
                    "json": {
                        "message": "Tokens per minute limit exceeded",
                        "type": "too_many_tokens_error",
                        "param": "quota",
                        "code": "token_quota_exceeded",
                    },
                },
            },
            queue_backoff_seconds=0.0,
            retry_buffer_seconds=1.0,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.code, "token_quota_exceeded")
        self.assertEqual(signal.source, "direct_probe")
        self.assertEqual(signal.retry_after_seconds, 60.0)
        self.assertEqual(signal.schedule_wait_seconds, 61.0)

        token_signal = _extract_cerebras_rate_limit_signal(
            {
                "status_code": None,
                "provider_error_body": None,
                "direct_cerebras_error": {
                    "status_code": 429,
                    "headers": {"retry-after": "60"},
                    "json": {
                        "message": "Tokens per minute limit exceeded",
                        "type": "too_many_tokens_error",
                        "param": "quota",
                        "code": "token_quota_exceeded",
                    },
                },
            },
            queue_backoff_seconds=45.0,
            retry_buffer_seconds=1.0,
        )

        self.assertIsNotNone(token_signal)
        self.assertEqual(token_signal.code, "token_quota_exceeded")
        self.assertEqual(token_signal.source, "direct_probe")
        self.assertEqual(token_signal.retry_after_seconds, 60.0)
        self.assertEqual(token_signal.schedule_wait_seconds, 61.0)
        self.assertEqual(token_signal.schedule_reason, "direct_probe_retry_after")

    def test_cerebras_rate_limit_report_writes_current_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = LiteLLMCompletionClient(raw_error_probe=False)
            client.rate_limit_report_dir = Path(tmpdir)
            client._record_attempt(
                model="cerebras/gpt-oss-120b",
                estimated_tokens=1234,
            )
            client._record_successful_call(
                model="cerebras/gpt-oss-120b",
                token_usage=LiteLLMTokenUsage(
                    input_tokens=1000,
                    cached_input_tokens=250,
                    output_tokens=100,
                    reasoning_output_tokens=40,
                    total_tokens=1140,
                ),
            )

            path = client._write_rate_limit_report(
                model="cerebras/gpt-oss-120b",
                messages=[{"role": "user", "content": "hello"}],
                response_schema={"type": "object"},
                response_schema_name="next_action",
                max_completion_tokens=1024,
                estimated_tokens=1234,
                duration_ms=61115.6,
                error_details={
                    "status_code": 429,
                    "provider_error_body": {
                        "code": "queue_exceeded",
                        "param": "queue",
                    },
                },
                rate_limit_signal=_extract_cerebras_rate_limit_signal(
                    {
                        "status_code": 429,
                        "provider_error_body": {
                            "code": "queue_exceeded",
                            "param": "queue",
                        },
                    },
                    queue_backoff_seconds=60.0,
                    retry_buffer_seconds=1.0,
                ),
            )

            self.assertIsNotNone(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["event"], "cerebras_rate_limit")
            self.assertEqual(payload["model"], "cerebras/gpt-oss-120b")
            self.assertEqual(payload["successful_litellm_calls"], 1)
            self.assertEqual(payload["attempted_litellm_calls"], 1)
            self.assertEqual(payload["tokens_consumed"]["input_tokens"], 1000)
            self.assertEqual(payload["tokens_consumed"]["output_tokens"], 100)
            self.assertEqual(payload["estimated_request_tokens_scheduled"], 1234)
            self.assertEqual(
                payload["current_call"]["duration_ms_until_error"],
                61115.6,
            )
            self.assertEqual(
                payload["rate_limit_signal"]["code"],
                "queue_exceeded",
            )
            self.assertEqual(payload["wait_seconds"], 60.0)
            self.assertEqual(payload["scheduler_config"]["min_interval_seconds"], 0.0)

    def test_evaluator_successful_llm_time_uses_agent_state_latency(self) -> None:
        result = SimpleNamespace(
            info={"total_llm_induced_latency_ms": 1234.5},
        )

        value = _sum_successful_llm_time_seconds({"base": [result]})

        self.assertEqual(value, 1.2345)

    def test_codex_respond_action_returns_text_part(self) -> None:
        parts, history_message = CARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "respond",
                "content": "Done.",
                "tool_calls": [],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "text")
        self.assertEqual(parts[0].text, "Done.")
        self.assertEqual(history_message, {"role": "assistant", "content": "Done."})

    def test_codex_tool_action_returns_tool_calls_data_part(self) -> None:
        parts, history_message = CARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "tool_calls",
                "content": "",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "data")
        data = MessageToDict(parts[0].data)
        self.assertEqual(
            data,
            {
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ]
            },
        )
        self.assertIsNone(history_message["content"])
        self.assertEqual(
            history_message["tool_calls"][0]["function"]["name"],
            "open_close_sunshade",
        )

    def test_cerebras_respond_action_returns_text_part(self) -> None:
        parts, history_message = CerebrasCARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "respond",
                "content": "Done.",
                "tool_calls": [],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "text")
        self.assertEqual(parts[0].text, "Done.")
        self.assertEqual(history_message, {"role": "assistant", "content": "Done."})

    def test_cerebras_tool_action_returns_tool_calls_data_part(self) -> None:
        parts, history_message = CerebrasCARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "tool_calls",
                "content": "",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "data")
        data = MessageToDict(parts[0].data)
        self.assertEqual(
            data,
            {
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ]
            },
        )
        self.assertIsNone(history_message["content"])
        self.assertEqual(
            history_message["tool_calls"][0]["function"]["name"],
            "open_close_sunshade",
        )

    def test_codex_turn_metrics_are_public_metadata_shape(self) -> None:
        executor = CARBenchAgentExecutor(model="gpt-5.3-codex-spark")

        executor._record_turn_metrics(
            "ctx",
            100.0,
            token_usage=CodexTokenUsage(
                input_tokens=1200,
                cached_input_tokens=400,
                output_tokens=80,
                reasoning_output_tokens=25,
                total_tokens=1305,
            ),
            quota_wait_ms=7000.0,
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[PROMPT_TOKENS], 1200)
        self.assertEqual(metrics[COMPLETION_TOKENS], 80)
        self.assertEqual(metrics[THINKING_TOKENS], 25)
        self.assertEqual(metrics[COST], 0.0)
        self.assertEqual(metrics[MODEL], "gpt-5.3-codex-spark")
        self.assertEqual(metrics[NUM_LLM_CALLS], 1)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 100.0)
        self.assertEqual(metrics[NUM_PASSES], 1)
        self.assertEqual(metrics[QUOTA_WAIT_TIME_MS], 7000.0)
        self.assertNotIn("_total_llm_time_ms", metrics)

    def test_cerebras_turn_metrics_are_public_metadata_shape(self) -> None:
        executor = CerebrasCARBenchAgentExecutor(model="cerebras/gpt-oss-120b")

        executor._record_turn_metrics(
            "ctx",
            100.0,
            token_usage=LiteLLMTokenUsage(
                input_tokens=1200,
                cached_input_tokens=400,
                output_tokens=80,
                reasoning_output_tokens=25,
                total_tokens=1305,
            ),
            cost=0.25,
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[PROMPT_TOKENS], 1200)
        self.assertEqual(metrics[COMPLETION_TOKENS], 80)
        self.assertEqual(metrics[THINKING_TOKENS], 25)
        self.assertEqual(metrics[COST], 0.25)
        self.assertEqual(metrics[MODEL], "cerebras/gpt-oss-120b")
        self.assertEqual(metrics[NUM_LLM_CALLS], 1)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 100.0)
        self.assertEqual(metrics[NUM_PASSES], 1)
        self.assertEqual(metrics[QUOTA_WAIT_TIME_MS], 0.0)
        self.assertNotIn("_total_llm_time_ms", metrics)

    def test_codex_usage_limit_retry_time_is_parsed(self) -> None:
        retry_at = _parse_usage_limit_retry_at(
            "You've hit your usage limit for GPT-5.3-Codex-Spark. "
            "Switch to another model now, or try again at 5:39 PM.",
            now=datetime(2026, 5, 28, 14, 29, tzinfo=timezone.utc),
        )

        self.assertEqual(
            retry_at,
            datetime(2026, 5, 28, 17, 39, tzinfo=timezone.utc),
        )

    def test_codex_usage_limit_same_minute_retry_does_not_roll_to_tomorrow(self) -> None:
        retry_at = _parse_usage_limit_retry_at(
            "You've hit your usage limit for GPT-5.3-Codex-Spark. "
            "Switch to another model now, or try again at 10:11 PM.",
            now=datetime(2026, 6, 2, 22, 11, 18, tzinfo=timezone.utc),
        )

        self.assertEqual(
            retry_at,
            datetime(2026, 6, 2, 22, 11, tzinfo=timezone.utc),
        )

    def test_codex_usage_limit_report_writes_current_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = CodexAppServerClient(model="gpt-5.3-codex-spark")
            client.rate_limit_report_dir = Path(tmpdir)
            client._record_successful_turn(
                model="gpt-5.3-codex-spark",
                token_usage=CodexTokenUsage(
                    input_tokens=1000,
                    cached_input_tokens=250,
                    output_tokens=100,
                    reasoning_output_tokens=40,
                    total_tokens=1140,
                ),
            )

            path = client._write_usage_limit_report(
                error_message="usage limit",
                raw_error={
                    "message": "usage limit",
                    "code": "model_usage_limit",
                },
                raw_error_source="turn.error",
                raw_payload={
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {
                            "id": "turn-1",
                            "status": "failed",
                            "error": {
                                "message": "usage limit",
                                "code": "model_usage_limit",
                            },
                        },
                    },
                },
                retry_at=datetime(2026, 5, 28, 17, 39, tzinfo=timezone.utc),
                wait_seconds=120.0,
                model="gpt-5.3-codex-spark",
                reasoning_effort="medium",
                prompt="hello",
                output_schema={"name": "next_action"},
                quota_retries=1,
            )

            self.assertIsNotNone(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["event"], "codex_usage_limit")
            self.assertGreaterEqual(
                payload["wall_time_until_rate_limit_seconds"],
                0.0,
            )
            self.assertEqual(payload["wait_seconds"], 120.0)
            self.assertIn("retry_with_buffer_at", payload)
            self.assertEqual(payload["raw_error_source"], "turn.error")
            self.assertEqual(payload["raw_error"]["message"], "usage limit")
            self.assertEqual(payload["raw_error"]["code"], "model_usage_limit")
            self.assertEqual(payload["raw_payload"]["method"], "turn/completed")
            self.assertEqual(
                payload["raw_payload"]["params"]["turn"]["error"]["message"],
                "usage limit",
            )
            self.assertIsNone(payload["previous_retry_at"])
            self.assertIsNone(
                payload["wall_time_since_previous_retry_at_seconds"]
            )
            self.assertEqual(payload["successful_codex_calls"], 1)
            self.assertEqual(payload["tokens_consumed"]["input_tokens"], 1000)
            self.assertEqual(payload["tokens_consumed"]["output_tokens"], 100)
            self.assertEqual(
                payload["tokens_consumed_by_model"]["gpt-5.3-codex-spark"][
                    "reasoning_output_tokens"
                ],
                40,
            )
            self.assertEqual(payload["current_call"]["prompt_chars"], 5)

            client._previous_usage_limit_retry_at = (
                datetime.now().astimezone() - timedelta(seconds=42)
            )
            second_path = client._write_usage_limit_report(
                error_message="usage limit again",
                retry_at=datetime(2026, 5, 28, 19, 0, tzinfo=timezone.utc),
                wait_seconds=60.0,
                model="gpt-5.3-codex-spark",
                reasoning_effort="medium",
                prompt="hello again",
                output_schema={"name": "next_action"},
                quota_retries=2,
            )
            second_payload = json.loads(second_path.read_text())
            self.assertIsNotNone(second_payload["previous_retry_at"])
            self.assertGreaterEqual(
                second_payload["wall_time_since_previous_retry_at_seconds"],
                41.0,
            )
            self.assertLessEqual(
                second_payload["wall_time_since_previous_retry_at_seconds"],
                45.0,
            )

    def test_planner_executor_metrics_report_internal_passes(self) -> None:
        executor = PlannerExecutorCARBenchAgentExecutor(
            planner_model="gpt-5.5",
            executor_model="gpt-5.3-codex-spark",
        )
        executor._last_internal_call_count = 2

        executor._record_turn_metrics(
            "ctx",
            300.0,
            token_usage=CodexTokenUsage(
                input_tokens=3000,
                output_tokens=200,
                reasoning_output_tokens=75,
                total_tokens=3275,
            ),
            quota_wait_ms=9000.0,
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[MODEL], "gpt-5.5->gpt-5.3-codex-spark")
        self.assertEqual(metrics[PROMPT_TOKENS], 3000)
        self.assertEqual(metrics[COMPLETION_TOKENS], 200)
        self.assertEqual(metrics[THINKING_TOKENS], 75)
        self.assertEqual(metrics[NUM_LLM_CALLS], 2)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 150.0)
        self.assertEqual(metrics[NUM_PASSES], 2)
        self.assertEqual(metrics[QUOTA_WAIT_TIME_MS], 9000.0)

    def test_cerebras_planner_executor_metrics_report_internal_passes(self) -> None:
        executor = CerebrasPlannerExecutor(
            planner_model="azure/gpt-5.5",
            executor_model="cerebras/gpt-oss-120b",
            api_base=DEFAULT_CEREBRAS_API_BASE,
        )
        executor._last_internal_call_count = 2

        executor._record_turn_metrics(
            "ctx",
            300.0,
            token_usage=LiteLLMTokenUsage(
                input_tokens=3000,
                output_tokens=200,
                reasoning_output_tokens=75,
                total_tokens=3275,
            ),
            cost=0.5,
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[MODEL], "azure/gpt-5.5->cerebras/gpt-oss-120b")
        self.assertEqual(metrics[PROMPT_TOKENS], 3000)
        self.assertEqual(metrics[COMPLETION_TOKENS], 200)
        self.assertEqual(metrics[THINKING_TOKENS], 75)
        self.assertEqual(metrics[COST], 0.5)
        self.assertEqual(metrics[NUM_LLM_CALLS], 2)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 150.0)
        self.assertEqual(metrics[NUM_PASSES], 2)
        self.assertEqual(metrics[QUOTA_WAIT_TIME_MS], 0.0)


if __name__ == "__main__":
    unittest.main()
