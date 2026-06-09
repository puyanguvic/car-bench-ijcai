"""Small LiteLLM client for Track 2 Cerebras template agents."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from litellm import completion


DEFAULT_CEREBRAS_API_BASE = "https://api.cerebras.ai/v1"
DEFAULT_EXECUTOR_MODEL = "cerebras/gpt-oss-120b"
CEREBRAS_INTEGRATION_HEADER = "X-Cerebras-3rd-Party-Integration"
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4.0
DEFAULT_TOKEN_SAFETY_FACTOR = 1.1
DEFAULT_RAW_ERROR_PROBE_TIMEOUT_SECONDS = 30.0
DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECONDS = 60.0
DEFAULT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS = 1.0
CEREBRAS_RATE_LIMIT_HEADER_NAMES = (
    "x-ratelimit-limit-requests-day",
    "x-ratelimit-limit-tokens-minute",
    "x-ratelimit-remaining-requests-day",
    "x-ratelimit-remaining-tokens-minute",
    "x-ratelimit-reset-requests-day",
    "x-ratelimit-reset-tokens-minute",
)


class LiteLLMTemplateError(RuntimeError):
    """Raised when a LiteLLM-backed template call fails."""


class MalformedModelResponseError(LiteLLMTemplateError):
    """Raised when the model output cannot be parsed as the expected JSON."""


@dataclass(frozen=True)
class LiteLLMSchedulerConfig:
    """Process-local pre-call pacing knobs for request and token quotas."""

    min_interval_seconds: float = 0.0
    requests_per_minute: float | None = None
    requests_per_hour: float | None = None
    requests_per_day: float | None = None
    tokens_per_minute: float | None = None
    tokens_per_hour: float | None = None
    tokens_per_day: float | None = None
    token_estimate_chars_per_token: float = DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN
    token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR
    max_schedule_wait_seconds: float | None = None

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "min_interval_seconds": self.min_interval_seconds,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_hour": self.tokens_per_hour,
            "tokens_per_day": self.tokens_per_day,
            "token_estimate_chars_per_token": self.token_estimate_chars_per_token,
            "token_safety_factor": self.token_safety_factor,
            "max_schedule_wait_seconds": self.max_schedule_wait_seconds,
        }


@dataclass
class LiteLLMTokenUsage:
    """Token usage reported by LiteLLM for one completion call."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_litellm(cls, usage: Any) -> "LiteLLMTokenUsage | None":
        if usage is None:
            return None
        completion_details = _get_field(usage, "completion_tokens_details")
        prompt_details = _get_field(usage, "prompt_tokens_details")
        return cls(
            input_tokens=_safe_int(_get_field(usage, "prompt_tokens")),
            cached_input_tokens=_safe_int(
                _get_field(prompt_details, "cached_tokens")
            ),
            output_tokens=_safe_int(_get_field(usage, "completion_tokens")),
            reasoning_output_tokens=_safe_int(
                _get_field(completion_details, "reasoning_tokens")
            ),
            total_tokens=_safe_int(_get_field(usage, "total_tokens")),
        )

    def __bool__(self) -> bool:
        return any(
            (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_output_tokens,
                self.total_tokens,
            )
        )


@dataclass(frozen=True)
class CerebrasRateLimitHeaders:
    """Cerebras rate-limit headers from a provider response."""

    limit_requests_day: float | None = None
    limit_tokens_minute: float | None = None
    remaining_requests_day: float | None = None
    remaining_tokens_minute: float | None = None
    reset_requests_day_seconds: float | None = None
    reset_tokens_minute_seconds: float | None = None
    raw_headers: dict[str, str] | None = None

    @classmethod
    def from_headers(cls, headers: Any) -> "CerebrasRateLimitHeaders | None":
        headers_dict = _headers_dict(headers)
        if not headers_dict:
            return None
        relevant_headers = {
            name: _header_value(headers_dict, name)
            for name in CEREBRAS_RATE_LIMIT_HEADER_NAMES
        }
        if not any(value is not None for value in relevant_headers.values()):
            return None
        return cls(
            limit_requests_day=_safe_float_or_none(
                relevant_headers["x-ratelimit-limit-requests-day"]
            ),
            limit_tokens_minute=_safe_float_or_none(
                relevant_headers["x-ratelimit-limit-tokens-minute"]
            ),
            remaining_requests_day=_safe_float_or_none(
                relevant_headers["x-ratelimit-remaining-requests-day"]
            ),
            remaining_tokens_minute=_safe_float_or_none(
                relevant_headers["x-ratelimit-remaining-tokens-minute"]
            ),
            reset_requests_day_seconds=_safe_float_or_none(
                relevant_headers["x-ratelimit-reset-requests-day"]
            ),
            reset_tokens_minute_seconds=_safe_float_or_none(
                relevant_headers["x-ratelimit-reset-tokens-minute"]
            ),
            raw_headers={
                key: value
                for key, value in relevant_headers.items()
                if value is not None
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit_requests_day": self.limit_requests_day,
            "limit_tokens_minute": self.limit_tokens_minute,
            "remaining_requests_day": self.remaining_requests_day,
            "remaining_tokens_minute": self.remaining_tokens_minute,
            "reset_requests_day_seconds": self.reset_requests_day_seconds,
            "reset_tokens_minute_seconds": self.reset_tokens_minute_seconds,
            "raw_headers": self.raw_headers,
        }


@dataclass
class LiteLLMCallResult:
    """Final model text, duration, token usage, and provider cost."""

    text: str
    duration_ms: float
    model: str
    token_usage: LiteLLMTokenUsage | None = None
    cost: float = 0.0
    estimated_request_tokens: int = 0
    rate_limit_headers: CerebrasRateLimitHeaders | None = None


@dataclass(frozen=True)
class CerebrasRateLimitSignal:
    """Provider-visible rate-limit signal used for reports and local pacing."""

    code: str | None
    type: str | None
    param: str | None
    message: str | None
    source: str
    retry_after_seconds: float | None = None
    schedule_wait_seconds: float | None = None
    schedule_reason: str | None = None
    x_should_retry: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "type": self.type,
            "param": self.param,
            "message": self.message,
            "source": self.source,
            "retry_after_seconds": self.retry_after_seconds,
            "schedule_wait_seconds": self.schedule_wait_seconds,
            "schedule_reason": self.schedule_reason,
            "x_should_retry": self.x_should_retry,
        }


def add_token_usage(
    left: LiteLLMTokenUsage | None,
    right: LiteLLMTokenUsage | None,
) -> LiteLLMTokenUsage | None:
    """Return the sum of two optional token usage records."""

    if left is None:
        return right
    if right is None:
        return left
    return LiteLLMTokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_output_tokens=(
            left.reasoning_output_tokens + right.reasoning_output_tokens
        ),
        total_tokens=left.total_tokens + right.total_tokens,
    )


class LiteLLMCompletionClient:
    """Synchronous LiteLLM completion wrapper with local call scheduling."""

    def __init__(
        self,
        *,
        api_base: str = DEFAULT_CEREBRAS_API_BASE,
        service_tier: str | None = None,
        min_interval_seconds: float = 0.0,
        scheduler_config: LiteLLMSchedulerConfig | None = None,
        raw_error_probe: bool | None = None,
        raw_error_probe_timeout_seconds: float | None = None,
        logger: Any | None = None,
    ) -> None:
        self.api_base = api_base
        self.service_tier = service_tier.strip() if service_tier else None
        self.logger = logger
        self.raw_error_probe = (
            _env_bool("TRACK2_CEREBRAS_RAW_ERROR_PROBE", False)
            if raw_error_probe is None
            else raw_error_probe
        )
        self.raw_error_probe_timeout_seconds = (
            _env_float(
                "TRACK2_CEREBRAS_RAW_ERROR_PROBE_TIMEOUT_SECONDS",
                DEFAULT_RAW_ERROR_PROBE_TIMEOUT_SECONDS,
            )
            if raw_error_probe_timeout_seconds is None
            else max(0.0, raw_error_probe_timeout_seconds)
        )
        self.queue_backoff_seconds = _env_float(
            "TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS",
            DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECONDS,
        )
        self.rate_limit_retry_buffer_seconds = _env_float(
            "TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS",
            DEFAULT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS,
        )
        self.rate_limit_report_dir = Path(
            os.getenv(
                "CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR",
                os.getenv(
                    "CAR_BENCH_RATE_LIMIT_REPORT_DIR",
                    "/tmp/car-bench-rate-limit-reports",
                ),
            )
        )
        self._scheduler = LiteLLMCallScheduler(
            scheduler_config
            or LiteLLMSchedulerConfig(min_interval_seconds=min_interval_seconds),
            logger=logger,
        )
        self._session_started_at = datetime.now().astimezone()
        self._session_started_monotonic = time.perf_counter()
        self._metrics_lock = threading.Lock()
        self._successful_calls = 0
        self._successful_calls_by_model: dict[str, int] = {}
        self._attempted_calls = 0
        self._attempted_calls_by_model: dict[str, int] = {}
        self._total_token_usage = LiteLLMTokenUsage()
        self._token_usage_by_model: dict[str, LiteLLMTokenUsage] = {}
        self._estimated_request_tokens = 0
        self._estimated_request_tokens_by_model: dict[str, int] = {}
        self._last_estimated_request_tokens_by_model: dict[str, int] = {}
        self._last_successful_token_usage_by_model: dict[str, LiteLLMTokenUsage] = {}
        self._last_rate_limit_headers_by_model: dict[
            str,
            tuple[CerebrasRateLimitHeaders, float],
        ] = {}
        self._previous_rate_limit_at: datetime | None = None
        self._previous_rate_limit_retry_at: datetime | None = None

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        temperature: float | None,
    ) -> LiteLLMCallResult:
        estimated_tokens = self._scheduler.estimate_request_tokens(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        header_wait_hint = self._rate_limit_header_wait_hint(
            model=model,
            estimated_tokens=estimated_tokens,
        )
        if header_wait_hint is not None:
            wait_seconds, wait_reason, header_snapshot = header_wait_hint
            self._scheduler.apply_external_pause(
                wait_seconds,
                reason=wait_reason,
            )
            if self.logger:
                self.logger.info(
                    "Cerebras rate-limit headers suggest waiting before next request",
                    wait_seconds=round(wait_seconds, 3),
                    resume_at=_format_future_time(wait_seconds),
                    wait_reason=wait_reason,
                    estimated_request_tokens=estimated_tokens,
                    previous_rate_limit_headers=header_snapshot.as_dict(),
                )
        estimated_tokens = self._scheduler.wait_before_call(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            estimated_tokens=estimated_tokens,
        )
        previous_request_state = self._record_attempt(
            model=model,
            estimated_tokens=estimated_tokens,
        )
        kwargs = self._completion_kwargs(
            model=model,
            messages=messages,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
        if self.logger:
            self.logger.info(
                "Sending LiteLLM completion request",
                model=model,
                estimated_request_tokens=estimated_tokens,
                previous_estimated_request_tokens=previous_request_state[
                    "previous_estimated_request_tokens"
                ],
                estimated_request_token_delta_since_previous=(
                    previous_request_state[
                        "estimated_request_token_delta_since_previous"
                    ]
                ),
                previous_successful_token_usage=previous_request_state[
                    "previous_successful_token_usage"
                ],
                previous_rate_limit_headers=previous_request_state[
                    "previous_rate_limit_headers"
                ],
                projected_remaining_tokens_minute_after_request=(
                    previous_request_state[
                        "projected_remaining_tokens_minute_after_request"
                    ]
                ),
                max_completion_tokens=max_completion_tokens,
                has_output_schema=response_schema is not None,
            )
        start = time.perf_counter()
        try:
            response = completion(**kwargs)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            direct_cerebras_error = None
            if _is_cerebras_model(model) and self.raw_error_probe:
                direct_cerebras_error = _probe_direct_cerebras_error_response(
                    completion_kwargs=kwargs,
                    api_base=self.api_base,
                    timeout_seconds=self.raw_error_probe_timeout_seconds,
                )
            error_details, rate_limit_signal, report_path = (
                self._handle_completion_error(
                    exc=exc,
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    response_schema_name=response_schema_name,
                    max_completion_tokens=max_completion_tokens,
                    estimated_tokens=estimated_tokens,
                    duration_ms=duration_ms,
                    direct_cerebras_error=direct_cerebras_error,
                )
            )
            _log_cerebras_error(
                exc,
                logger=self.logger,
                details=error_details,
                rate_limit_signal=rate_limit_signal,
                report_path=report_path,
            )
            raise LiteLLMTemplateError(
                f"LiteLLM completion failed for {model}: {exc}"
            ) from exc

        duration_ms = (time.perf_counter() - start) * 1000.0
        message = response.choices[0].message
        usage = LiteLLMTokenUsage.from_litellm(getattr(response, "usage", None))
        rate_limit_headers = _response_rate_limit_headers(response)
        self._record_successful_call(
            model=model,
            token_usage=usage,
            rate_limit_headers=rate_limit_headers,
        )
        if self.logger and rate_limit_headers is not None:
            self.logger.info(
                "Cerebras rate-limit headers observed",
                model=model,
                estimated_request_tokens=estimated_tokens,
                token_usage=_token_usage_to_dict(usage),
                rate_limit_headers=rate_limit_headers.as_dict(),
            )
        return LiteLLMCallResult(
            text=_message_content(message),
            duration_ms=duration_ms,
            model=getattr(response, "model", None) or model,
            token_usage=usage,
            cost=_response_cost(response),
            estimated_request_tokens=estimated_tokens,
            rate_limit_headers=rate_limit_headers,
        )

    def _completion_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        temperature: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema_name or "car_bench_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        if _is_cerebras_model(model):
            kwargs["api_base"] = self.api_base
            kwargs["custom_llm_provider"] = "cerebras"
            kwargs["extra_headers"] = {
                CEREBRAS_INTEGRATION_HEADER: "litellm",
            }
            if self.service_tier:
                kwargs["service_tier"] = self.service_tier
        return kwargs

    def _handle_completion_error(
        self,
        *,
        exc: BaseException,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        estimated_tokens: int,
        duration_ms: float,
        direct_cerebras_error: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], CerebrasRateLimitSignal | None, Path | None]:
        details = _exception_details(exc, direct_cerebras_error=direct_cerebras_error)
        signal = _extract_cerebras_rate_limit_signal(
            details,
            queue_backoff_seconds=self.queue_backoff_seconds,
            retry_buffer_seconds=self.rate_limit_retry_buffer_seconds,
        )
        if signal and signal.schedule_wait_seconds is not None:
            self._scheduler.apply_external_pause(
                signal.schedule_wait_seconds,
                reason=signal.schedule_reason or signal.code or "cerebras_rate_limit",
            )
        report_path = None
        if signal is not None:
            report_path = self._write_rate_limit_report(
                model=model,
                messages=messages,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                max_completion_tokens=max_completion_tokens,
                estimated_tokens=estimated_tokens,
                duration_ms=duration_ms,
                error_details=details,
                rate_limit_signal=signal,
            )
        return details, signal, report_path

    def _rate_limit_header_wait_hint(
        self,
        *,
        model: str,
        estimated_tokens: int,
    ) -> tuple[float, str, CerebrasRateLimitHeaders] | None:
        with self._metrics_lock:
            snapshot = self._last_rate_limit_headers_by_model.get(model)
        if snapshot is None:
            return None

        rate_limit_headers, observed_at = snapshot
        elapsed_seconds = max(0.0, time.perf_counter() - observed_at)
        waits: list[tuple[float, str]] = []
        if (
            rate_limit_headers.remaining_tokens_minute is not None
            and rate_limit_headers.remaining_tokens_minute < estimated_tokens
            and rate_limit_headers.reset_tokens_minute_seconds is not None
        ):
            waits.append(
                (
                    max(
                        0.0,
                        rate_limit_headers.reset_tokens_minute_seconds
                        - elapsed_seconds
                        + max(0.0, self.rate_limit_retry_buffer_seconds),
                    ),
                    "cerebras_headers_tokens_minute",
                )
            )
        if (
            rate_limit_headers.remaining_requests_day is not None
            and rate_limit_headers.remaining_requests_day < 1
            and rate_limit_headers.reset_requests_day_seconds is not None
        ):
            waits.append(
                (
                    max(
                        0.0,
                        rate_limit_headers.reset_requests_day_seconds
                        - elapsed_seconds
                        + max(0.0, self.rate_limit_retry_buffer_seconds),
                    ),
                    "cerebras_headers_requests_day",
                )
            )
        if not waits:
            return None
        wait_seconds, wait_reason = max(waits, key=lambda item: item[0])
        if wait_seconds <= 0:
            return None
        return wait_seconds, wait_reason, rate_limit_headers

    def _record_attempt(
        self,
        *,
        model: str,
        estimated_tokens: int,
    ) -> dict[str, Any]:
        with self._metrics_lock:
            previous_estimated_tokens = self._last_estimated_request_tokens_by_model.get(
                model
            )
            previous_successful_usage = self._last_successful_token_usage_by_model.get(
                model
            )
            previous_headers_snapshot = self._last_rate_limit_headers_by_model.get(
                model
            )
            self._attempted_calls += 1
            self._attempted_calls_by_model[model] = (
                self._attempted_calls_by_model.get(model, 0) + 1
            )
            self._estimated_request_tokens += estimated_tokens
            self._estimated_request_tokens_by_model[model] = (
                self._estimated_request_tokens_by_model.get(model, 0)
                + estimated_tokens
            )
            self._last_estimated_request_tokens_by_model[model] = estimated_tokens

        previous_headers = (
            previous_headers_snapshot[0] if previous_headers_snapshot else None
        )
        projected_remaining_tokens = None
        if (
            previous_headers is not None
            and previous_headers.remaining_tokens_minute is not None
        ):
            projected_remaining_tokens = round(
                previous_headers.remaining_tokens_minute - estimated_tokens,
                3,
            )
        return {
            "previous_estimated_request_tokens": previous_estimated_tokens,
            "estimated_request_token_delta_since_previous": (
                estimated_tokens - previous_estimated_tokens
                if previous_estimated_tokens is not None
                else None
            ),
            "previous_successful_token_usage": _token_usage_to_dict(
                previous_successful_usage
            )
            if previous_successful_usage is not None
            else None,
            "previous_rate_limit_headers": (
                previous_headers.as_dict()
                if previous_headers is not None
                else None
            ),
            "projected_remaining_tokens_minute_after_request": (
                projected_remaining_tokens
            ),
        }

    def _record_successful_call(
        self,
        *,
        model: str,
        token_usage: LiteLLMTokenUsage | None,
        rate_limit_headers: CerebrasRateLimitHeaders | None = None,
    ) -> None:
        with self._metrics_lock:
            self._successful_calls += 1
            self._successful_calls_by_model[model] = (
                self._successful_calls_by_model.get(model, 0) + 1
            )
            if token_usage is not None:
                self._total_token_usage = add_token_usage(
                    self._total_token_usage,
                    token_usage,
                ) or LiteLLMTokenUsage()
                self._token_usage_by_model[model] = add_token_usage(
                    self._token_usage_by_model.get(model),
                    token_usage,
                ) or LiteLLMTokenUsage()
                self._last_successful_token_usage_by_model[model] = token_usage
            if rate_limit_headers is not None:
                self._last_rate_limit_headers_by_model[model] = (
                    rate_limit_headers,
                    time.perf_counter(),
                )

    def _write_rate_limit_report(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        estimated_tokens: int,
        duration_ms: float,
        error_details: dict[str, Any],
        rate_limit_signal: CerebrasRateLimitSignal,
    ) -> Path | None:
        created_at = datetime.now().astimezone()
        retry_at = (
            created_at + timedelta(seconds=rate_limit_signal.schedule_wait_seconds)
            if rate_limit_signal.schedule_wait_seconds is not None
            else None
        )
        previous_rate_limit_at = self._previous_rate_limit_at
        previous_retry_at = self._previous_rate_limit_retry_at
        wall_time_since_previous_rate_limit = (
            max(0.0, (created_at - previous_rate_limit_at).total_seconds())
            if previous_rate_limit_at is not None
            else None
        )
        wall_time_since_previous_retry_at = (
            max(0.0, (created_at - previous_retry_at).total_seconds())
            if previous_retry_at is not None
            else None
        )
        with self._metrics_lock:
            successful_calls = self._successful_calls
            successful_calls_by_model = dict(self._successful_calls_by_model)
            attempted_calls = self._attempted_calls
            attempted_calls_by_model = dict(self._attempted_calls_by_model)
            total_token_usage = self._total_token_usage
            token_usage_by_model = dict(self._token_usage_by_model)
            estimated_request_tokens = self._estimated_request_tokens
            estimated_request_tokens_by_model = dict(
                self._estimated_request_tokens_by_model
            )
            last_estimated_request_tokens_by_model = dict(
                self._last_estimated_request_tokens_by_model
            )
            last_rate_limit_headers_by_model = {
                key: value[0]
                for key, value in self._last_rate_limit_headers_by_model.items()
            }
        payload = {
            "schema_version": 1,
            "event": "cerebras_rate_limit",
            "created_at": created_at.isoformat(),
            "session_started_at": self._session_started_at.isoformat(),
            "wall_time_until_rate_limit_seconds": round(
                time.perf_counter() - self._session_started_monotonic,
                3,
            ),
            "previous_rate_limit_at": (
                previous_rate_limit_at.isoformat()
                if previous_rate_limit_at is not None
                else None
            ),
            "wall_time_since_previous_rate_limit_seconds": (
                round(wall_time_since_previous_rate_limit, 3)
                if wall_time_since_previous_rate_limit is not None
                else None
            ),
            "previous_retry_at": (
                previous_retry_at.isoformat()
                if previous_retry_at is not None
                else None
            ),
            "wall_time_since_previous_retry_at_seconds": (
                round(wall_time_since_previous_retry_at, 3)
                if wall_time_since_previous_retry_at is not None
                else None
            ),
            "retry_at": retry_at.isoformat() if retry_at is not None else None,
            "wait_seconds": (
                round(rate_limit_signal.schedule_wait_seconds, 3)
                if rate_limit_signal.schedule_wait_seconds is not None
                else None
            ),
            "rate_limit_signal": rate_limit_signal.as_dict(),
            "scheduler_config": self._scheduler.config.as_log_dict(),
            "scheduler_state": self._scheduler.snapshot(),
            "model": model,
            "service_tier": self.service_tier,
            "successful_litellm_calls": successful_calls,
            "successful_litellm_calls_by_model": successful_calls_by_model,
            "attempted_litellm_calls": attempted_calls,
            "attempted_litellm_calls_by_model": attempted_calls_by_model,
            "tokens_consumed": _token_usage_to_dict(total_token_usage),
            "tokens_consumed_by_model": {
                key: _token_usage_to_dict(value)
                for key, value in sorted(token_usage_by_model.items())
            },
            "estimated_request_tokens_scheduled": estimated_request_tokens,
            "estimated_request_tokens_scheduled_by_model": dict(
                sorted(estimated_request_tokens_by_model.items())
            ),
            "last_estimated_request_tokens_by_model": dict(
                sorted(last_estimated_request_tokens_by_model.items())
            ),
            "last_rate_limit_headers_by_model": {
                key: value.as_dict()
                for key, value in sorted(last_rate_limit_headers_by_model.items())
            },
            "current_call": {
                "num_messages": len(messages),
                "prompt_chars": len(
                    json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                ),
                "estimated_request_tokens": estimated_tokens,
                "max_completion_tokens": max_completion_tokens,
                "has_output_schema": response_schema is not None,
                "output_schema_name": response_schema_name,
                "duration_ms_until_error": round(duration_ms, 1),
            },
            "error_details": _json_safe(error_details),
        }

        timestamp = created_at.strftime("%Y%m%d-%H%M%S")
        filename = (
            f"cerebras-rate-limit-{timestamp}-"
            f"{uuid4().hex[:8]}.json"
        )
        try:
            self.rate_limit_report_dir.mkdir(parents=True, exist_ok=True)
            path = self.rate_limit_report_dir / filename
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._previous_rate_limit_at = created_at
            self._previous_rate_limit_retry_at = retry_at
            return path
        except OSError as write_exc:
            if self.logger:
                self.logger.warning(
                    "Failed to write Cerebras rate-limit report",
                    report_dir=str(self.rate_limit_report_dir),
                    error=str(write_exc),
                )
            self._previous_rate_limit_at = created_at
            self._previous_rate_limit_retry_at = retry_at
            return None


def _is_cerebras_rate_limit_error(
    exc: BaseException,
    *,
    logger: Any | None = None,
    direct_cerebras_error: dict[str, Any] | None = None,
) -> bool:
    """Return whether the error carries a known Cerebras rate-limit signal."""

    details = _exception_details(exc, direct_cerebras_error=direct_cerebras_error)
    signal = _extract_cerebras_rate_limit_signal(
        details,
        queue_backoff_seconds=DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECONDS,
        retry_buffer_seconds=DEFAULT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS,
    )
    _log_cerebras_error(
        exc,
        logger=logger,
        details=details,
        rate_limit_signal=signal,
        report_path=None,
    )
    return signal is not None


def _log_cerebras_error(
    exc: BaseException,
    *,
    logger: Any | None,
    details: dict[str, Any],
    rate_limit_signal: CerebrasRateLimitSignal | None,
    report_path: Path | None,
) -> None:
    if logger is None:
        return
    signal_dict = (
        rate_limit_signal.as_dict() if rate_limit_signal is not None else None
    )
    logger.warning(
        "LiteLLM/Cerebras error observed | raw_error_details={raw_error_details}",
        raw_error_details=_compact_json(details),
        rate_limit_signal=signal_dict,
        rate_limit_report_path=str(report_path) if report_path is not None else None,
        exception=str(exc),
        **details,
    )


def _extract_cerebras_rate_limit_signal(
    details: dict[str, Any],
    *,
    queue_backoff_seconds: float,
    retry_buffer_seconds: float,
) -> CerebrasRateLimitSignal | None:
    provider_candidate = _rate_limit_candidate(
        body=details.get("provider_error_body") or details.get("response_json"),
        headers=details.get("provider_error_headers")
        or details.get("response_headers"),
        status_code=details.get("status_code"),
        source="provider",
        queue_backoff_seconds=queue_backoff_seconds,
        retry_buffer_seconds=retry_buffer_seconds,
    )

    direct_error = details.get("direct_cerebras_error")
    direct_candidate = None
    if isinstance(direct_error, dict):
        direct_candidate = _rate_limit_candidate(
            body=direct_error.get("json"),
            headers=direct_error.get("headers"),
            status_code=direct_error.get("status_code"),
            source="direct_probe",
            queue_backoff_seconds=queue_backoff_seconds,
            retry_buffer_seconds=retry_buffer_seconds,
        )

    candidates = [
        candidate
        for candidate in (provider_candidate, direct_candidate)
        if candidate is not None
    ]
    if not candidates:
        return None
    retry_after_candidates = [
        candidate
        for candidate in candidates
        if candidate.retry_after_seconds is not None
        and candidate.schedule_wait_seconds is not None
    ]
    if retry_after_candidates:
        return max(
            retry_after_candidates,
            key=lambda candidate: candidate.schedule_wait_seconds or 0.0,
        )
    if provider_candidate is not None:
        return provider_candidate
    return direct_candidate


def _rate_limit_candidate(
    *,
    body: Any,
    headers: Any,
    status_code: Any,
    source: str,
    queue_backoff_seconds: float,
    retry_buffer_seconds: float,
) -> CerebrasRateLimitSignal | None:
    headers_dict = _headers_dict(headers) or {}
    payload = _error_payload(body)
    code = _payload_str(payload, "code")
    error_type = _payload_str(payload, "type")
    param = _payload_str(payload, "param")
    message = _payload_str(payload, "message")
    retry_after_seconds = _retry_after_seconds(headers_dict)
    is_429 = _safe_int(status_code) == 429
    is_rate_limit_payload = (
        code
        in {
            "queue_exceeded",
            "token_quota_exceeded",
            "request_quota_exceeded",
        }
        or error_type
        in {
            "too_many_requests_error",
            "too_many_tokens_error",
            "rate_limit_error",
        }
    )
    if not is_429 and not is_rate_limit_payload:
        return None

    schedule_wait_seconds = None
    schedule_reason = None
    if code == "queue_exceeded" or param == "queue":
        schedule_wait_seconds = max(0.0, queue_backoff_seconds)
        schedule_reason = f"{source}_queue_exceeded_backoff"
    elif retry_after_seconds is not None and (
        code in {"token_quota_exceeded", "request_quota_exceeded"}
        or param == "quota"
        or error_type in {"too_many_tokens_error", "too_many_requests_error"}
    ):
        schedule_wait_seconds = max(
            0.0,
            retry_after_seconds + max(0.0, retry_buffer_seconds),
        )
        schedule_reason = f"{source}_retry_after"

    return CerebrasRateLimitSignal(
        code=code,
        type=error_type,
        param=param,
        message=message,
        source=source,
        retry_after_seconds=retry_after_seconds,
        schedule_wait_seconds=schedule_wait_seconds,
        schedule_reason=schedule_reason,
        x_should_retry=_header_value(headers_dict, "x-should-retry"),
    )


class LiteLLMCallScheduler:
    """Local token-bucket scheduler for development-time Cerebras quota pacing."""

    def __init__(
        self,
        config: LiteLLMSchedulerConfig,
        *,
        logger: Any | None = None,
    ) -> None:
        self.config = _normalize_scheduler_config(config)
        self.logger = logger
        self._lock = threading.Lock()
        self._next_call_at = 0.0
        self._external_pause_until = 0.0
        self._external_pause_reason: str | None = None
        now = time.perf_counter()
        self._request_buckets = [
            _QuotaBucket("requests_per_minute", value, SECONDS_PER_MINUTE, now)
            for value in [self.config.requests_per_minute]
            if value is not None
        ] + [
            _QuotaBucket("requests_per_hour", value, SECONDS_PER_HOUR, now)
            for value in [self.config.requests_per_hour]
            if value is not None
        ] + [
            _QuotaBucket("requests_per_day", value, SECONDS_PER_DAY, now)
            for value in [self.config.requests_per_day]
            if value is not None
        ]
        self._token_buckets = [
            _QuotaBucket("tokens_per_minute", value, SECONDS_PER_MINUTE, now)
            for value in [self.config.tokens_per_minute]
            if value is not None
        ] + [
            _QuotaBucket("tokens_per_hour", value, SECONDS_PER_HOUR, now)
            for value in [self.config.tokens_per_hour]
            if value is not None
        ] + [
            _QuotaBucket("tokens_per_day", value, SECONDS_PER_DAY, now)
            for value in [self.config.tokens_per_day]
            if value is not None
        ]

    def wait_before_call(
        self,
        *,
        messages: list[dict[str, Any]],
        max_completion_tokens: int,
        estimated_tokens: int | None = None,
    ) -> int:
        estimated_tokens = estimated_tokens or self.estimate_request_tokens(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        total_waited_seconds = 0.0

        while True:
            with self._lock:
                now = time.perf_counter()
                wait_breakdown = self._wait_breakdown(
                    now=now,
                    estimated_tokens=estimated_tokens,
                )
                wait_reason, wait_seconds = max(
                    wait_breakdown,
                    key=lambda item: item[1],
                )
                if wait_seconds <= 0:
                    self._reserve(now=now, estimated_tokens=estimated_tokens)
                    if total_waited_seconds > 0 and self.logger:
                        self.logger.info(
                            "Local LiteLLM scheduler wait complete; sending next request",
                            waited_seconds=round(total_waited_seconds, 3),
                            estimated_tokens=estimated_tokens,
                        )
                    return estimated_tokens
                if math.isinf(wait_seconds):
                    raise LiteLLMTemplateError(
                        "Local LiteLLM quota schedule cannot admit this request: "
                        f"estimated {estimated_tokens} tokens exceeds at least "
                        "one configured token bucket. Lower "
                        "TRACK2_MAX_COMPLETION_TOKENS, reduce prompt size, or "
                        "raise the configured TRACK2_LLM_TOKENS_* limit."
                    )
                if (
                    self.config.max_schedule_wait_seconds is not None
                    and wait_seconds > self.config.max_schedule_wait_seconds
                ):
                    raise LiteLLMTemplateError(
                        "Local LiteLLM schedule wait would exceed "
                        f"TRACK2_LLM_MAX_SCHEDULE_WAIT_SECONDS: "
                        f"{wait_seconds:.3f}s needed for estimated "
                        f"{estimated_tokens} tokens."
                    )

            if self.logger:
                self.logger.info(
                    "Local LiteLLM scheduler waiting before next request",
                    wait_seconds=round(wait_seconds, 3),
                    resume_at=_format_future_time(wait_seconds),
                    wait_reason=wait_reason,
                    estimated_tokens=estimated_tokens,
                    scheduler=self.config.as_log_dict(),
                )
            time.sleep(wait_seconds)
            total_waited_seconds += wait_seconds

    def estimate_request_tokens(
        self,
        *,
        messages: list[dict[str, Any]],
        max_completion_tokens: int,
    ) -> int:
        return estimate_request_tokens(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            chars_per_token=self.config.token_estimate_chars_per_token,
            safety_factor=self.config.token_safety_factor,
        )

    def apply_external_pause(self, wait_seconds: float, *, reason: str) -> None:
        if wait_seconds <= 0:
            return
        with self._lock:
            pause_until = time.perf_counter() + wait_seconds
            if pause_until > self._external_pause_until:
                self._external_pause_until = pause_until
                self._external_pause_reason = reason

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.perf_counter()
            return {
                "next_call_wait_seconds": round(
                    max(0.0, self._next_call_at - now),
                    3,
                ),
                "external_pause_wait_seconds": round(
                    max(0.0, self._external_pause_until - now),
                    3,
                ),
                "external_pause_reason": self._external_pause_reason,
                "request_buckets": [
                    bucket.snapshot(now) for bucket in self._request_buckets
                ],
                "token_buckets": [
                    bucket.snapshot(now) for bucket in self._token_buckets
                ],
            }

    def _required_wait_seconds(self, *, now: float, estimated_tokens: int) -> float:
        return max(
            wait_seconds
            for _, wait_seconds in self._wait_breakdown(
                now=now,
                estimated_tokens=estimated_tokens,
            )
        )

    def _wait_breakdown(
        self,
        *,
        now: float,
        estimated_tokens: int,
    ) -> list[tuple[str, float]]:
        waits = [
            ("min_interval", max(0.0, self._next_call_at - now)),
            (
                self._external_pause_reason or "external_pause",
                max(0.0, self._external_pause_until - now),
            ),
        ]
        waits.extend(
            (bucket.name, bucket.wait_seconds(1.0, now))
            for bucket in self._request_buckets
        )
        waits.extend(
            (bucket.name, bucket.wait_seconds(float(estimated_tokens), now))
            for bucket in self._token_buckets
        )
        return waits

    def _reserve(self, *, now: float, estimated_tokens: int) -> None:
        for bucket in self._request_buckets:
            bucket.reserve(1.0, now)
        for bucket in self._token_buckets:
            bucket.reserve(float(estimated_tokens), now)
        if self.config.min_interval_seconds > 0:
            self._next_call_at = now + self.config.min_interval_seconds


@dataclass
class _QuotaBucket:
    name: str
    limit: float
    window_seconds: float
    updated_at: float
    available: float | None = None

    def __post_init__(self) -> None:
        if self.available is None:
            self.available = self.limit

    @property
    def refill_rate_per_second(self) -> float:
        return self.limit / self.window_seconds

    def wait_seconds(self, cost: float, now: float) -> float:
        if cost > self.limit:
            return math.inf
        available = self._available_at(now)
        if available >= cost:
            return 0.0
        return (cost - available) / self.refill_rate_per_second

    def reserve(self, cost: float, now: float) -> None:
        self.available = max(0.0, self._available_at(now) - cost)
        self.updated_at = now

    def _available_at(self, now: float) -> float:
        elapsed = max(0.0, now - self.updated_at)
        return min(
            self.limit,
            float(self.available or 0.0) + elapsed * self.refill_rate_per_second,
        )

    def snapshot(self, now: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "available": round(self._available_at(now), 3),
        }


def estimate_request_tokens(
    *,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    chars_per_token: float = DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN,
    safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
) -> int:
    """Estimate Cerebras pre-admission token cost for local quota pacing."""

    chars_per_token = max(chars_per_token, 1.0)
    safety_factor = max(safety_factor, 1.0)
    prompt_chars = len(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    )
    estimated_prompt_tokens = max(1, math.ceil(prompt_chars / chars_per_token))
    completion_budget = max(0, max_completion_tokens)
    return math.ceil(
        (estimated_prompt_tokens + completion_budget) * safety_factor
    )


def _normalize_scheduler_config(
    config: LiteLLMSchedulerConfig,
) -> LiteLLMSchedulerConfig:
    return LiteLLMSchedulerConfig(
        min_interval_seconds=max(0.0, config.min_interval_seconds),
        requests_per_minute=_positive_or_none(config.requests_per_minute),
        requests_per_hour=_positive_or_none(config.requests_per_hour),
        requests_per_day=_positive_or_none(config.requests_per_day),
        tokens_per_minute=_positive_or_none(config.tokens_per_minute),
        tokens_per_hour=_positive_or_none(config.tokens_per_hour),
        tokens_per_day=_positive_or_none(config.tokens_per_day),
        token_estimate_chars_per_token=max(
            1.0,
            config.token_estimate_chars_per_token,
        ),
        token_safety_factor=max(1.0, config.token_safety_factor),
        max_schedule_wait_seconds=(
            None
            if config.max_schedule_wait_seconds is None
            else max(0.0, config.max_schedule_wait_seconds)
        ),
    )


def _positive_or_none(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return float(value)


def _token_usage_to_dict(usage: LiteLLMTokenUsage | None) -> dict[str, int]:
    if usage is None:
        usage = LiteLLMTokenUsage()
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _exception_details(
    exc: BaseException,
    *,
    direct_cerebras_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None) or getattr(
        response,
        "status_code",
        None,
    )
    response_text = getattr(response, "text", None)
    response_json = None
    if response is not None and hasattr(response, "json"):
        try:
            response_json = response.json()
        except Exception:
            response_json = None
    return {
        "exception_type": type(exc).__name__,
        "exception_module": type(exc).__module__,
        "message": str(exc),
        "status_code": status_code,
        "llm_provider": getattr(exc, "llm_provider", None),
        "model": getattr(exc, "model", None),
        "code": getattr(exc, "code", None),
        "type": getattr(exc, "type", None),
        "litellm_debug_info": getattr(exc, "litellm_debug_info", None),
        "response_headers": _headers_dict(getattr(response, "headers", None)),
        "response_text": response_text,
        "response_json": _json_safe(response_json),
        "exception_attrs": _exception_attrs(exc),
        "cause": _exception_summary(getattr(exc, "__cause__", None)),
        "context": _exception_summary(getattr(exc, "__context__", None)),
        "provider_error_body": _provider_error_body(exc),
        "provider_error_headers": _provider_error_headers(exc),
        "direct_cerebras_error": direct_cerebras_error,
    }


def _probe_direct_cerebras_error_response(
    *,
    completion_kwargs: dict[str, Any],
    api_base: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        return {"enabled": True, "error": "CEREBRAS_API_KEY is not set"}

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = _direct_cerebras_payload(completion_kwargs)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **{
            str(key): str(value)
            for key, value in (
                completion_kwargs.get("extra_headers") or {}
            ).items()
        },
    }
    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds or DEFAULT_RAW_ERROR_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as probe_exc:
        return {
            "enabled": True,
            "request_url": url,
            "probe_exception": _exception_summary(probe_exc),
        }

    response_json = None
    try:
        response_json = response.json()
    except Exception:
        response_json = None
    return {
        "enabled": True,
        "request_url": url,
        "status_code": response.status_code,
        "headers": _headers_dict(response.headers),
        "text": response.text,
        "json": _json_safe(response_json),
    }


def _direct_cerebras_payload(completion_kwargs: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": _strip_cerebras_prefix(str(completion_kwargs["model"])),
        "messages": completion_kwargs["messages"],
    }
    for key in (
        "response_format",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "seed",
        "service_tier",
    ):
        if key in completion_kwargs:
            payload[key] = completion_kwargs[key]
    if "max_completion_tokens" in completion_kwargs:
        payload["max_tokens"] = completion_kwargs["max_completion_tokens"]
    elif "max_tokens" in completion_kwargs:
        payload["max_tokens"] = completion_kwargs["max_tokens"]
    return payload


def _strip_cerebras_prefix(model: str) -> str:
    if model.startswith("cerebras/"):
        return model.split("/", 1)[1]
    return model


def _exception_attrs(exc: BaseException) -> dict[str, Any]:
    attrs = {}
    for key, value in getattr(exc, "__dict__", {}).items():
        if key in {"response", "_request"}:
            continue
        attrs[key] = _json_safe(value)
    return attrs


def _exception_summary(exc: BaseException | None) -> dict[str, Any] | None:
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    return {
        "exception_type": type(exc).__name__,
        "exception_module": type(exc).__module__,
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None)
        or getattr(response, "status_code", None),
        "response_headers": _headers_dict(getattr(response, "headers", None)),
        "attrs": _exception_attrs(exc),
    }


def _provider_error_body(exc: BaseException) -> Any:
    for candidate in _exception_chain(exc):
        attrs = getattr(candidate, "__dict__", {})
        body = attrs.get("body")
        if body:
            return _json_safe(body)
        message = attrs.get("message")
        parsed_message_body = _parse_error_body_from_message(message)
        if parsed_message_body is not None:
            return parsed_message_body
    return None


def _provider_error_headers(exc: BaseException) -> dict[str, str] | None:
    for candidate in _exception_chain(exc):
        headers = getattr(candidate, "__dict__", {}).get("headers")
        if headers:
            return _headers_dict(headers)
        response_headers = _headers_dict(
            getattr(getattr(candidate, "response", None), "headers", None)
        )
        if response_headers:
            return response_headers
    return None


def _exception_chain(exc: BaseException):
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "__cause__", None) or getattr(
            current,
            "__context__",
            None,
        )


def _parse_error_body_from_message(message: Any) -> Any:
    if not isinstance(message, str):
        return None
    marker = "Error code: "
    if marker not in message:
        return None
    _, _, after_marker = message.partition(" - ")
    if not after_marker:
        return None
    try:
        return json.loads(after_marker)
    except json.JSONDecodeError:
        return None


def _error_payload(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    if isinstance(error, dict):
        return error
    return body


def _payload_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def _header_value(headers: dict[str, str] | None, key: str) -> str | None:
    if not headers:
        return None
    expected = key.lower()
    for header_key, value in headers.items():
        if header_key.lower() == expected:
            return value
    return None


def _retry_after_seconds(headers: dict[str, str] | None) -> float | None:
    value = _header_value(headers, "retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.astimezone()
    now = datetime.now(retry_at.tzinfo)
    return max(0.0, (retry_at - now).total_seconds())


def _headers_dict(headers: Any) -> dict[str, str] | None:
    if headers is None:
        return None
    try:
        return {str(key): str(value) for key, value in dict(headers).items()}
    except Exception:
        return {"repr": repr(headers)}


def _message_content(message: Any) -> str:
    content = _get_field(message, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            text = _get_field(item, "text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()
    return ""


def _response_cost(response: Any) -> float:
    hidden_params = getattr(response, "_hidden_params", {}) or {}
    if isinstance(hidden_params, dict):
        return float(hidden_params.get("response_cost") or 0.0)
    return 0.0


def _response_rate_limit_headers(response: Any) -> CerebrasRateLimitHeaders | None:
    for headers in _response_header_candidates(response):
        rate_limit_headers = CerebrasRateLimitHeaders.from_headers(headers)
        if rate_limit_headers is not None:
            return rate_limit_headers
    return None


def _response_header_candidates(response: Any):
    yield getattr(response, "headers", None)
    raw_response = getattr(response, "response", None) or getattr(
        response,
        "_response",
        None,
    )
    yield getattr(raw_response, "headers", None)
    hidden_params = getattr(response, "_hidden_params", {}) or {}
    if isinstance(hidden_params, dict):
        for key in (
            "response_headers",
            "headers",
            "raw_response_headers",
            "llm_provider_response_headers",
            "additional_headers",
        ):
            yield hidden_params.get(key)
        hidden_response = hidden_params.get("response") or hidden_params.get(
            "raw_response"
        )
        yield getattr(hidden_response, "headers", None)


def _is_cerebras_model(model: str) -> bool:
    return model.startswith("cerebras/")


def _get_field(value: Any, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return repr(value)


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return repr(value)


def _format_future_time(wait_seconds: float) -> str:
    return (
        datetime.now().astimezone() + timedelta(seconds=max(0.0, wait_seconds))
    ).isoformat(timespec="seconds")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default
