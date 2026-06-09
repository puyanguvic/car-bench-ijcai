"""Server entry point for the Track 2 Cerebras CAR-bench agent."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

if __package__:
    from .car_bench_agent import CARBenchAgentExecutor
    from .litellm_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
        LiteLLMSchedulerConfig,
    )
else:
    from car_bench_agent import CARBenchAgentExecutor
    from litellm_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
        LiteLLMSchedulerConfig,
    )

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="server")


def _env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_float(name: str, default: float | None = None) -> float | None:
    value = _env_or_default(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = _env_or_default(name)
    if value is None:
        return default
    return int(value)


def _scheduler_config_from_args(args) -> LiteLLMSchedulerConfig:
    return LiteLLMSchedulerConfig(
        min_interval_seconds=(
            args.min_interval_seconds
            if args.min_interval_seconds is not None
            else _env_float("TRACK2_LLM_MIN_INTERVAL_SECONDS", 0.0)
        )
        or 0.0,
        requests_per_minute=(
            args.requests_per_minute
            if args.requests_per_minute is not None
            else _env_float("TRACK2_LLM_REQUESTS_PER_MINUTE")
        ),
        requests_per_hour=(
            args.requests_per_hour
            if args.requests_per_hour is not None
            else _env_float("TRACK2_LLM_REQUESTS_PER_HOUR")
        ),
        requests_per_day=(
            args.requests_per_day
            if args.requests_per_day is not None
            else _env_float("TRACK2_LLM_REQUESTS_PER_DAY")
        ),
        tokens_per_minute=(
            args.tokens_per_minute
            if args.tokens_per_minute is not None
            else _env_float("TRACK2_LLM_TOKENS_PER_MINUTE")
        ),
        tokens_per_hour=(
            args.tokens_per_hour
            if args.tokens_per_hour is not None
            else _env_float("TRACK2_LLM_TOKENS_PER_HOUR")
        ),
        tokens_per_day=(
            args.tokens_per_day
            if args.tokens_per_day is not None
            else _env_float("TRACK2_LLM_TOKENS_PER_DAY")
        ),
        token_estimate_chars_per_token=(
            args.token_estimate_chars_per_token
            if args.token_estimate_chars_per_token is not None
            else _env_float("TRACK2_LLM_TOKEN_ESTIMATE_CHARS_PER_TOKEN", 4.0)
        )
        or 4.0,
        token_safety_factor=(
            args.token_safety_factor
            if args.token_safety_factor is not None
            else _env_float("TRACK2_LLM_TOKEN_SAFETY_FACTOR", 1.1)
        )
        or 1.1,
        max_schedule_wait_seconds=(
            args.max_schedule_wait_seconds
            if args.max_schedule_wait_seconds is not None
            else _env_float("TRACK2_LLM_MAX_SCHEDULE_WAIT_SECONDS")
        ),
    )


def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the Cerebras agent under test."""

    card = AgentCard(
        name="car_bench_agent_cerebras",
        description=(
            "In-car voice assistant agent for CAR-bench using direct "
            "Cerebras inference through LiteLLM"
        ),
        version="1.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )

    iface = card.supported_interfaces.add()
    iface.url = url
    iface.protocol_binding = "JSONRPC"
    iface.protocol_version = "1.0"

    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "car_assistant"
    skill.name = "In-Car Voice Assistant (Cerebras)"
    skill.description = "Returns CAR-bench text responses or tool calls through A2A"
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant", "cerebras"])

    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAR-bench Track 2 Cerebras agent under test."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--service-tier", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--min-interval-seconds", type=float, default=None)
    parser.add_argument("--requests-per-minute", type=float, default=None)
    parser.add_argument("--requests-per-hour", type=float, default=None)
    parser.add_argument("--requests-per-day", type=float, default=None)
    parser.add_argument("--tokens-per-minute", type=float, default=None)
    parser.add_argument("--tokens-per-hour", type=float, default=None)
    parser.add_argument("--tokens-per-day", type=float, default=None)
    parser.add_argument("--token-estimate-chars-per-token", type=float, default=None)
    parser.add_argument("--token-safety-factor", type=float, default=None)
    parser.add_argument("--max-schedule-wait-seconds", type=float, default=None)
    parser.add_argument("--malformed-retries", type=int, default=None)
    args = parser.parse_args()

    if not _env_or_default("CEREBRAS_API_KEY"):
        raise SystemExit("CEREBRAS_API_KEY must be set for Track 2 Cerebras runs.")

    executor_model = (
        args.executor_model
        if args.executor_model is not None
        else _env_or_default("TRACK2_EXECUTOR_MODEL", DEFAULT_EXECUTOR_MODEL)
    )
    api_base = (
        args.api_base
        if args.api_base is not None
        else _env_or_default("TRACK2_CEREBRAS_API_BASE", DEFAULT_CEREBRAS_API_BASE)
    )
    service_tier = (
        args.service_tier
        if args.service_tier is not None
        else _env_or_default("TRACK2_CEREBRAS_SERVICE_TIER")
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else _env_float("TRACK2_TEMPERATURE", 0.0)
    )
    max_completion_tokens = (
        args.max_completion_tokens
        if args.max_completion_tokens is not None
        else _env_int("TRACK2_MAX_COMPLETION_TOKENS", 1024)
    )
    min_interval_seconds = (
        args.min_interval_seconds
        if args.min_interval_seconds is not None
        else _env_float("TRACK2_LLM_MIN_INTERVAL_SECONDS", 0.0)
    )
    scheduler_config = _scheduler_config_from_args(args)
    malformed_retries = (
        args.malformed_retries
        if args.malformed_retries is not None
        else _env_int("TRACK2_LLM_MALFORMED_RETRIES", 1)
    )

    logger.info(
        "Starting CAR-bench agent (Cerebras)",
        executor_model=executor_model,
        api_base=api_base,
        service_tier=service_tier,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        min_interval_seconds=min_interval_seconds,
        scheduler=scheduler_config.as_log_dict(),
        malformed_retries=malformed_retries,
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=CARBenchAgentExecutor(
            model=executor_model or DEFAULT_EXECUTOR_MODEL,
            api_base=api_base or DEFAULT_CEREBRAS_API_BASE,
            service_tier=service_tier,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            min_interval_seconds=min_interval_seconds or 0.0,
            scheduler_config=scheduler_config,
            malformed_retries=malformed_retries,
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True)
    card_routes = create_agent_card_routes(card)
    app = Starlette(routes=routes + card_routes)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()
