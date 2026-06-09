# Track 2 Cerebras Planner/Executor Agent Under Test

This package is a Track 2 planner/executor template. A private planner model
creates compact internal guidance after a user turn, and the Cerebras-hosted
`gpt-oss` executor returns the benchmark-visible next action through the normal
A2A interface.

The planner model is intentionally environment-required because the correct
closed-source `gpt-5.5` LiteLLM route is deployment-specific. Examples include
`openai/gpt-5.5` or `azure/gpt-5.5`.

## What This Agent Demonstrates

- Keeps the public A2A boundary identical to Track 1 and the direct Track 2
  Cerebras template.
- Uses `TRACK2_PLANNER_MODEL` for private plan creation.
- Uses `TRACK2_EXECUTOR_MODEL`, default `cerebras/gpt-oss-120b`, as the
  Cerebras-hosted `gpt-oss` model for final next-action execution.
- Reuses the private plan across tool-result continuation turns until the
  executor can answer the user.
- Aggregates planner and executor token usage and call counts in
  `turn_metrics`.

## Configuration

Set the required keys in `.env`:

```bash
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
TRACK2_PLANNER_MODEL=azure/gpt-5.5
TRACK2_EXECUTOR_MODEL=cerebras/gpt-oss-120b
```

Also set the provider key required by your `TRACK2_PLANNER_MODEL`, for example
`OPENAI_API_KEY` or Azure LiteLLM environment variables.

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRACK2_PLANNER_MODEL` | required | Private planner model string. |
| `TRACK2_PLANNER_MAX_COMPLETION_TOKENS` | `2048` | Completion-token cap for planner calls. |
| `TRACK2_EXECUTOR_MODEL` | `cerebras/gpt-oss-120b` | Cerebras-hosted `gpt-oss` executor model string. |
| `TRACK2_MAX_COMPLETION_TOKENS` | `1024` | Completion-token cap for executor calls. |
| `CEREBRAS_API_KEY` | required | Cerebras API key used by executor calls. |
| `TRACK2_CEREBRAS_SERVICE_TIER` | unset | Optional service tier, for example `default`, `priority`, `auto`, or `flex`. |
| `TRACK2_LLM_MIN_INTERVAL_SECONDS` | `0` | Hard minimum spacing between LiteLLM call starts. |
| `TRACK2_LLM_REQUESTS_PER_MINUTE` / `HOUR` / `DAY` | unset | Optional process-local request quota pacing across planner and executor calls. |
| `TRACK2_LLM_TOKENS_PER_MINUTE` / `HOUR` / `DAY` | unset | Optional process-local token quota pacing using prompt estimates plus each call's completion-token cap. |
| `TRACK2_LLM_TOKEN_ESTIMATE_CHARS_PER_TOKEN` | `4` | Prompt-size estimator used before provider usage is known. |
| `TRACK2_LLM_TOKEN_SAFETY_FACTOR` | `1.1` | Multiplier applied to estimated request tokens. |
| `TRACK2_LLM_MAX_SCHEDULE_WAIT_SECONDS` | unset | Optional fail-fast cap if local pacing would sleep too long inside a benchmark turn. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS` | `60` | Local pause after a Cerebras executor `queue_exceeded` 429. |
| `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS` | `1` | Safety buffer added to provider `retry-after` seconds before the next scheduled call. |
| `CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR` | `/tmp/car-bench-rate-limit-reports` | Directory for Cerebras rate-limit JSON reports. Falls back to `CAR_BENCH_RATE_LIMIT_REPORT_DIR` when set. |
| `TRACK2_LLM_MALFORMED_RETRIES` | `1` | Retry budget for malformed planner or executor JSON. |

Planner/executor runs may consume two LiteLLM calls for one benchmark-visible
assistant step. If you configure request or token quotas, size them for both
the private planner and Cerebras `gpt-oss` executor. Local scheduler sleeps are
ordinary harness queueing and are not reported as `quota_wait_time_ms`.
Evaluator-facing `avg_llm_call_time_ms` is computed only from successful
provider calls; scheduler sleeps, failed 429 attempts, and optional raw-error
probes are kept in logs/reports instead.

## Run

Local smoke:

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_cerebras_planner/local_smoke.toml --show-logs
```

Docker smoke:

```bash
uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_cerebras_planner/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_2_agent_under_test_cerebras_planner/docker-compose.yml up --abort-on-container-exit
```

## Notes

The planner output is private harness state. It is never returned as a
CAR-bench tool call. If the executor needs benchmark-visible planning behavior,
it may call the supplied `planning_tool` like any other available CAR-bench
tool.

Rate-limit wait accounting is deliberately not implemented for this template
yet. The client logs raw LiteLLM/Cerebras exception details so the exact
rate-limit error can be captured before trusted quota-wait metadata is added.
