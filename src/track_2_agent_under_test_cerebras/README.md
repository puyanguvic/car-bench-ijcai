# Track 2 Cerebras Agent Under Test

This package is the direct Track 2 Cerebras Fast-Reasoning starter agent for
CAR-bench A2A evaluation. It calls Cerebras-hosted `gpt-oss` models directly
through LiteLLM and returns the same benchmark-visible A2A text responses or
tool calls as the Track 1 template.

The default executor model is `cerebras/gpt-oss-120b`. Participants should use a
Cerebras-hosted `gpt-oss` executor model, while replacing prompting,
validation, or harnessing strategy as long as the external A2A contract stays
unchanged.

## What This Agent Demonstrates

- Parses evaluator messages into policy/user text, tool definitions, and tool
  results.
- Maintains conversation history per `context_id`.
- Calls LiteLLM `completion(...)` with Cerebras provider settings.
- Uses Cerebras structured JSON schema output for a strict next-action object.
- Applies a process-local call scheduler through
  `TRACK2_LLM_MIN_INTERVAL_SECONDS`.
- Logs raw LiteLLM/Cerebras errors for future rate-limit handling, but does not
  claim `quota_wait_time_ms` until the exact rate-limit error shape is known.

## Configuration

Set the evaluator key and Cerebras key in `.env`:

```bash
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
TRACK2_EXECUTOR_MODEL=cerebras/gpt-oss-120b
```

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | required | Cerebras API key used by LiteLLM. |
| `TRACK2_EXECUTOR_MODEL` | `cerebras/gpt-oss-120b` | Cerebras-hosted `gpt-oss` executor model string passed to LiteLLM. |
| `TRACK2_CEREBRAS_API_BASE` | `https://api.cerebras.ai/v1` | Cerebras API base URL. |
| `TRACK2_CEREBRAS_SERVICE_TIER` | unset | Optional Cerebras service tier, for example `default`, `priority`, `auto`, or `flex`. |
| `TRACK2_MAX_COMPLETION_TOKENS` | `1024` | Completion-token cap for executor calls. |
| `TRACK2_TEMPERATURE` | `0` | LiteLLM temperature. |
| `TRACK2_LLM_MIN_INTERVAL_SECONDS` | `0` | Hard minimum spacing between LiteLLM call starts. |
| `TRACK2_LLM_REQUESTS_PER_MINUTE` / `HOUR` / `DAY` | unset | Optional process-local request quota pacing. |
| `TRACK2_LLM_TOKENS_PER_MINUTE` / `HOUR` / `DAY` | unset | Optional process-local token quota pacing using estimated prompt tokens plus `TRACK2_MAX_COMPLETION_TOKENS`. |
| `TRACK2_LLM_TOKEN_ESTIMATE_CHARS_PER_TOKEN` | `4` | Prompt-size estimator used before the provider returns usage. |
| `TRACK2_LLM_TOKEN_SAFETY_FACTOR` | `1.1` | Multiplier applied to estimated request tokens. |
| `TRACK2_LLM_MAX_SCHEDULE_WAIT_SECONDS` | unset | Optional fail-fast cap if local pacing would sleep too long inside a benchmark turn. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS` | `60` | Local pause after a provider `queue_exceeded` 429. |
| `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS` | `1` | Safety buffer added to provider `retry-after` seconds before the next scheduled call. |
| `CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR` | `/tmp/car-bench-rate-limit-reports` | Directory for Cerebras rate-limit JSON reports. Falls back to `CAR_BENCH_RATE_LIMIT_REPORT_DIR` when set. |
| `TRACK2_CEREBRAS_RAW_ERROR_PROBE` | `0` | Debug-only: after a LiteLLM Cerebras error, send one extra direct Cerebras request and log the raw HTTP response. |
| `TRACK2_CEREBRAS_RAW_ERROR_PROBE_TIMEOUT_SECONDS` | `30` | Timeout for the debug raw-error probe. |
| `TRACK2_LLM_MALFORMED_RETRIES` | `1` | Retry budget for malformed next-action JSON. |

## Rate Limits And Development Windows

During normal development, participants are expected to use the Cerebras public
tier, where rate limits can be strict. Use smaller smoke scenarios first, keep
`TRACK2_MAX_COMPLETION_TOKENS` as low as the task allows, and configure local
request/token pacing for longer runs when needed.

Example public-tier style configuration:

```bash
TRACK2_LLM_REQUESTS_PER_MINUTE=30
TRACK2_LLM_TOKENS_PER_MINUTE=60000
TRACK2_LLM_TOKENS_PER_DAY=1000000
TRACK2_LLM_MAX_SCHEDULE_WAIT_SECONDS=60
```

The scheduler is process-local and approximate. It uses token buckets for each
configured request/token window and estimates token cost as prompt size plus the
completion-token cap. Unset dimensions are ignored. `TRACK2_LLM_MIN_INTERVAL_SECONDS`
can still be used as a blunt hard floor between calls.

Local scheduler sleeps are ordinary harness queueing and are not reported as
`quota_wait_time_ms`. Use `TRACK2_LLM_MAX_SCHEDULE_WAIT_SECONDS` to fail fast
when a long hour/day wait should be handled by an external task-run scheduler
rather than by blocking one CAR-bench turn.
Evaluator-facing `avg_llm_call_time_ms` is computed only from successful
provider calls; scheduler sleeps, failed 429 attempts, and optional raw-error
probes are kept in logs/reports instead.

When a Cerebras 429 is observed, the client writes a JSON report by default to
`/tmp/car-bench-rate-limit-reports`. The report includes session start time,
wall time until the limit, wall time since the previous limit/retry marker,
successful-call token usage, estimated scheduled request tokens, the current
failed call shape, and the raw provider/probe payload. A provider
`queue_exceeded` error applies a configurable local backoff. A quota error with
`retry-after` applies that reset hint plus the configured buffer to future calls.
These waits and failed attempts are not reported as `quota_wait_time_ms`.

On successful Cerebras responses, the client also reads provider
`x-ratelimit-*` headers when LiteLLM exposes them. Terminal logs show the
previous request's estimated token cost, the upcoming request estimate, the
delta since the previous request, remaining tokens/minute from the last
successful response, and projected remaining tokens after the upcoming request.
If those headers indicate that the upcoming request would exceed remaining
tokens/minute or daily requests, the scheduler waits until the provider reset
hint plus `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS`.

For debugging only, set `TRACK2_CEREBRAS_RAW_ERROR_PROBE=1` to capture the raw
Cerebras HTTP error body when LiteLLM has already mapped the provider response
into a LiteLLM exception. This sends one additional direct Cerebras API request
after the LiteLLM error, so keep it disabled for normal benchmark runs.

Organizers will provide a few test windows with elevated rate limits and
priority tier access so participants can test harness behavior at higher speed
and with less throttling. Participants may also self-host the open-source models
used by the Cerebras `gpt-oss` executor during development, then validate
speed-sensitive behavior during those windows.

Codex Pro plans are still provided for selected Track 2 teams to accelerate
harness engineering and development. They are not the submitted-agent runtime
for this template. Plans are allocated by June 15.

## Run

Local smoke:

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_cerebras/local_smoke.toml --show-logs
```

Docker smoke:

```bash
uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_cerebras/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_2_agent_under_test_cerebras/docker-compose.yml up --abort-on-container-exit
```

## Read More

- [Main README](../../README.md): setup, validation modes, and submission shape.
- [Development guide](../../docs/development-guide.md): detailed A2A turn
  contract.
- [Harnessing guide](../../docs/agent-under-test-harnessing.md): allowed
  harness boundaries.
- [Track 2 harness patterns](../../docs/cerebras-harness-patterns.md): direct
  Cerebras, planner/executor, and rate-limit development guidance.
