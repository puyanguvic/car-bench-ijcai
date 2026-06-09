# Track 2 Cerebras Fast-Reasoning Harness Patterns

Track 2 agents use direct Cerebras-hosted `gpt-oss` inference through LiteLLM.
The reference templates keep the public A2A boundary identical to Track 1 while
giving participants a starting point for compute-aware harnesses.

## Reference Agent Map

| Agent | Package | Local Scenario | Internal Strategy |
|-------|---------|----------------|-------------------|
| Direct Cerebras agent | [`src/track_2_agent_under_test_cerebras/`](../src/track_2_agent_under_test_cerebras/) | [`scenarios/track_2_agent_under_test_cerebras/local_smoke.toml`](../scenarios/track_2_agent_under_test_cerebras/local_smoke.toml) | Cerebras `gpt-oss` executor returns schema-constrained next-action JSON. |
| Cerebras planner/executor | [`src/track_2_agent_under_test_cerebras_planner/`](../src/track_2_agent_under_test_cerebras_planner/) | [`scenarios/track_2_agent_under_test_cerebras_planner/local_smoke.toml`](../scenarios/track_2_agent_under_test_cerebras_planner/local_smoke.toml) | Env-configured private planner writes a compact plan; Cerebras `gpt-oss` executor returns normal A2A output. |

## Model Selection

The direct executor defaults to:

```env
CEREBRAS_API_KEY=...
TRACK2_EXECUTOR_MODEL=cerebras/gpt-oss-120b
TRACK2_CEREBRAS_API_BASE=https://api.cerebras.ai/v1
```

The planner/executor template additionally requires:

```env
TRACK2_PLANNER_MODEL=azure/gpt-5.5
```

Use the LiteLLM model string accepted by your provider route. For example,
`openai/gpt-5.5` and `azure/gpt-5.5` are documented examples, but the starter
does not hard-code a closed-source planner provider.

## Public-Tier Development Logistics

Most participants will use the Cerebras public tier during development. Rate
limits can be strict, so the templates include process-local quota pacing. You
can configure any mix of request and token windows:

```env
TRACK2_LLM_REQUESTS_PER_MINUTE=30
TRACK2_LLM_REQUESTS_PER_HOUR=
TRACK2_LLM_REQUESTS_PER_DAY=
TRACK2_LLM_TOKENS_PER_MINUTE=60000
TRACK2_LLM_TOKENS_PER_HOUR=
TRACK2_LLM_TOKENS_PER_DAY=1000000
TRACK2_LLM_MAX_SCHEDULE_WAIT_SECONDS=60
TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS=60
TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS=1
```

Unset dimensions are ignored. The scheduler estimates each request as prompt
tokens plus the call's completion-token cap, then reserves capacity from every
configured request/token bucket. `TRACK2_LLM_MIN_INTERVAL_SECONDS` remains
available as a simple hard floor between calls.

This is ordinary local queueing and must not be reported as
`quota_wait_time_ms`. For hour/day exhaustion, prefer an external run scheduler
or fail-fast cap instead of sleeping for hours inside one CAR-bench turn.
The reference templates compute `avg_llm_call_time_ms` only from successful
provider calls. Scheduler sleeps, failed 429 attempts, and optional raw-error
probe calls are visible in logs/reports but do not enter submitted LLM latency.

Organizers will provide a few elevated-rate/priority test windows for Track 2
teams. Use those windows to validate speed-sensitive harness behavior and
larger public validation runs. Participants may also self-host the open-source
models used by the Cerebras `gpt-oss` executor during development, then switch
to the Cerebras endpoint for test-window validation.

Codex Pro plans are allocated by June 15 for selected Track 2 participants to
speed up harness engineering and development. Codex Pro is not the runtime used
by the new submitted-agent templates.

## Pattern 1: Direct Next-Action Baseline

Each CAR-bench assistant step becomes one LiteLLM call:

```text
A2A input from evaluator
  -> build transcript and task-filtered tool prompt
  -> Cerebras next-action JSON
  -> parse JSON
  -> return text Part or data Part(tool_calls) to evaluator
```

This is the lowest-latency and easiest-to-debug Track 2 template. It is the best
starting point before adding planners, verifiers, rerankers, or ensembles.

## Pattern 2: Private Planner Plus Cerebras `gpt-oss` Executor

Use a larger closed-source model only to write compact private guidance, then
let the Cerebras `gpt-oss` executor produce the benchmark-visible next action. The
reference planner runs when a new user turn arrives. Tool-result continuation
turns reuse the active private plan until the executor returns a final user
response.

The private plan is not a CAR-bench tool call and is never returned to the
evaluator. If the executor needs benchmark-visible planning behavior, it can
call CAR-bench's supplied `planning_tool` like any other available tool.

## Rate-Limit Accounting

The Cerebras templates do not claim quota-wait discounts. They keep
`quota_wait_time_ms` at `0.0` and write an audit report when a known Cerebras 429
shape is observed. Reports are written to
`CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR`, falling back to
`CAR_BENCH_RATE_LIMIT_REPORT_DIR`, then `/tmp/car-bench-rate-limit-reports`.

The observed Cerebras shapes are:

- `queue_exceeded` with `param: "queue"` on the original provider error. This
  means the provider queue is saturated, so the scheduler applies
  `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS` before future calls.
- `token_quota_exceeded` with `param: "quota"` and a `retry-after` header. This
  means a request/token quota window has a reset hint, so the scheduler applies
  that wait plus `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS`.

Successful Cerebras responses may include `x-ratelimit-*` headers with daily
request and per-minute token limits, remaining quota, and reset seconds. When
LiteLLM exposes those headers, the template logs them and uses them for the next
pre-call scheduling decision. The terminal shows:

- previous estimated request tokens,
- upcoming estimated request tokens,
- estimated token delta since the previous request,
- previous remaining tokens/minute and reset seconds,
- projected remaining tokens/minute after the upcoming request.

If the next estimated request would exceed the previously observed remaining
tokens/minute or daily request count, the scheduler waits until the header reset
hint plus `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS`. This is still
ordinary harness scheduling, not `quota_wait_time_ms`.

LiteLLM can map a provider 429 into `litellm.exceptions.RateLimitError`. For
debugging only, set `TRACK2_CEREBRAS_RAW_ERROR_PROBE=1` to send one additional
direct Cerebras API request after the LiteLLM error and capture the raw HTTP
status, headers, text, and JSON body. Keep this disabled for normal benchmark
runs because it consumes an extra request.

Rate-limit reports include session start time, wall time until the limit, wall
time since the previous rate-limit and retry markers, successful-call token
usage, estimated scheduled request tokens, the failed call shape, scheduler
state, and raw provider/probe payloads. These files are for diagnosis and manual
audit; they do not reduce evaluator-observed wall time unless the benchmark
contract later introduces a trusted Cerebras quota-wait discount.

Do not report local scheduling, queue backoff, failed provider attempts,
malformed-output repair work, planner latency, raw-error probes, or ordinary
slow inference as `quota_wait_time_ms`. Successful planner/executor calls still
count as LLM calls for `num_llm_calls` and `avg_llm_call_time_ms`.
