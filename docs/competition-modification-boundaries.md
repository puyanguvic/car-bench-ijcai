# Competition Modification Boundaries

This repository contains both participant-owned code and local evaluation
helpers. Keep that boundary explicit. A change is competition-safe only when it
does not make local evaluation differ from the official evaluator in a way that
improves or changes benchmark outcomes.

## Core Rule

The submitted system is the agent under test plus its declared runtime
dependencies. The CAR-bench evaluator, benchmark tasks, tools, hidden state,
mock data, reward logic, and simulated user are official infrastructure. Do not
change them to make a run pass.

If a local test disagrees with the official evaluator behavior, align the test
or mark it as an expected local limitation. Do not patch `third_party/car-bench`
to satisfy this repository's tests.

## Participant-Owned Code

These areas are safe places to implement agent behavior:

| Path | Allowed changes |
| --- | --- |
| `src/track_1_agent_under_test/` | Track 1 agent implementation, server wiring, Dockerfile, track README. |
| `src/track_2_agent_under_test_cerebras/` | Track 2 direct Cerebras agent, SDK adapter, server wiring, Dockerfile, track README. |
| `src/track_2_agent_under_test_cerebras_planner/` | Track 2 planner/executor agent, server wiring, Dockerfile, track README. |
| `src/carbench_agent_core/` | Shared participant-owned agent logic used by agent-under-test implementations. |
| `src/tool_call_types.py`, `src/turn_metrics.py`, `src/logging_utils.py` | Shared participant-side schemas, metadata keys, and logging helpers. |
| `pyproject.toml`, `uv.lock` | Dependencies needed by the submitted agent or local harness. Keep dependency changes intentional and reproducible. |

Agent code may use only benchmark-visible inputs when deciding the next action:
system prompt, conversation text, exposed tool schemas, and evaluator-returned
tool results. It must not read hidden CAR-bench files, task answers, evaluator
state, or mock data to choose actions.

## Local Harness Code

These areas can be changed for local orchestration, packaging, diagnostics, or
developer ergonomics, but a score improvement must still reproduce against the
official evaluator:

| Path | Allowed changes |
| --- | --- |
| `src/agentbeats/` | A2A runner/client helpers, output formatting, local orchestration. |
| `generate_compose.py` | Compose generation for local and Docker validation. |
| `scenarios/` | Scenario configs for smoke, test-set, local Docker, GHCR Docker runs. |
| `scripts/` | Setup scripts that fetch dependencies or prepare local validation. |
| `tests/` | Repository contract tests. Tests must reflect official evaluator behavior. |
| `docs/`, `README.md`, `third_party/README.md` | Documentation. |
| `outputs/`, `figures/` | Reporting and generated analysis assets. |

Do not rely on local harness changes for official results unless the same
behavior is inside the submitted agent container or official scenario config.

## Official Infrastructure

Do not modify these for competition behavior:

| Path or component | Rule |
| --- | --- |
| `third_party/car-bench/` | Treat as read-only official upstream. It is ignored by git and can be deleted and recreated by `scripts/setup_car_bench.sh`. |
| `third_party/car-bench/car_bench/envs/` | Do not change tools, reward calculators, simulated user behavior, task loading, hidden state, or evaluator internals. |
| `src/evaluator/` | Do not change scoring semantics, tool execution semantics, task selection semantics, or reward interpretation. Only wrapper compatibility/logging changes are acceptable, and official Docker validation must remain the authority. |
| Official evaluator image | Treat `ghcr.io/car-bench/car-bench-evaluator:latest` as the source of truth for Docker validation. |
| `.env`, provider keys, local caches | Local-only secrets and caches. Never make benchmark behavior depend on untracked local state. |

Generated files such as `docker-compose.yml`, `a2a-scenario.toml`, `output/`,
and Python caches are local artifacts and should not be committed.

## Test Policy

Tests should protect our agent and harness contracts without inventing a
different evaluator.

- Keep active tests for participant-owned behavior and A2A contracts.
- Keep active tests for explicit errors that official tools record themselves.
- Mark tests as `expectedFailure` when they document desired behavior that the
  current official evaluator does not implement.
- Do not make tests pass by patching `third_party/car-bench`.
- When official upstream behavior changes, remove the expected-failure marker
  and let the test become active.

## Review Checklist

Before committing or opening a PR, verify:

1. `git status -sb` from the repository root shows only intended files.
2. `git -C third_party/car-bench status -sb` is clean.
3. The diff does not change official tools, tasks, hidden data, reward logic, or
   simulated user behavior.
4. Agent decisions use only A2A-visible inputs.
5. Local unit tests either pass or have explicit expected failures tied to
   official evaluator behavior.
6. Important score claims are validated with an official-evaluator path, such as
   the Docker local build or GHCR scenario, not only a modified local checkout.

If a proposed change fails one of these checks, treat it as local analysis or
documentation, not as competition code.
