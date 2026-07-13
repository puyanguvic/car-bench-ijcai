# Track 2 technical-report outline

Use the IJCAI author kit and keep the main text to four pages, excluding
references. Replace every bracketed item with measured evidence before
submission; do not claim hidden-set results.

## Title, authors, and abstract

- **Title:** [Method name]: Compute-Aware Reliable Tool-Using Agents for
  CAR-bench
- **Authors and affiliations:** [team members and institutions]
- **Abstract:** State the Track 2 direct-Cerebras `gpt-oss` setup, the
  controller/harness contribution, and only public-validation measurements.

## 1. Method

Describe the agent's two decision layers:

1. `PolicyAwareController` resolves deterministic safety, policy, and
   multi-turn workflow cases without an LLM call.
2. The residual Cerebras-hosted `gpt-oss` next-action model receives compact
   callable tool schemas and a bounded transcript that preserves the system
   policy, initial user request, and recent tool state.
3. The client uses prior successful Cerebras rate-limit headers to pace a
   request whose estimated tokens would exceed the exposed remaining
   token-minute quota.

For the planner/executor variant, describe the private plan, executor, and
plan-reuse rule separately. Clearly state which variant was submitted.

## 2. Architecture and compute audit

Include one architecture diagram with the following path:

```text
Evaluator A2A input
  -> policy-aware controller (0 model calls when decisive)
  -> [optional private Cerebras planner]
  -> compact prompt (tool constraints + bounded transcript)
  -> Cerebras gpt-oss executor with quota pacing
  -> schema validation / malformed-response retry
  -> A2A text or tool-call response + aggregate turn_metrics
```

State the maximum sequential LLM calls per baseline step. The direct agent
permits at most five executor calls. The planner/executor agent permits at most
five calls along its worst malformed-output path. Parallelism, if introduced,
must be shown explicitly.

## 3. Validation

Report only reproducible public or train results:

| Configuration | Split/tasks/trials | Pass rate | Pass^3 | Median/mean latency | Average input/reasoning/output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Smoke | [fill] | [fill] | N/A | [fill] | [fill] |
| Public batch | [fill] | [fill] | [fill if 3 trials] | [fill] | [fill] |
| Ablation: controller disabled | [fill] | [fill] | [fill] | [fill] | [fill] |
| Ablation: planner enabled/disabled | [fill] | [fill] | [fill] | [fill] | [fill] |

Explain rate-limit waits separately from successful inference latency. Token
totals must come from A2A `turn_metrics.prompt_tokens`,
`completion_tokens`, and `thinking_tokens`.

## 4. Lessons, limitations, and reproducibility

Discuss known remaining failure clusters, the impact of Cerebras rate limits,
the public GHCR digest, required environment-variable names, and the exact
scenario configuration. Do not include secrets.

## References

Include the required CAR-bench citation from the project README, plus any
model or framework citations used by the final method.
