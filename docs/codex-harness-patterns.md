# Legacy Codex Harness Patterns

Track 2 has moved from Codex app-server runtime to direct Cerebras inference
through LiteLLM. New participants should use
[`cerebras-harness-patterns.md`](cerebras-harness-patterns.md) and the new
Track 2 templates:

- [`src/track_2_agent_under_test_cerebras/`](../src/track_2_agent_under_test_cerebras/)
- [`src/track_2_agent_under_test_cerebras_planner/`](../src/track_2_agent_under_test_cerebras_planner/)

The older `src/track_2_agent_under_test_codex*` packages remain in the
repository temporarily for migration only and should not be used as the current
Track 2 starter.
