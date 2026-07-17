# CAR-bench Track 2 final-submission package

The only final agent in this package is PACT, built from
`src/track_2_agent_under_test_cerebras/`. Its submission template is
[`track_2_direct/scenario.toml.template`](track_2_direct/scenario.toml.template).
The `track_2_planner_executor/` directory is archived implementation history;
do **not** build, render, validate, or submit that variant.

Do not put any API key or secret value in the final TOML. It must contain only
environment-variable interpolation expressions.

## Release checklist

1. Build the PACT agent for `linux/amd64`.
2. Run the manual GitHub Actions workflow
   [`publish-track2-ghcr.yml`](../.github/workflows/publish-track2-ghcr.yml)
   to push the PACT image to GHCR. It grants `packages: write` only to that job
   and records the immutable digest in the workflow summary.
3. Set the newly created GHCR package to **Public**. Then render the final
   scenario without manually editing it:

   ```bash
   uv run python scripts/prepare_track2_submission.py \
     --variant direct \
     --image-digest "${PACT_DIGEST}" \
     --output dist/track2-direct/scenario.toml
   ```

   Set `PACT_DIGEST` to the actual `sha256:` value recorded by the publish
   workflow. The utility rejects mutable tags, literal secrets, non-hidden task
   splits, and incorrect task counts.
4. Put the same digest and PACT environment block into
   `scenarios/track_2_agent_under_test_cerebras/ghcr_smoke.toml`, and run README
   option C. Retain the output as release evidence.
5. Submit the resulting `scenario.toml`, selected Track 2, and the required
   four-page IJCAI-format technical report through the organizer form.

PACT makes one compiler call and permits at most one bounded repair for each
compilation. It reports aggregate usage through the required A2A
`turn_metrics` fields, including prompt, completion, and thinking tokens.

Use [`technical_report_outline.md`](technical_report_outline.md) to prepare
the required four-page report and compute-audit diagram.
