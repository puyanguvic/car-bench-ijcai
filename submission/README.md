# CAR-bench final-submission package

This directory contains final-submission scenario templates for both Track 2
agents. Choose **one** template after validating the release candidate, replace
the GHCR placeholder with the public image digest, and save the result as
`scenario.toml` for the organizer submission form.

Do not put any API key or secret value in the final TOML. It must contain only
environment-variable interpolation expressions.

## Release checklist

1. Build the selected agent for `linux/amd64`.
2. Run the manual GitHub Actions workflow
   [`publish-track2-ghcr.yml`](../.github/workflows/publish-track2-ghcr.yml)
   to push the image to GHCR. It grants `packages: write` only to that job and
   records the immutable digest in the workflow summary.
3. Set the newly created GHCR package to **Public**. Then render the final
   scenario without manually editing it:

   ```bash
   uv run python scripts/prepare_track2_submission.py \
     --variant direct \
     --image-digest sha256:REPLACE_WITH_64_HEX_CHAR_DIGEST \
     --output dist/track2-direct/scenario.toml
   ```

   The utility rejects mutable tags, literal secrets, non-hidden task splits,
   and incorrect task counts.
4. Run the matching GHCR smoke scenario and retain its output as release
   evidence.
5. Submit the resulting `scenario.toml`, selected Track 2, and the required
   four-page IJCAI-format technical report through the organizer form.

The direct template allows at most five sequential executor calls per
benchmark step. The planner/executor template allows at most five sequential
calls in the worst malformed-output path. Both report aggregate usage via the
required A2A `turn_metrics` fields.

Use [`technical_report_outline.md`](technical_report_outline.md) to prepare
the required four-page report and compute-audit diagram.
