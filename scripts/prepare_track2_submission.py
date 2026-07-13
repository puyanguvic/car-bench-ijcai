#!/usr/bin/env python3
"""Render and validate an immutable CAR-bench Track 2 scenario.toml."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "direct": ROOT / "submission/track_2_direct/scenario.toml.template",
    "planner_executor": (
        ROOT / "submission/track_2_planner_executor/scenario.toml.template"
    ),
}
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}$")
PLACEHOLDER = "sha256:REPLACE_WITH_PUBLIC_IMAGE_DIGEST"
TASK_COUNT_KEYS = (
    "tasks_base_num_tasks",
    "tasks_hallucination_num_tasks",
    "tasks_disambiguation_num_tasks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an immutable Track 2 final-submission scenario TOML."
    )
    parser.add_argument("--variant", choices=sorted(TEMPLATES), required=True)
    parser.add_argument(
        "--image-digest",
        required=True,
        help="Published GHCR digest in sha256:<64 lowercase hex chars> form.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination scenario.toml path. Existing files require --force.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def validate_submission(data: dict, *, digest: str) -> None:
    image = data["agent_under_test"]["image"]
    if not image.endswith(f"@{digest}"):
        raise ValueError("Agent image is not pinned to the requested digest.")

    config = data["config"]
    if config.get("task_split") != "hidden":
        raise ValueError('Final submission must set task_split = "hidden".')
    if config.get("num_trials") != 3:
        raise ValueError("Final submission must set num_trials = 3.")
    for key in TASK_COUNT_KEYS:
        if config.get(key) != -1:
            raise ValueError(f"Final submission must set {key} = -1.")

    for section in ("evaluator", "agent_under_test"):
        for key, value in data[section].get("env", {}).items():
            if not isinstance(value, str) or "${" not in value:
                raise ValueError(
                    f"{section}.env.{key} must be an environment-variable reference, "
                    "not a literal value."
                )


def main() -> None:
    args = parse_args()
    if not DIGEST_PATTERN.fullmatch(args.image_digest):
        raise SystemExit("--image-digest must be sha256: followed by 64 lowercase hex chars.")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {args.output}. Use --force.")

    template = TEMPLATES[args.variant]
    content = template.read_text()
    if PLACEHOLDER not in content:
        raise SystemExit(f"Template placeholder missing from {template}.")
    rendered = content.replace(PLACEHOLDER, args.image_digest)
    data = tomllib.loads(rendered)
    validate_submission(data, digest=args.image_digest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote validated Track 2 {args.variant} scenario: {args.output}")
    print(f"Pinned image: {data['agent_under_test']['image']}")


if __name__ == "__main__":
    main()
