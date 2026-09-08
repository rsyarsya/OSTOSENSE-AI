"""Deterministic file-based integration, with the same bundled model as HTTP."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from ostosense_ai import bundled_model, live_window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("examples", "evaluate-window"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--overwrite", action="store_true")
        if name == "evaluate-window":
            command.add_argument("--input", required=True, type=Path)
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return 0 if error.code == 0 else 1
    try:
        output = args.output.resolve()
        if output.is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("Output cannot replace package source or model assets")
        if args.command == "evaluate-window" and output == args.input.resolve():
            raise ValueError("Input and output must be different files")
        if output.exists() and not args.overwrite:
            raise FileExistsError("Output exists; use a different path or --overwrite")
        bundle = bundled_model.load_bundled_model()
        if args.command == "examples":
            payload = bundle["examples"]
            code = 0
        else:
            if args.input.stat().st_size > 524288:
                raise ValueError("Input exceeds the 512 KiB request limit")
            request = json.loads(args.input.read_bytes())
            payload = live_window.evaluate_window(request, bundle["model"], bundle["model_sha256"])
            code = 0 if payload["ai"]["prediction_available"] else 2
        encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".ostosense-integration-", dir=output.parent
        ) as stage:
            staged = Path(stage) / "result.json"
            staged.write_text(encoded, encoding="utf-8", newline="")
            if args.overwrite:
                staged.replace(output)
            else:
                # An exclusive hard link closes the check/write overwrite race on local filesystems.
                os.link(staged, output)
        print(f"Wrote {args.command}: {output}")
        return code
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as error:
        print(f"Integration error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
