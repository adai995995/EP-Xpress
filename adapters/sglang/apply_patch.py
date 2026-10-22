#!/usr/bin/env python3
"""Apply the integration only to its audited SGLang source revision."""

import argparse
from pathlib import Path
import subprocess


REVISION = "e00930c5489053f26d86b179cee0d087f846acbb"


def git(source, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=check, text=True, capture_output=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SGLang source checkout")
    parser.add_argument("--check", action="store_true", help="Validate without changing files")
    args = parser.parse_args()
    patch = Path(__file__).with_name("sglang-v0.5.21.patch").resolve()
    source = args.source.resolve()
    if git(source, "rev-parse", "HEAD").stdout.strip() != REVISION:
        raise SystemExit(f"Unsupported SGLang revision; expected {REVISION}")
    if git(source, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        print("EP-Xpress SGLang patch is already applied")
        return
    targets = [line[6:] for line in patch.read_text().splitlines() if line.startswith("+++ b/")]
    for command in (("diff", "--name-only"), ("diff", "--cached", "--name-only")):
        if git(source, *command, "--", *targets).stdout.strip():
            raise SystemExit("Integration target files contain local changes; preserve them before applying")
    git(source, "apply", "--check", str(patch))
    if args.check:
        print("EP-Xpress SGLang patch passes git apply --check")
    else:
        git(source, "apply", str(patch))
        print("Applied EP-Xpress SGLang integration; install both projects in the same environment")


if __name__ == "__main__":
    main()
