#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pre-commit hook to enforce a maximum line count per Python file.

Prevents files from growing beyond 1000 lines. Generated files and
vendored specs are excluded.
"""

import sys
from pathlib import Path

MAX_LINES = 1000

EXCLUDE_PATTERNS = [
    "docs/static/",
    "client-sdks/stainless/",
    "tests/integration/recordings/",
]


def main() -> int:
    failures: list[tuple[str, int]] = []

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.suffix == ".py":
            continue
        if any(pattern in path_str for pattern in EXCLUDE_PATTERNS):
            continue
        try:
            line_count = sum(1 for _ in path.open())
        except OSError:
            continue
        if line_count > MAX_LINES:
            failures.append((path_str, line_count))

    if failures:
        print(f"Files exceeding {MAX_LINES} lines:")
        for path_str, count in sorted(failures):
            print(f"  {path_str}: {count} lines")
        print(f"\nSplit large files into focused modules to stay under {MAX_LINES} lines.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
