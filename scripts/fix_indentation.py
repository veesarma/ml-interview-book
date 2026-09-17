"""Repair admonition and content-tab bodies that lost their indentation.

    python scripts/fix_indentation.py docs/part04-vision
    python scripts/fix_indentation.py --dry-run

An earlier version of fix_style.py collapsed runs of spaces, which stripped the
4-space indent that MkDocs requires for the body of a `!!! note` / `??? success`
block or a `=== "NumPy"` content tab. Without the indent the body renders as a
plain paragraph outside the box.

This re-indents any under-indented body line that belongs to such a block, using
the opener's indent plus four. It stops at the first line that is clearly outside
the block: a heading, a new opener at the same level, a horizontal rule, or a
second consecutive blank line followed by unindented text.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OPENER = re.compile(r'^(?P<indent>\s*)(?:(?:!!!|\?\?\?\+?)\s+[\w-]+(?:\s+"[^"]*")?|===\s+"[^"]*")\s*$')
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^#{1,6}\s")
RULE = re.compile(r"^\s*(?:---|\*\*\*|___)\s*$")


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def repair(path: Path, dry_run: bool) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = list(lines)
    fixed = 0
    i = 0
    in_fence = False

    while i < len(lines):
        raw = lines[i]
        if FENCE.match(raw):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            i += 1
            continue

        m = OPENER.match(raw)
        if not m:
            i += 1
            continue

        want = len(m.group("indent")) + 4
        j = i + 1
        blanks = 0
        block_fence = False
        while j < len(lines):
            line = lines[j]
            if FENCE.match(line):
                block_fence = not block_fence
                # a fence inside the block still needs the body indent
                if line.strip() and (len(line) - len(line.lstrip())) < want:
                    out[j] = " " * want + line.lstrip()
                    fixed += 1
                j += 1
                continue
            if block_fence:
                j += 1
                continue
            if not line.strip():
                blanks += 1
                if blanks >= 2:
                    break
                j += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0 and (HEADING.match(line) or RULE.match(line) or OPENER.match(line)):
                break
            if blanks and indent == 0:
                break
            blanks = 0
            if indent < want:
                out[j] = " " * want + line.lstrip()
                fixed += 1
            j += 1
        i = j

    if fixed and not dry_run:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    if fixed:
        print(f"{_rel(path)}: re-indented {fixed} lines")
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    roots = [Path(p) for p in args.paths] if args.paths else [ROOT / "docs"]
    files = sorted(f for r in roots for f in ([r] if r.is_file() else r.rglob("*.md")))
    total = sum(repair(f, args.dry_run) for f in files)
    print(f"\n{total} lines {'would be ' if args.dry_run else ''}re-indented across {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
