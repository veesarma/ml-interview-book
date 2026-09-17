"""Mechanically remove em dashes where the replacement is unambiguous.

    python scripts/fix_style.py docs/part05-sequence-transformers
    python scripts/fix_style.py docs/part04-vision/04-detection.md --dry-run

Handles the four positions where the right substitute is decided by position
rather than by meaning:

* markdown headings            `## Detection, anchors` -> `## Detection: anchors`
* admonition / quoted titles   `!!! production "Meta, SAM"` -> `"Meta: SAM"`
* table cells                  `| **Yes**, the order matters |` -> `| **Yes**: ... |`
* reference list entries       `"Title" (2023), [link]` -> `"Title" (2023). [link]`

In body prose it converts a paired dash to parentheses and a single dash to a
comma, which is correct English but not always the *best* sentence. Re-read
every prose line it touches; the script prints each one.

Code fences, math blocks and snippet includes are never modified.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = "—"

FENCE = re.compile(r"^(```|~~~)")
MATH = re.compile(r"^\s*\$\$")
# a complete one-line display block must not flip the parity
MATH_ONELINE = re.compile(r"^\s*\$\$.*\$\$\s*$")
SNIPPET = re.compile(r"^\s*--8<--")
HEADING = re.compile(r"^#{1,6}\s")
TABLE_ROW = re.compile(r"^\s*\|")
LIST_ITEM = re.compile(r"^\s*(?:[*+-]|\d+\.)\s")
ADMONITION = re.compile(r'^\s*(?:!!!|\?\?\?\+?)\s+\w+\s+"')


def fix_line(line: str) -> tuple[str, bool]:
    """Return (new_line, touched_prose) for one non-code line."""
    if DASH not in line:
        return line, False

    # 1. headings and admonition titles: the dash introduces a gloss -> colon
    if HEADING.match(line) or ADMONITION.match(line):
        out = re.sub(rf"\s*{DASH}\s*", ": ", line, count=1)
        out = re.sub(rf"\s*{DASH}\s*", ", ", out)          # any further dashes
        return re.sub(r"::+", ":", out), False

    # 2. table cells: gloss after a verdict -> colon
    if TABLE_ROW.match(line):
        out = re.sub(rf"\s*{DASH}\s*", ": ", line)
        return re.sub(r"::+", ":", out), False

    # 3. reference entries: "Title" (year) — [link]  -> "Title" (year). [link]
    if LIST_ITEM.match(line) and "](" in line:
        indent = line[: len(line) - len(line.lstrip())]
        out = re.sub(rf"\s*{DASH}\s*", ". ", line.lstrip())
        return indent + re.sub(r"\.\s*\.", ".", out).rstrip(), False

    # 4. prose: paired dashes become parentheses, a single dash becomes a comma.
    #    Leading whitespace is preserved verbatim: admonition bodies, list
    #    continuations and indented blocks all depend on their exact indent.
    indent = line[: len(line) - len(line.lstrip())]
    body = line[len(indent) :]

    n = body.count(DASH)
    if n >= 2:
        out = re.sub(rf"\s*{DASH}\s*(.*?)\s*{DASH}\s*", r" (\1) ", body, count=n // 2)
    else:
        out = re.sub(rf"\s*{DASH}\s*", ", ", body, count=1)

    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"[ ]{2,}", " ", out)
    # a dash that introduced a link or ended the line leaves a dangling joint
    out = re.sub(r",\s*$", "", out)
    out = re.sub(r",\s*(?=\[)", ". ", out)
    out = re.sub(r"::+", ":", out)
    return indent + out.rstrip(), True


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def process(path: Path, dry_run: bool) -> int:
    text = path.read_text(encoding="utf-8")
    if DASH not in text:
        return 0

    lines = text.splitlines(keepends=False)
    out: list[str] = []
    in_fence = in_math = False
    changed = 0
    prose_touched: list[tuple[int, str, str]] = []

    for i, raw in enumerate(lines, start=1):
        if FENCE.match(raw.strip()):
            in_fence = not in_fence
            out.append(raw)
            continue
        if MATH.match(raw):
            if not MATH_ONELINE.match(raw):
                in_math = not in_math
            out.append(raw)
            continue
        if in_fence or in_math or SNIPPET.match(raw):
            out.append(raw)
            continue

        new, prose = fix_line(raw)
        if new != raw:
            changed += 1
            if prose:
                prose_touched.append((i, raw.strip(), new.strip()))
        out.append(new)

    if changed and not dry_run:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")

    if changed:
        print(f"{_rel(path)}: {changed} lines")
        for lineno, before, after in prose_touched:
            print(f"  prose {lineno}: {before[:90]}")
            print(f"     ->  {after[:90]}")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for p in args.paths:
        path = Path(p)
        files = [path] if path.is_file() else sorted(path.rglob("*.md"))
        for f in files:
            total += process(f, args.dry_run)
    print(f"\n{total} lines {'would be' if args.dry_run else ''} changed.")
    print("Re-read every prose line listed above; a comma is correct but rarely best.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
