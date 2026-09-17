"""Turn printed arXiv identifiers into links to their abstract pages.

    python scripts/linkify_arxiv.py --dry-run
    python scripts/linkify_arxiv.py docs/part04-vision

A chapter that already prints `arXiv:1512.03385` has made its claim: that this
identifier denotes the paper under discussion. Rewriting it as
`[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)` adds no new claim, it only
makes the existing one clickable, so this is safe to do mechanically. It does NOT
substitute for verifying that the identifier is the right one; that needs a search
pass, and `--list-ids` prints the distinct identifiers for exactly that purpose.

Skipped: identifiers already inside a link, inside code fences or inline code,
and inside the citations queue.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# modern (0704.0001 onward) and legacy (math/0211159) identifiers
ARXIV = re.compile(
    r"arXiv:\s?((?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?))",
    re.I,
)
FENCE = re.compile(r"^\s*(```|~~~)")
ALREADY_LINKED = re.compile(r"\]\([^)]*arxiv\.org[^)]*\)|\(https?://arxiv\.org", re.I)


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def process(path: Path, dry_run: bool, ids: Counter) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    changed = 0
    in_fence = False

    for raw in lines:
        if FENCE.match(raw):
            in_fence = not in_fence
            out.append(raw)
            continue
        if in_fence or "arxiv" not in raw.lower():
            out.append(raw)
            continue

        # protect inline code and existing links by splitting on them
        pieces = re.split(r"(`[^`]*`|\[[^\]]*\]\([^)]*\))", raw)
        for i, piece in enumerate(pieces):
            if i % 2 == 1:  # a protected span
                continue
            if ALREADY_LINKED.search(piece):
                continue

            def repl(m: re.Match) -> str:
                ident = m.group(1)
                ids[ident.split("v")[0]] += 1
                return f"[arXiv:{ident}](https://arxiv.org/abs/{ident})"

            new_piece, n = ARXIV.subn(repl, piece)
            if n:
                pieces[i] = new_piece
                changed += n
        out.append("".join(pieces))

    if changed and not dry_run:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    if changed:
        print(f"{_rel(path)}: {changed} identifiers linked")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-ids", action="store_true",
                    help="print distinct identifiers for a verification pass")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths] if args.paths else [ROOT / "docs"]
    files = sorted(f for r in roots for f in ([r] if r.is_file() else r.rglob("*.md")))
    files = [f for f in files if "_citations_todo" not in f.parts]

    ids: Counter = Counter()
    total = sum(process(f, args.dry_run or args.list_ids, ids) for f in files)

    if args.list_ids:
        for ident, n in sorted(ids.items()):
            print(f"{ident}\t{n}")
        print(f"\n{len(ids)} distinct identifiers, {total} occurrences", file=sys.stderr)
        return 0

    print(f"\n{total} identifiers {'would be ' if args.dry_run else ''}linked "
          f"across {len(files)} files ({len(ids)} distinct).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
