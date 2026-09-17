"""Structural checks on the book's Markdown that MkDocs will not catch.

    python scripts/check_markdown.py            # report, exit 1 on any problem
    python scripts/check_markdown.py docs/part06-llm-training

Checks:

1. **Admonition bodies are indented.** A `!!! note` / `??? success` block whose
   body sits at fewer than 4 spaces renders as a paragraph outside the box. This
   is the exact damage an earlier version of fix_style.py caused, so it is worth
   a standing check rather than a one-off sweep.
2. **Content-tab bodies are indented.** Same failure for `=== "NumPy"` blocks.
3. **Code fences are balanced** per file.
4. **Snippet includes resolve.** `--8<-- "path"` must point at a real file.
5. **Figure references resolve.** Every `](../assets/figures/x.png)` exists.
6. **No dangling punctuation** left by an automated rewrite: lines ending in a
   bare comma or semicolon, or containing `::` or ` ,`.
7. **Mandatory chapter sections are present** in numbered chapter files.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

ADMONITION = re.compile(r'^(?P<indent>\s*)(?:!!!|\?\?\?\+?)\s+[\w-]+(?:\s+"[^"]*")?\s*$')
TAB = re.compile(r'^(?P<indent>\s*)===\s+"')
FENCE = re.compile(r"^\s*(```|~~~)")
SNIPPET = re.compile(r'--8<--\s*"([^"]+)"')
IMG = re.compile(r"!\[[^\]]*\]\(([^)]+\.(?:png|jpg|jpeg|gif|svg))")
DANGLING = re.compile(r"[,;]\s*$")
MANDATORY = ["## TL;DR", "## 1.", "## 2.", "## 3.", "## Retype by hand",
             "## 4.", "## 5.", "## 6.", "## 7.", "## References"]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    problems: list[str] = []

    # fences
    n_fence = sum(1 for l in lines if FENCE.match(l))
    if n_fence % 2:
        problems.append(f"{n_fence} code-fence markers (odd, so one is unclosed)")

    in_fence = False
    for i, raw in enumerate(lines):
        if FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # 1 + 2. block bodies must be indented past their opener
        m = ADMONITION.match(raw) or TAB.match(raw)
        if m:
            opener_indent = len(m.group("indent"))
            # find the next non-blank line
            for nxt in lines[i + 1 : i + 6]:
                if not nxt.strip():
                    continue
                body_indent = len(nxt) - len(nxt.lstrip())
                if body_indent < opener_indent + 4:
                    kind = "admonition" if ADMONITION.match(raw) else "content tab"
                    problems.append(
                        f"line {i + 1}: {kind} body indented {body_indent}, "
                        f"needs {opener_indent + 4}: {nxt.strip()[:60]!r}"
                    )
                break

        # 6. dangling punctuation left by an automated rewrite. A trailing comma
        #    mid-paragraph is ordinary soft wrapping, so only flag one that ends
        #    a paragraph (next line blank or end of file).
        stripped = raw.strip()
        at_para_end = (i + 1 >= len(lines)) or not lines[i + 1].strip()
        if stripped and at_para_end and DANGLING.search(stripped) and not stripped.startswith("|"):
            problems.append(f"line {i + 1}: paragraph ends in bare punctuation: {stripped[-50:]!r}")
        if "::" in raw and "http" not in raw and "::=" not in raw and ":::" not in raw:
            problems.append(f"line {i + 1}: double colon: {stripped[:60]!r}")
        if re.search(r"\w\s+,", raw):
            problems.append(f"line {i + 1}: space before comma: {stripped[:60]!r}")

    # 4. snippets
    for m in SNIPPET.finditer(text):
        target = ROOT / m.group(1)
        if not target.exists():
            problems.append(f"snippet include missing: {m.group(1)}")

    # 5. figures
    for m in IMG.finditer(text):
        ref = m.group(1).split()[0]
        if ref.startswith("http"):
            continue
        target = (path.parent / ref).resolve()
        if not target.exists():
            problems.append(f"figure missing: {ref}")

    # 7. mandatory sections in numbered chapters
    if re.match(r"^\d\d-", path.name) and "part1" not in path.parent.name[:5]:
        missing = [s for s in MANDATORY if s not in text]
        if len(missing) > 3:
            problems.append(f"missing chapter sections: {', '.join(missing)}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()
    roots = [Path(p) for p in args.paths] if args.paths else [DOCS]
    files = sorted(f for r in roots for f in ([r] if r.is_file() else r.rglob("*.md")))
    files = [f for f in files if f.name != "STYLE.md" and "_citations_todo" not in f.parts]

    total = 0
    for f in files:
        problems = check(f)
        if problems:
            print(f"\n{_rel(f)}")
            for p in problems:
                print(f"  {p}")
            total += len(problems)

    if total:
        print(f"\n{total} structural problems across {len(files)} files.")
        return 1
    print(f"Markdown structure OK across {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
