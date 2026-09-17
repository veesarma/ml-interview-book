"""Fail the build on AI-tell writing patterns and banned phrasing.

Run over every Markdown file in docs/:

    python scripts/check_style.py            # report and exit 1 on any hit
    python scripts/check_style.py --stats     # per-pattern counts, no failure
    python scripts/check_style.py docs/part05 # limit to a path

Fenced code blocks, inline code, MathJax display blocks and HTML comments are
skipped, so a legitimate `not` in a code sample or a minus sign in an equation
never trips the checker.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (regex, human explanation, suggested repair)
RULES: list[tuple[str, str, str]] = [
    # --- the contrastive-binary template -------------------------------------
    (r"\b(?:it|this|that|the \w+)\s+(?:is|'s)\s+not\s+(?:just\s+)?(?:about\s+)?[^.;\n]{1,60}[,;]\s*(?:it|this|that)?\s*(?:is|'s)\b",
     "contrastive binary ('it's not X, it's Y')",
     "state the positive claim directly: 'Decode is bound by memory bandwidth.'"),
    (r"\bnot\s+(?:just|merely|only)\s+\w[^.;\n]{0,40}[,;]\s*but\b",
     "'not just X, but Y' template",
     "drop the template and name the thing"),
    (r"\bisn'?t\s+(?:just\s+)?(?:a|an|the)\b[^.;\n]{1,50}[,;—]\s*it'?s\b",
     "contrastive binary",
     "say what it is"),
    # --- conversational filler -----------------------------------------------
    (r"\bhere'?s\s+(?:the\s+(?:thing|kicker|catch|rub|key|point|problem)|where\s+it\s+gets)\b",
     "'here's the thing' filler",
     "delete and start with the point"),
    (r"\b(?:let'?s\s+(?:be\s+honest|face\s+it|unpack|break\s+(?:this|it)\s+down|dive\s+in)|to\s+be\s+honest|honestly[,\s]|the\s+honest\s+(?:answer|truth|version))\b",
     "false-candour filler",
     "delete; it implies the surrounding sentences were less honest"),
    (r"\bthe\s+real\s+(?:question|issue|problem|answer|reason|trick|insight|win|cost)\s+(?:is|here)\b",
     "'the real X is' reveal",
     "just state it"),
    (r"\bit'?s\s+(?:worth\s+noting|important\s+to\s+(?:note|remember|understand))\b",
     "'it's worth noting' preamble",
     "delete the preamble, keep the fact"),
    (r"\bat\s+the\s+end\s+of\s+the\s+day\b", "'at the end of the day'", "delete"),
    (r"\bthat\s+said[,\s]", "'that said' pivot (overused)", "use 'but' or restructure"),
    (r"\bthink\s+of\s+it\s+(?:like|as)\b", "'think of it like' analogy opener",
     "give the analogy without announcing it"),
    (r"\benter\s+(?:the\s+)?[A-Z][A-Za-z-]+\.", "'Enter FlashAttention.' reveal",
     "introduce it in a normal sentence"),
    (r"\bthis\s+is\s+where\s+\w+\s+(?:shines|comes\s+in|really\s+matters)\b",
     "'this is where X shines'", "state the condition under which it wins"),
    (r"\blet\s+me\s+(?:explain|walk\s+you\s+through|show\s+you)\b", "'let me explain'",
     "delete and explain"),
    (r"\bwhether\s+you'?re\s+(?:a|an)\b", "'whether you're a X or a Y'", "address the reader once"),
    # --- vocabulary slop ------------------------------------------------------
    (r"\b(?:delve|delving)\b", "'delve'", "use 'look at', 'work through'"),
    (r"\b(?:tapestry|landscape\s+of|realm\s+of|the\s+world\s+of|journey\s+(?:of|through))\b",
     "abstract-noun slop", "name the actual subject"),
    (r"\b(?:unlock|unleash|harness)\s+(?:the\s+)?(?:power|potential|full)\b",
     "'unlock the power of'", "say what it does"),
    (r"\b(?:game[- ]chang(?:er|ing)|paradigm\s+shift|seamless(?:ly)?|cutting[- ]edge|state[- ]of[- ]the[- ]art\s+\(SOTA\)|revolutioniz|bulletproof|battle[- ]tested|first[- ]class\s+citizen|under\s+the\s+hood)\b",
     "marketing vocabulary", "describe the mechanism instead"),
    (r"\bcannot\s+be\s+overstated\b", "'cannot be overstated'", "give the magnitude"),
    (r"\bin\s+today'?s\s+\w+\s+world\b", "'in today's ... world'", "delete"),
    (r"\b(?:crucial|pivotal|vital)\b", "intensifier filler",
     "say why it matters, or cut the adjective"),
    (r"\b(?:significantly|dramatically|substantially|vastly)\s+(?:better|worse|faster|slower|improv|reduc|increas)",
     "unquantified intensifier", "give the number or say 'faster'"),
    (r"\bdouble[- ]edged\s+sword\b", "'double-edged sword'", "name both effects"),
    (r"\bsilver\s+bullet\b", "'silver bullet'", "say 'no single choice wins everywhere'"),
    # --- structural tics ------------------------------------------------------
    (r"—", "em dash",
     "use a comma, a full stop, or parentheses; em dashes are the loudest AI tell"),
    (r"^\s*(?:In\s+summary|In\s+conclusion|To\s+summarize|The\s+takeaway|Bottom\s+line)\b",
     "summary-restating close", "end on the last substantive sentence"),
    (r"\b(?:Why\s+does\s+this\s+matter\?|What\s+does\s+this\s+mean\?|So\s+what\?)\s",
     "rhetorical question then answer", "make the assertion directly"),
    (r"\bnot\s+only\b[^.;\n]{1,60}\bbut\s+also\b", "'not only ... but also'", "use 'and'"),
    (r"[✅🚀🔥💡🎯⚡️🌟👉✨]", "decorative emoji", "remove"),
    (r"\bas\s+we\s+all\s+know\b", "'as we all know'", "delete"),
    (r"\b(?:simply|trivially|just\s+)?(?:it\s+is|it's)\s+easy\s+to\s+see\b",
     "'it is easy to see'", "show the step"),
    (r"\b(?:simply|trivially)\b", "'simply' / 'trivially'", "delete or show the step"),
    (r"\bI\s+(?:could\s+be\s+wrong|might\s+be\s+wrong|think\s+it'?s\s+worth)\b",
     "hedging tic", "state the claim or its uncertainty precisely"),
    (r"\bas\s+an\s+AI\b", "self-reference as an AI", "delete"),
]

FENCE = re.compile(r"^(```|~~~)")
MATH = re.compile(r"^\s*\$\$")
INLINE_CODE = re.compile(r"`[^`]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
SNIPPET = re.compile(r"^\s*--8<--")


def strip_noncontent(text: str) -> list[tuple[int, str]]:
    """Return (line_number, prose_only_line) skipping code, math and includes."""
    text = HTML_COMMENT.sub("", text)
    out: list[tuple[int, str]] = []
    in_fence = False
    in_math = False
    for i, raw in enumerate(text.splitlines(), start=1):
        if FENCE.match(raw.strip()):
            in_fence = not in_fence
            continue
        if in_fence or SNIPPET.match(raw):
            continue
        if MATH.match(raw):
            in_math = not in_math
            continue
        if in_math:
            continue
        line = INLINE_CODE.sub(" ", raw)
        line = LINK_TARGET.sub("] ", line)
        out.append((i, line))
    return out


def check_file(path: Path) -> list[tuple[int, str, str, str]]:
    hits = []
    for lineno, line in strip_noncontent(path.read_text(encoding="utf-8")):
        for pattern, label, fix in RULES:
            m = re.search(pattern, line, flags=re.I | re.M)
            if m:
                hits.append((lineno, label, m.group(0).strip()[:70], fix))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--stats", action="store_true", help="counts only, always exit 0")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths] if args.paths else [ROOT / "docs"]
    files = sorted(
        f for r in roots for f in ([r] if r.is_file() else r.rglob("*.md"))
    )

    total = 0
    counts: dict[str, int] = {}
    for f in files:
        hits = check_file(f)
        if not hits:
            continue
        if not args.stats:
            print(f"\n{f.relative_to(ROOT)}")
        for lineno, label, snippet, fix in hits:
            counts[label] = counts.get(label, 0) + 1
            total += 1
            if not args.stats:
                print(f"  {lineno:>5}  {label}")
                print(f"         found: {snippet!r}")
                print(f"         fix:   {fix}")

    if args.stats:
        for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"{n:>5}  {label}")
        print(f"{total:>5}  TOTAL across {len(files)} files")
        return 0

    if total:
        print(f"\n{total} style violations in {len(files)} files. See STYLE.md §9.")
        return 1
    print(f"Style check passed over {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
