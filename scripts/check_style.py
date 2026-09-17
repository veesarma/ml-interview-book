"""Fail the build on AI-prose stylistic attractors.

The pattern catalogue is Ivo Velitchkov's "22 Claude-prose Patterns: A Catalog of
Stylistic Attractors in Generated Texts" (Link & Think, 25 June 2026), extended
with a vocabulary list. Each rule carries the catalogue's acronym.

    python scripts/check_style.py                    # report, exit 1 on any error
    python scripts/check_style.py --stats            # counts per pattern
    python scripts/check_style.py docs/part05 ...    # limit to paths

Two severities:

* ERROR     the formulaic construction. Any occurrence fails the build.
* DENSITY   a construction that is legitimate in technical prose but reads as a
            verbal reflex when repeated. Fails only above a per-1000-word budget,
            so one "in other words" is fine and nine are not.

Fenced code, inline code, MathJax blocks, link targets and snippet includes are
skipped, so a `not` in a code sample or a minus sign in an equation never trips it.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ERROR, DENSITY = "error", "density"

# (code, name, regex, severity, budget per 1000 words, repair)
RULES: list[tuple[str, str, str, str, float, str]] = [
    # ------------------------------------------------ the catalogue, 22 patterns
    ("SDA", "spaced-dash aside",
     r"[—]|(?<=\s)[–](?=\s)", ERROR, 0,
     "use a comma, colon, full stop or parentheses"),

    ("CB", "contrastive binary, rhetorical form",
     r"\b(?:it|this|that|which)\s*(?:'s|\s+is|\s+was)\s+not\s+(?:just\s+|merely\s+|simply\s+|only\s+)?(?:a|an|the|about)\b[^.;\n]{1,70}[,;]\s*(?:it|this|that)?\s*(?:'s|\s*is)\b"
     r"|\bnot\s+(?:just|merely|simply|only)\s+\w[^.;\n]{0,50}[,;]?\s+but\b"
     r"|\bis\s+not\s+\w[^.;\n]{1,50};\s*it\s+is\b"
     r"|\bisn'?t\s+(?:a|an|the)\b[^.;\n]{1,50}[,;]\s*it'?s\b", ERROR, 0,
     "state the positive claim: 'Decode is bound by memory bandwidth.'"),

    ("CB2", "contrastive binary, bare 'X, not Y'",
     r",\s+not\s+(?:a|an|the|its|their|his|her)?\s*\w+[.;]|\brather\s+than\b", DENSITY, 2.0,
     "fine occasionally for a factual contrast, a reflex when repeated"),

    ("SS", "significance-signaling",
     r"\b(?:this|that|which|it)\s+(?:matters?|is\s+important|is\s+significant)\s+(?:because|for|here)\b"
     r"|\bthe\s+reason\s+(?:this|that|it)\s+matters\b"
     r"|\bwhy\s+(?:this|that|it)\s+matters\s*:", ERROR, 0,
     "show the consequence instead of announcing that one is coming"),

    ("AE", "aphoristic ender",
     r"(?:^|\.\s)[A-Z][^.\n]{0,45},\s+not\s+[^.\n]{0,35}\.\s*$"
     r"|(?:^|\.\s)[A-Z][^.\n]{0,40}\sis\sthe\s\w+\.\s*$", DENSITY, 1.0,
     "do not land every section on a quotable epigram"),

    ("MCS", "mirrored-clause symmetry",
     r"\b(\w+)\b[^.;\n]{5,60};\s+\1\b", DENSITY, 1.5,
     "two clauses in one frame with slots swapped; vary the second"),

    ("MS", "meta-signposting about the document",
     r"\b(?:below\s+I|as\s+(?:noted|discussed|we\s+saw)\s+above|in\s+(?:this|the\s+next)\s+section\s+(?:we|I)"
     r"|the\s+rest\s+of\s+this\s+(?:piece|chapter|section)|as\s+we(?:'ll|\s+will)\s+see"
     r"|in\s+what\s+follows|\w+\s+caveats?\s+belong|before\s+we\s+(?:begin|start|dive))\b", ERROR, 0,
     "have a structure rather than narrating one"),

    ("SRC", "self-ranking your own claims",
     r"\bthe\s+(?:single\s+)?(?:most\s+important|deepest|cleanest|sharpest|central)\s+(?:point|thing|idea|insight|question)\b"
     r"|\bthe\s+key\s+insight\s+(?:is|here)\b", ERROR, 0,
     "let the reader judge which point is the deep one"),

    ("SH", "suspense hook",
     r"\b(?:has\s+a\s+name|there\s+is\s+a\s+name\s+for|and\s+here\s+is\s+why"
     r"|the\s+\w+\s+(?:idea|answer|version)\s+is\s+this)\b", ERROR, 0,
     "give the term immediately"),

    ("SK", "stakes-raising",
     r"\b(?:shapes?\s+everything\s+that\s+follows|everything\s+(?:downstream\s+)?depends\s+on"
     r"|the\s+stakes\s+(?:here\s+)?are|this\s+is\s+the\s+whole\s+ball\s*game)\b", ERROR, 0,
     "the payoff rarely matches the build-up"),

    ("CDF", "candor flag",
     r"\b(?:the\s+honest\s+(?:answer|truth|version)|let'?s\s+be\s+honest|to\s+be\s+honest"
     r"|honestly[,\s]|candidly|let'?s\s+face\s+it|if\s+we'?re\s+being\s+honest)\b", ERROR, 0,
     "it implies your other sentences were less honest"),

    ("VP", "validate, then promise precision",
     r"\b(?:is|that'?s)\s+(?:correct|right|true),\s+and\s+(?:it\s+can\s+be\s+made|this\s+is\s+its|here\s+is)\b",
     ERROR, 0, "let the precise version do its own work"),

    ("RF", "the reframe",
     r"\b(?:better\s+posed|better\s+asked|the\s+better\s+question\s+is|the\s+harder\s+(?:skill|question)\s+is"
     r"|put\s+more\s+precisely)\b", ERROR, 0,
     "ask your question directly instead of repositioning theirs"),

    ("CP", "corrective pivot",
     r"\bit\s+would\s+be\s+(?:wrong|a\s+mistake|unfair)\s*,?\s*(?:though|however)?\s*,?\s*to\b",
     ERROR, 0, "do not stage a small debate with yourself"),

    ("ARR", "anticipate-and-rebut reversal",
     r"\b(?:as\s+though\s+it\s+(?:carried|were|had)\b[^.\n]{0,60}\.\s+It\s+(?:does|is|carries|has)\b"
     r"|but\s+that'?s\s+not\s+all\s+it'?s\s+doing)", ERROR, 0,
     "make the claim once"),

    ("CCC", "clean-consequence connector",
     r"\b(?:falls?\s+(?:out\s+of|directly\s+out)|follows?\s+directly|comes?\s+straight\s+out\s+of"
     r"|drops?\s+out\s+of\s+(?:this|that|the))\b", ERROR, 0,
     "show the step; do not imply inevitability you have not earned"),

    ("CL", "confidence by litotes",
     r"\bnot\s+(?:difficult|hard|complicated|optional|uncommon|unusual|surprising|accidental|unlike|trivial)\b"
     r"|\bno\s+accident\b", ERROR, 0,
     "assert it plainly so the reader can disagree"),

    ("DT", "deflating tail clause",
     r",\s+and\s+(?:no|nothing)\s+(?:more|less|else)\b|\band\s+that\s+is\s+all\.", ERROR, 0,
     "a theatrical qualifier tacked on to look precise"),

    ("CF", "contribution framing",
     r"\b(?:supplies?|fills?)\s+the\s+(?:other\s+half|gap|missing\s+piece)\b"
     r"|\bpins?\s+down\s+(?:something|what)\b", ERROR, 0,
     "do not inflate the contribution by naming a gap it closes"),

    ("AHM", "reflexive AI-humility",
     r"\b(?:as\s+an\s+AI|I\s+(?:could|might)\s+be\s+wrong|structured\s+impression"
     r"|as\s+a\s+language\s+model)\b", ERROR, 0,
     "state the claim, or state its uncertainty precisely"),

    ("CR", "colon-reveal",
     r"\b(?:is|comes\s+down\s+to|boils\s+down\s+to)\s+this\s*:", DENSITY, 0.5,
     "a setup, a colon and a tidy payload builds suspense you do not need"),

    ("RG", "restatement gloss",
     r"\b(?:in\s+other\s+words|put\s+differently|that\s+is\s+to\s+say|to\s+put\s+it\s+another\s+way)\b",
     DENSITY, 1.5, "useful once to restate maths in words, a reflex when repeated"),

    ("RH", "reflexive hedging",
     r"\b(?:tends?\s+to|roughly|largely|generally\s+speaking|with\s+few\s+exceptions"
     r"|more\s+or\s+less|by\s+and\s+large)\b", DENSITY, 4.0,
     "calibration is good, uniform hedging reads as a verbal reflex"),

    # ------------------------------------------------------------- vocabulary
    ("VOC-meta", "metadiscourse filler",
     r"\bhere'?s\s+(?:the\s+(?:thing|kicker|catch|rub|key|point|problem)|where\s+it\s+gets)\b"
     r"|\blet'?s\s+(?:unpack|break\s+(?:this|it)\s+down|dive\s+in)\b"
     r"|\bthe\s+real\s+(?:question|issue|problem|answer|reason|trick|insight)\s+(?:is|here)\b"
     r"|\bit'?s\s+(?:worth\s+noting|important\s+to\s+(?:note|remember|understand))\b"
     r"|\bthink\s+of\s+it\s+(?:like|as)\b|\blet\s+me\s+(?:explain|walk\s+you\s+through|show\s+you)\b"
     r"|\bthis\s+is\s+where\s+\w+\s+(?:shines|comes\s+in|really\s+matters)\b"
     r"|\benter\s+(?:the\s+)?[A-Z][A-Za-z-]+\.", ERROR, 0,
     "delete the scaffolding and make the point"),

    ("VOC-trans", "overused connective",
     r"^\s*(?:Moreover|Furthermore|Additionally|Thus|Indeed|Notably|Crucially|Importantly)\b",
     DENSITY, 1.5, "start the sentence with its subject"),

    ("VOC-mkt", "marketing vocabulary",
     r"\b(?:game[- ]chang(?:er|ing)|paradigm\s+shift|seamless(?:ly)?|cutting[- ]edge|revolutionis|revolutioniz"
     r"|bulletproof|battle[- ]tested|first[- ]class\s+citizen|under\s+the\s+hood|delve|delving"
     r"|tapestry|landscape\s+of|realm\s+of|the\s+world\s+of|journey\s+(?:of|through)"
     r"|double[- ]edged\s+sword|silver\s+bullet|in\s+today'?s\s+\w+\s+world)\b"
     r"|\b(?:unlock|unleash|harness)\s+(?:the\s+)?(?:power|potential|full)\b"
     r"|\bcannot\s+be\s+overstated\b", ERROR, 0,
     "describe the mechanism instead"),

    ("VOC-int", "empty intensifier",
     r"\b(?:crucial|pivotal|vital|simply|trivially|obviously)\b"
     r"|\b(?:it\s+is|it's)\s+easy\s+to\s+see\b|\bas\s+we\s+all\s+know\b"
     r"|\b(?:significantly|dramatically|substantially|vastly)\s+(?:better|worse|faster|slower|improv|reduc|increas)",
     ERROR, 0, "say what breaks without it, or give the factor"),

    ("VOC-tic", "lexical tic",
     r"\b(?:genuinely|structurally|fundamentally|inherently)\b", DENSITY, 1.5,
     "adds emphasis, not information"),

    ("FMT-emoji", "decorative emoji",
     r"[✅\U0001F680\U0001F525\U0001F4A1\U0001F3AF⚡\U0001F31F\U0001F449✨\U0001F644]",
     ERROR, 0, "remove"),

    ("FMT-close", "summary-restating close",
     r"^\s*(?:In\s+summary|In\s+conclusion|To\s+summari[sz]e|The\s+takeaway|Bottom\s+line|At\s+the\s+end\s+of\s+the\s+day)\b",
     ERROR, 0, "stop on the last substantive sentence"),

    ("FMT-rq", "rhetorical question then answer",
     r"\b(?:Why\s+does\s+this\s+matter\?|What\s+does\s+this\s+mean\?|So\s+what\?|Why\s+bother\?)", ERROR, 0,
     "make the assertion directly"),
]

FENCE = re.compile(r"^(```|~~~)")
MATH = re.compile(r"^\s*\$\$")
# a complete one-line display block must not flip the parity
MATH_ONELINE = re.compile(r"^\s*\$\$.*\$\$\s*$")
SNIPPET = re.compile(r"^\s*--8<--")
INLINE_CODE = re.compile(r"`[^`]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
# the mandated chapter opener is a structural affordance, not rhetorical padding
EXEMPT_LINE = re.compile(r"^\s*>\s*\*\*Why this matters at staff level")
TABLE_ROW = re.compile(r"^\s*\|")
# Clause-shape rules read a table row as prose: "| Part VI ...; Part VII ... |"
# looks like mirrored symmetry but is a list of cells. Skip them for those rules.
CLAUSE_RULES = {"MCS", "AE", "CR"}

COMPILED = [
    (code, name, re.compile(rx, re.I), sev, budget, fix)
    for code, name, rx, sev, budget, fix in RULES
]


def prose_lines(text: str) -> list[tuple[int, str]]:
    text = HTML_COMMENT.sub("", text)
    out: list[tuple[int, str]] = []
    in_fence = in_math = False
    for i, raw in enumerate(text.splitlines(), start=1):
        if FENCE.match(raw.strip()):
            in_fence = not in_fence
            continue
        if MATH.match(raw):
            if not MATH_ONELINE.match(raw):
                in_math = not in_math
            continue
        if in_fence or in_math or SNIPPET.match(raw) or EXEMPT_LINE.match(raw):
            continue
        line = INLINE_CODE.sub(" ", raw)
        line = LINK_TARGET.sub("] ", line)
        out.append((i, line))
    return out


def check_file(path: Path):
    lines = prose_lines(path.read_text(encoding="utf-8"))
    words = sum(len(l.split()) for _, l in lines) or 1
    errors, density = [], defaultdict(list)
    for lineno, line in lines:
        is_table = bool(TABLE_ROW.match(line))
        for code, name, rx, sev, budget, fix in COMPILED:
            if is_table and code in CLAUSE_RULES:
                continue
            for m in rx.finditer(line):
                hit = (lineno, code, name, m.group(0).strip()[:60], fix)
                (errors if sev == ERROR else density[code]).append(hit)
    over = []
    for code, hits in density.items():
        budget = next(b for c, _, _, _, b, _ in COMPILED if c == code)
        per_k = 1000.0 * len(hits) / words
        # A single occurrence is never a reflex, and on a short file one hit can
        # exceed any per-1000-word budget. Require a repeat before failing.
        if len(hits) >= 2 and per_k > budget:
            over.append((code, hits, per_k, budget))
    return errors, over, words


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths] if args.paths else [ROOT / "docs"]
    files = sorted(f for r in roots for f in ([r] if r.is_file() else r.rglob("*.md")))
    # STYLE.md quotes every banned pattern as an example, and the citations queue
    # stores raw source titles, so neither can be held to the rules they describe.
    files = [f for f in files if f.name != "STYLE.md" and "_citations_todo" not in f.parts]

    n_err = n_over = 0
    counts: dict[str, int] = defaultdict(int)
    for f in files:
        errors, over, words = check_file(f)
        for _, code, *_ in errors:
            counts[code] += 1
        for code, hits, *_ in over:
            counts[code] += len(hits)
        if args.stats or not (errors or over):
            continue
        print(f"\n{_rel(f)}  ({words} words)")
        for lineno, code, name, snippet, fix in errors:
            n_err += 1
            print(f"  {lineno:>5}  [{code}] {name}")
            print(f"         found: {snippet!r}")
            print(f"         fix:   {fix}")
        for code, hits, per_k, budget in over:
            n_over += 1
            name = next(n for c, n, _, _, _, _ in COMPILED if c == code)
            fix = next(x for c, _, _, _, _, x in COMPILED if c == code)
            print(f"  DENSITY [{code}] {name}: {len(hits)} hits, "
                  f"{per_k:.1f} per 1000 words (budget {budget})")
            print(f"         lines: {', '.join(str(h[0]) for h in hits[:12])}")
            print(f"         fix:   {fix}")

    if args.stats:
        for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            name = next(nm for c, nm, _, _, _, _ in COMPILED if c == code)
            print(f"{n:>6}  [{code}] {name}")
        print(f"{sum(counts.values()):>6}  TOTAL across {len(files)} files")
        return 0

    if n_err or n_over:
        print(f"\n{n_err} errors and {n_over} over-budget patterns "
              f"across {len(files)} files. See STYLE.md section 8.")
        return 1
    print(f"Style check passed over {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
