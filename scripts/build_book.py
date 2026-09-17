"""Assemble the whole book into one file for e-readers and print.

    python scripts/build_book.py --epub          # ML-Interview-Book.epub
    python scripts/build_book.py --pdf           # ML-Interview-Book.pdf
    python scripts/build_book.py --html          # one self-contained HTML file
    python scripts/build_book.py --all
    python scripts/build_book.py --part part05   # just one part, for a quick look

The MkDocs site is the primary format. This exists for the cases a website cannot
serve: reading on a Kindle or iPad offline, printing a part to annotate, or
handing someone a single file.

What it does to make that possible, since the chapters are written for MkDocs
Material and pandoc understands none of its extensions:

* walks `nav` in mkdocs.yml so the order matches the site exactly;
* expands `--8<--` snippet includes;
* converts `!!!` and `???` admonitions to titled block quotes, keeping the body;
* converts `=== "Tab"` content tabs to sub-headings;
* rewrites image paths to absolute paths so pandoc can embed them;
* turns relative chapter links into internal anchors;
* leaves TeX math alone, which pandoc renders natively in every target;
* labels mermaid blocks, which no offline format can draw, and points at the site.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BUILD = ROOT / "build"
TITLE = "The ML Interview Book"
SUBTITLE = "From first principles to the frontier, for staff-level ML interviews"

NAV_ENTRY = re.compile(r"^(\s*)-\s+(?:(.+?):\s*)?((?:preface|part\d\d|references|index)[^\s]*\.md)\s*$")
PART_HEADING = re.compile(r"^(\s*)-\s+([IVXL]+\.[^:]*|Preface|Home|References):\s*$")
SNIPPET = re.compile(r'^(\s*)--8<--\s*"([^"]+)"\s*$')
ADMONITION = re.compile(r'^(\s*)(?:!!!|\?\?\?\+?)\s+([\w-]+)(?:\s+"([^"]*)")?\s*$')
TAB = re.compile(r'^(\s*)===\s+"([^"]*)"\s*$')
FENCE = re.compile(r"^\s*(```|~~~)")
IMAGE = re.compile(r"!\[([^\]]*)\]\((\.\./[^)\s]+|assets/[^)\s]+)(\s+\{[^}]*\})?\)")
MD_LINK = re.compile(r"\[([^\]]+)\]\((?!https?://)([^)#]+)\.md(#[^)]*)?\)")
ATTR_LIST = re.compile(r"\{\s*\.[^}]*\}")

# The chapters use the macros defined for MathJax in docs/javascripts/mathjax.js.
# Pandoc knows nothing about that file, so the macros are expanded textually here.
# Keeping the table next to the source it mirrors means one place to update.
ZERO_ARG_MACROS = {
    r"\R": r"\mathbb{R}",
    r"\Var": r"\operatorname{Var}",
    r"\Cov": r"\operatorname{Cov}",
    r"\E": r"\mathbb{E}",
    r"\KL": r"D_{\mathrm{KL}}",
    r"\softmax": r"\operatorname{softmax}",
    r"\argmax": r"\operatorname{arg\,max}",
    r"\argmin": r"\operatorname{arg\,min}",
    r"\tr": r"\operatorname{tr}",
    r"\diag": r"\operatorname{diag}",
}
ONE_ARG_MACROS = {
    r"\norm": (r"\left\lVert ", r" \right\rVert"),
}



# MathJax renders these; pandoc's texmath does not implement them. They are
# rewritten only for the offline export, so the site keeps the nicer spacing.
UNSUPPORTED_TEX = [
    (re.compile(r"\\operatorname\*\{\\vphantom\{[^}]*\}([A-Za-z]+)\}"), r"\\operatorname{\1}"),
    (re.compile(r"\\vphantom\{[^}]*\}"), ""),
    (re.compile(r"\\(?:Bigg|Big|bigg|big)l\b"), r"\\left"),
    (re.compile(r"\\(?:Bigg|Big|bigg|big)r\b"), r"\\right"),
]


def expand_math_macros(text: str) -> str:
    for macro, (pre, post) in ONE_ARG_MACROS.items():
        out, i = [], 0
        while True:
            j = text.find(macro + "{", i)
            if j < 0:
                out.append(text[i:])
                break
            out.append(text[i:j])
            k = j + len(macro) + 1
            depth = 1
            while k < len(text) and depth:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            out.append(pre + text[j + len(macro) + 1 : k - 1] + post)
            i = k
        text = "".join(out)

    for pattern, repl in UNSUPPORTED_TEX:
        text = pattern.sub(repl, text)

    for macro, expansion in sorted(ZERO_ARG_MACROS.items(), key=lambda kv: -len(kv[0])):
        # a macro name must not swallow a longer one, so require a non-letter after it
        # Braced, so \beta\KL does not expand into \betaD_{...}
        text = re.sub(re.escape(macro) + r"(?![A-Za-z])",
                      lambda _m, e=expansion: "{" + e + "}", text)
    return text


def read_nav() -> list[tuple[str, Path]]:
    """Return [(part label, chapter path)] in the order the site presents them."""
    out: list[tuple[str, Path]] = []
    current = ""
    for line in (ROOT / "mkdocs.yml").read_text().splitlines():
        ph = PART_HEADING.match(line)
        if ph:
            current = ph.group(2).strip()
            continue
        m = NAV_ENTRY.match(line)
        if m:
            rel = m.group(3)
            path = DOCS / rel
            if path.exists():
                out.append((current, path))
    return out


def expand_snippets(text: str, depth: int = 0) -> str:
    """Inline `--8<--` includes.

    Chapters almost always put the directive inside their own ```python fence.
    Adding a second fence around the file closes the outer block early, and the
    chapter's own closing fence then opens a new one that swallows everything
    after it. So only fence the content when the directive is not already
    inside one.
    """
    if depth > 3:
        return text
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        m = SNIPPET.match(line)
        if not m:
            out.append(line)
            continue
        indent, target = m.group(1), m.group(2)
        path = ROOT / target
        if not path.exists():
            out.append(f"{indent}*(missing include: {target})*")
            continue
        body = expand_snippets(path.read_text(encoding="utf-8"), depth + 1)
        if path.suffix == ".py" and not in_fence:
            out.append(f"{indent}```python")
            out.extend(indent + l for l in body.splitlines())
            out.append(f"{indent}```")
        else:
            out.extend(indent + l for l in body.splitlines())
    return "\n".join(out)


def dedent_block(lines: list[str], indent: int) -> list[str]:
    return [l[indent:] if len(l) > indent and l[:indent].isspace() else l.lstrip() if l.strip() else l
            for l in lines]


def convert_blocks(text: str) -> str:
    """Admonitions and content tabs into plain Markdown pandoc understands."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    in_fence = False

    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        adm = ADMONITION.match(line)
        tab = TAB.match(line)
        if not (adm or tab):
            out.append(line)
            i += 1
            continue

        opener_indent = len((adm or tab).group(1))
        want = opener_indent + 4
        if adm:
            kind = adm.group(2)
            title = adm.group(3) if adm.group(3) is not None else kind.replace("-", " ").title()
        else:
            kind, title = "tab", tab.group(2)

        body: list[str] = []
        j = i + 1
        blanks = 0
        body_fence = False
        while j < len(lines):
            nxt = lines[j]
            if FENCE.match(nxt):
                body_fence = not body_fence
            if not nxt.strip():
                blanks += 1
                if blanks >= 2 and not body_fence:
                    break
                body.append("")
                j += 1
                continue
            indent = len(nxt) - len(nxt.lstrip())
            if indent < want and not body_fence:
                break
            blanks = 0
            body.append(nxt)
            j += 1

        inner = convert_blocks("\n".join(dedent_block(body, want)))
        if tab:
            out.append("")
            out.append(f"**{title}**")
            out.append("")
            out.extend(inner.splitlines())
        else:
            label = {"interview": "Interview question", "production": "In production",
                     "success": "Solution", "example": "Example"}.get(kind, kind.title())
            out.append("")
            out.append(f"> **{label}: {title}**" if title and title != label else f"> **{label}**")
            out.append(">")
            out.extend(("> " + l).rstrip() for l in inner.splitlines())
            out.append("")
        i = j

    return "\n".join(out)


def rewrite_links(text: str, path: Path) -> str:
    def img(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        resolved = (path.parent / src).resolve() if src.startswith("..") else (DOCS / src).resolve()
        return f"![{alt}]({resolved})"

    text = IMAGE.sub(img, text)
    def link(m: re.Match) -> str:
        target = (path.parent / (m.group(2) + ".md")).resolve()
        try:
            rel = target.relative_to(DOCS)
        except ValueError:
            return m.group(1)
        return f"[{m.group(1)}](#{chapter_id(DOCS / rel)})"

    text = MD_LINK.sub(link, text)
    text = ATTR_LIST.sub("", text)
    text = re.sub(r"\]\(#\d+-", "](#", text)
    text = re.sub(r"^\s*```mermaid\s*$", "```\n(diagram, see the web edition)", text, flags=re.M)
    return expand_math_macros(text)


def chapter_id(path: Path) -> str:
    """A unique anchor. Every part has its own index.md, so the stem alone collides."""
    parent = path.parent.name
    if parent == "docs":
        return f"book-{path.stem.lower()}"
    return f"{parent}-{path.stem}".lower()


def assemble(only: str | None) -> str:
    parts: list[str] = []
    seen_part = ""
    for label, path in read_nav():
        if only and only not in str(path):
            continue
        if label and label != seen_part:
            parts.append(f"\n\n# {label}\n")
            seen_part = label
        text = path.read_text(encoding="utf-8")
        text = expand_snippets(text)
        text = convert_blocks(text)
        text = rewrite_links(text, path)
        # every chapter heading drops one level so the part heading owns the level above
        text = re.sub(r"^(#{1,5})\s", r"#\1 ", text, flags=re.M)
        anchor = chapter_id(path)
        first = re.search(r"^##[ \t]+(.+?)[ \t]*$", text, flags=re.M)
        if first:
            text = (text[: first.start()]
                    + f"## {first.group(1)} {{#{anchor}}}"
                    + text[first.end():])
        else:
            text = f"\n## {path.stem} {{#{anchor}}}\n\n" + text
        parts.append(text)
    return "\n\n".join(parts)


def run(cmd: list[str]) -> None:
    print("  " + " ".join(cmd[:6]) + " ...")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epub", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--part", default=None, help="limit to one part, e.g. part05")
    args = ap.parse_args()
    if not (args.epub or args.pdf or args.html or args.all):
        args.all = True

    BUILD.mkdir(exist_ok=True)
    print("Assembling chapters in nav order ...")
    body = assemble(args.part)
    stem = f"ML-Interview-Book{'-' + args.part if args.part else ''}"
    src = BUILD / f"{stem}.md"
    meta = (
        "---\n"
        f'title: "{TITLE}"\n'
        f'subtitle: "{SUBTITLE}"\n'
        "lang: en\n"
        "toc-title: Contents\n"
        "---\n\n"
    )
    src.write_text(meta + body, encoding="utf-8")
    words = len(body.split())
    print(f"  {src.relative_to(ROOT)}: {words:,} words")

    common = [
        "pandoc", str(src),
        "--from", "markdown+tex_math_dollars+pipe_tables+fenced_code_blocks+backtick_code_blocks+raw_html",
        "--toc", "--toc-depth=2",
        "--standalone",
        "--resource-path", f"{ROOT}:{DOCS}",
        "--highlight-style", "tango",
    ]

    if args.epub or args.all:
        print("Building EPUB ...")
        run(common + ["--mathml", "-o", str(BUILD / f"{stem}.epub")])

    if args.html or args.all:
        print("Building single-file HTML ...")
        # MathML rather than MathJax: no JavaScript, no network, and it is what
        # makes the file genuinely self-contained once images are embedded.
        run(common + ["--mathml", "--embed-resources", "-o", str(BUILD / f"{stem}.html")])

    if args.pdf or args.all:
        print("Building PDF via weasyprint ...")
        try:
            run(common + ["--mathml", "--pdf-engine", "weasyprint",
                          "-V", "papersize=a4", "-V", "margin-top=18mm",
                          "-V", "margin-bottom=18mm", "-V", "margin-left=20mm",
                          "-V", "margin-right=20mm",
                          "-o", str(BUILD / f"{stem}.pdf")])
        except subprocess.CalledProcessError:
            print("  PDF failed. The EPUB and HTML are still usable; open the HTML")
            print("  in a browser and print to PDF for the same result.")

    for f in sorted(BUILD.glob(f"{stem}.*")):
        print(f"  {f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
