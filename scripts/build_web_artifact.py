"""Split the assembled book into one small HTML page per part, for hosting.

A single 12 MB page is too heavy for a browser to lay out comfortably. This
renders one page per part (about 20 pages, each well under a megabyte), plus a
contents page, and rewrites every cross-part anchor to point at the file that
actually contains it.

    python scripts/build_web_artifact.py      # writes build/site-split/
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build" / "ML-Interview-Book.md"
OUT = ROOT / "build" / "site-split"
FIGS = ROOT / "docs" / "assets" / "figures"

MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"

NAV_CSS = """
<style>
  body { max-width: 46em; }
  .booknav { position: sticky; top: 0; background: #fdfdfd; border-bottom: 1px solid #ddd;
             padding: .6em 0; margin: -50px -50px 1.5em; padding-left: 50px; padding-right: 50px;
             font-size: .9em; z-index: 10; }
  .booknav a { margin-right: 1.2em; text-decoration: none; font-weight: 600; }
  .booknav a:hover { text-decoration: underline; }
  @media (max-width: 600px) { .booknav { margin: -12px -12px 1em; padding-left: 12px; padding-right: 12px; } }
</style>
"""


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s or "part"


def main() -> int:
    if not SRC.exists():
        print("run scripts/build_book.py first", file=sys.stderr)
        return 1
    text = SRC.read_text(encoding="utf-8")
    body = text.split("---\n", 2)[-1]

    # split on the part headings the assembler emitted as level-1
    pieces = re.split(r"^# (.+)$", body, flags=re.M)
    lead = pieces[0]
    parts: list[tuple[str, str]] = []
    for i in range(1, len(pieces), 2):
        parts.append((pieces[i].strip(), pieces[i + 1]))
    if lead.strip():
        parts.insert(0, ("Front matter", lead))

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.html"):
        old.unlink()

    files = [(slugify(t) + ".html", t, c) for t, c in parts]

    # map every anchor to the file that defines it, so cross-part links work
    anchor_file: dict[str, str] = {}
    for fname, _title, content in files:
        for a in re.findall(r"\{#([a-z0-9][a-z0-9-]*)\}", content):
            anchor_file[a] = fname

    def fix_links(content: str, fname: str) -> str:
        def repl(m: re.Match) -> str:
            anchor = m.group(1)
            target = anchor_file.get(anchor)
            if target is None or target == fname:
                return f"](#{anchor})"
            return f"]({target}#{anchor})"
        return re.sub(r"\]\(#([a-z0-9][a-z0-9-]*)\)", repl, content)

    nav_links = " ".join(f'<a href="{f}">{t}</a>' for f, t, _ in files)
    nav_html = f'<div class="booknav"><a href="index.html">Contents</a>{nav_links}</div>'

    for idx, (fname, title, content) in enumerate(files):
        md = f"# {title}\n" + fix_links(content, fname)
        tmp = OUT / "_tmp.md"
        tmp.write_text(md, encoding="utf-8")
        subprocess.run(
            ["pandoc", str(tmp), "--from",
             "markdown+tex_math_dollars+pipe_tables+fenced_code_blocks+backtick_code_blocks+raw_html",
             "--toc", "--toc-depth=2", "--standalone", f"--mathjax={MATHJAX}",
             "--highlight-style", "tango",
             "--metadata", f"title={title}",
             "-o", str(OUT / fname)],
            check=True, cwd=ROOT, stderr=subprocess.DEVNULL)
        h = (OUT / fname).read_text(encoding="utf-8")
        h = re.sub(r'src="(?:[^"]*/)?([^"/]+\.(?:png|jpg|jpeg|gif|svg))"', r'src="figures/\1"', h)
        h = h.replace("</head>", NAV_CSS + "</head>")
        h = h.replace("<body>", "<body>\n" + nav_html)
        (OUT / fname).write_text(h, encoding="utf-8")
        tmp.unlink()
        print(f"  {fname}  {(OUT/fname).stat().st_size/1e6:.2f} MB  {title}")

    # contents page
    toc = ["# The ML Interview Book", "",
           "*From first principles to the frontier, for staff-level ML interviews.*", "",
           "128 chapters. Pick a part.", ""]
    for fname, title, content in files:
        chapters = re.findall(r"^## (.+?) \{#", content, flags=re.M)
        toc.append(f"### [{title}]({fname})")
        toc.append("")
        for c in chapters:
            toc.append(f"* [{c}]({fname})")
        toc.append("")
    tmp = OUT / "_toc.md"
    tmp.write_text("\n".join(toc), encoding="utf-8")
    subprocess.run(["pandoc", str(tmp), "--from", "markdown", "--standalone",
                    "--metadata", "title=The ML Interview Book",
                    "-o", str(OUT / "index.html")],
                   check=True, cwd=ROOT, stderr=subprocess.DEVNULL)
    h = (OUT / "index.html").read_text(encoding="utf-8")
    h = h.replace("</head>", NAV_CSS + "</head>")
    (OUT / "index.html").write_text(h, encoding="utf-8")
    tmp.unlink()

    figs_out = OUT / "figures"
    figs_out.mkdir(exist_ok=True)
    for p in FIGS.glob("*.png"):
        shutil.copy(p, figs_out / p.name)

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n{len(files)+1} pages + {len(list(figs_out.iterdir()))} figures, {total/1e6:.1f} MB total")
    print(f"largest page: {max((f.stat().st_size/1e6, f.name) for f in OUT.glob('*.html'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
