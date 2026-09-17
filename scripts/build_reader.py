"""Build the hosted reader: one page per part inside a real documentation shell.

Pandoc's standalone HTML is a bare document. This renders each part as a body
fragment and wraps it in a shell with a persistent chapter sidebar, type-to-filter
navigation, and a theme that follows the viewer.

    python scripts/build_reader.py      # writes build/reader/
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build" / "ML-Interview-Book.md"
OUT = ROOT / "build" / "reader"
FIGS = ROOT / "docs" / "assets" / "figures"
MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"

CSS = """
:root{
  --ink:#16181d; --ink-2:#3d434f; --ink-3:#6b7280;
  --ground:#fbfaf8; --surface:#ffffff; --rail:#f3f1ed;
  --line:#e2ded6; --line-2:#cfcabf;
  --accent:#8a5a1f; --accent-soft:#f0e6d6;
  --code-bg:#f5f3ef;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ink:#e8e6e1; --ink-2:#b3b0a8; --ink-3:#85827b;
    --ground:#14151a; --surface:#191b21; --rail:#101116;
    --line:#2a2d36; --line-2:#3b3f4a;
    --accent:#d9a45b; --accent-soft:#2b2318;
    --code-bg:#1d1f26;
  }
}
:root[data-theme="dark"]{
  --ink:#e8e6e1; --ink-2:#b3b0a8; --ink-3:#85827b;
  --ground:#14151a; --surface:#191b21; --rail:#101116;
  --line:#2a2d36; --line-2:#3b3f4a;
  --accent:#d9a45b; --accent-soft:#2b2318;
  --code-bg:#1d1f26;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Source Serif 4",Georgia,"Times New Roman",serif;
  font-size:1.0625rem; line-height:1.66;
  -webkit-font-smoothing:antialiased;
}
.shell{display:grid; grid-template-columns:19rem minmax(0,1fr); min-height:100%}
/* ---- sidebar ---- */
.rail{
  background:var(--rail); border-right:1px solid var(--line);
  position:sticky; top:env(safe-area-inset-top,0px);
  height:100vh; overflow-y:auto; overscroll-behavior:contain;
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:.8125rem;
}
.rail-inner{padding:1.25rem 1rem 3rem}
.brand{display:block; text-decoration:none; color:var(--ink); margin-bottom:.25rem}
.brand b{display:block; font-size:.9375rem; font-weight:600; letter-spacing:-.01em}
.brand span{display:block; color:var(--ink-3); font-size:.75rem; margin-top:.15rem}
.filter{
  width:100%; margin:.9rem 0 .4rem; padding:.45rem .6rem;
  font:inherit; color:var(--ink); background:var(--surface);
  border:1px solid var(--line-2); border-radius:5px;
}
.filter:focus{outline:2px solid var(--accent); outline-offset:1px}
.count{color:var(--ink-3); font-size:.6875rem; margin:0 0 .8rem}
.rail nav ol{list-style:none; margin:0; padding:0}
.part{margin-bottom:.55rem}
.part > a{
  display:block; text-decoration:none; color:var(--ink-2);
  font-weight:600; font-size:.75rem; letter-spacing:.055em; text-transform:uppercase;
  padding:.3rem 0;
}
.part > a:hover{color:var(--accent)}
.part ol{border-left:1px solid var(--line); margin-left:.15rem}
.part li a{
  display:block; text-decoration:none; color:var(--ink-2);
  padding:.22rem .55rem; margin-left:-1px; border-left:2px solid transparent;
  line-height:1.35;
}
.part li a:hover{color:var(--ink); background:var(--surface)}
.part li a[aria-current="page"]{color:var(--accent); border-left-color:var(--accent); font-weight:600}
/* ---- main ---- */
main{min-width:0}
.bar{
  display:none; position:sticky; top:0; z-index:20;
  padding:.55rem 1rem; padding-top:calc(.55rem + env(safe-area-inset-top,0px));
  background:var(--ground); border-bottom:1px solid var(--line);
  font-family:"IBM Plex Sans",system-ui,sans-serif;
}
.bar button{
  font:inherit; font-size:.8125rem; padding:.35rem .7rem; cursor:pointer;
  color:var(--ink); background:var(--surface);
  border:1px solid var(--line-2); border-radius:5px;
}
/* Prose stays at a readable measure; code, tables and figures break out of it,
   because a line like `x = x.view(B, T, H, d_head).transpose(1, 2)  # (B,H,T,d)`
   does not fit in 44rem and scrolling every block is miserable. */
article{
  display:grid;
  grid-template-columns:1fr min(44rem,100%) 1fr;
  padding:3rem 2rem 6rem;
  column-gap:0;
}
article > *{grid-column:2; min-width:0}
article > pre,
article > .table-wrap,
article > img,
article > .landing-parts{
  grid-column:1/-1;
  width:min(72rem,100%);
  justify-self:start;
}
article > *:first-child{margin-top:0}
h1,h2,h3,h4{font-family:"IBM Plex Sans",system-ui,sans-serif; line-height:1.22; text-wrap:balance}
h1{font-size:2.1rem; font-weight:600; letter-spacing:-.022em; margin:0 0 2.2rem}
h2{
  font-size:1.5rem; font-weight:600; letter-spacing:-.016em;
  margin:3.6rem 0 1rem; padding-top:1.6rem; border-top:1px solid var(--line);
}
h3{font-size:1.125rem; font-weight:600; margin:2.1rem 0 .7rem}
h4{font-size:.9375rem; font-weight:600; margin:1.5rem 0 .5rem; color:var(--ink-2)}
p{margin:0 0 1.05rem}
a{color:var(--accent); text-underline-offset:2px}
strong{font-weight:600}
ul,ol{margin:0 0 1.05rem; padding-left:1.35rem}
li{margin:.3rem 0}
blockquote{
  margin:1.4rem 0; padding:.85rem 1.1rem;
  background:var(--surface); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:0 5px 5px 0;
  font-size:.96em;
}
blockquote > *:last-child{margin-bottom:0}
code{
  font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; font-size:.855em;
  background:var(--code-bg); padding:.12em .34em; border-radius:3px;
}
pre{
  background:var(--code-bg); border:1px solid var(--line);
  border-radius:6px; padding:.85rem 1rem; margin:1.3rem 0;
  overflow-x:auto; overscroll-behavior-x:contain;
  font-size:.78rem; line-height:1.5; tab-size:4;
}
pre::-webkit-scrollbar{height:9px}
pre::-webkit-scrollbar-thumb{background:var(--line-2); border-radius:5px}
pre code{background:none; padding:0; font-size:1em}
.table-wrap{overflow-x:auto; margin:1.4rem 0}
table{border-collapse:collapse; width:100%; font-size:.875rem;
  font-family:"IBM Plex Sans",system-ui,sans-serif;
  font-variant-numeric:tabular-nums}
th,td{text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line); vertical-align:top}
th{font-weight:600; font-size:.78rem; letter-spacing:.03em; text-transform:uppercase; color:var(--ink-2)}
tbody tr:hover{background:var(--surface)}
img{max-width:100%; height:auto; display:block; margin:1.5rem 0; border:1px solid var(--line); border-radius:5px}
hr{border:0; border-top:1px solid var(--line); margin:2.4rem 0}
mjx-container{overflow-x:auto; overflow-y:hidden; max-width:100%}
#TOC{
  background:var(--surface); border:1px solid var(--line); border-radius:6px;
  padding:1rem 1.2rem; margin:0 0 2.5rem;
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:.875rem;
}
#TOC > ul{margin:0; padding-left:1.1rem}
#TOC li{margin:.2rem 0}
.landing-parts{display:grid; gap:.9rem; grid-template-columns:repeat(auto-fill,minmax(15rem,1fr)); margin:2rem 0}
.landing-parts a{
  display:block; text-decoration:none; padding:.9rem 1rem;
  background:var(--surface); border:1px solid var(--line); border-radius:6px;
  font-family:"IBM Plex Sans",system-ui,sans-serif;
}
.landing-parts a:hover{border-color:var(--accent)}
.landing-parts b{display:block; color:var(--ink); font-size:.9375rem; font-weight:600}
.landing-parts span{display:block; color:var(--ink-3); font-size:.78rem; margin-top:.2rem}
@media (max-width:900px){
  .shell{grid-template-columns:1fr}
  .rail{position:fixed; inset:0 auto 0 0; width:min(21rem,86vw); z-index:30;
    transform:translateX(-100%); transition:transform .18s ease}
  .rail.open{transform:none}
  .bar{display:flex; gap:.6rem; align-items:center}
  article{padding:1.4rem 1rem 4rem; grid-template-columns:1fr}
  article > *, article > pre, article > .table-wrap, article > img{grid-column:1; width:100%}
  h1{font-size:1.7rem}
  h2{font-size:1.3rem}
}
@media (prefers-reduced-motion:reduce){.rail{transition:none}}
.scrim{display:none}
@media (max-width:900px){
  .scrim.on{display:block; position:fixed; inset:0; background:#0008; z-index:25}
}
"""

JS = """
(function(){
  var rail=document.getElementById('rail'), scrim=document.getElementById('scrim'),
      btn=document.getElementById('navbtn'), f=document.getElementById('filter');
  function close(){rail.classList.remove('open');scrim.classList.remove('on');}
  if(btn){btn.addEventListener('click',function(){
    rail.classList.toggle('open');scrim.classList.toggle('on');});}
  if(scrim){scrim.addEventListener('click',close);}
  rail.addEventListener('click',function(e){if(e.target.tagName==='A')close();});
  if(f){
    f.addEventListener('input',function(){
      var q=f.value.trim().toLowerCase();
      document.querySelectorAll('.part').forEach(function(p){
        var any=false;
        p.querySelectorAll('li').forEach(function(li){
          var hit=!q||li.textContent.toLowerCase().indexOf(q)>-1;
          li.hidden=!hit; if(hit)any=true;
        });
        var head=p.querySelector(':scope > a');
        var headHit=!q||head.textContent.toLowerCase().indexOf(q)>-1;
        p.hidden=!(any||headHit);
        if(headHit&&q)p.querySelectorAll('li').forEach(function(li){li.hidden=false;});
      });
    });
    document.addEventListener('keydown',function(e){
      if(e.key==='/'&&document.activeElement!==f){e.preventDefault();
        rail.classList.add('open');scrim.classList.add('on');f.focus();}
    });
  }
  var cur=rail.querySelector('a[aria-current="page"]');
  if(cur&&cur.scrollIntoView)cur.scrollIntoView({block:'center'});
})();
"""


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") or "part"


def main() -> int:
    if not SRC.exists():
        print("run scripts/build_book.py first", file=sys.stderr)
        return 1
    body = SRC.read_text(encoding="utf-8").split("---\n", 2)[-1]

    pieces = re.split(r"^# (.+)$", body, flags=re.M)
    lead = pieces[0]
    parts = [(pieces[i].strip(), pieces[i + 1]) for i in range(1, len(pieces), 2)]
    if lead.strip():
        parts.insert(0, ("Front matter", lead))
    files = [(slug(t) + ".html", t, c) for t, c in parts]

    anchor_of: dict[str, str] = {}
    chapters: dict[str, list[tuple[str, str]]] = {}
    for fname, _t, content in files:
        chapters[fname] = re.findall(r"^## (.+?) \{#([a-z0-9-]+)\}", content, flags=re.M)
        for a in re.findall(r"\{#([a-z0-9-]+)\}", content):
            anchor_of[a] = fname

    total_chapters = sum(len(v) for v in chapters.values())

    def sidebar(active_file: str) -> str:
        out = ['<nav aria-label="Book contents"><ol>']
        for fname, title, _c in files:
            out.append('<li class="part">')
            out.append(f'<a href="{fname}">{html.escape(title)}</a><ol>')
            for ch, anchor in chapters[fname]:
                cur = ' aria-current="page"' if fname == active_file else ""
                out.append(f'<li><a href="{fname}#{anchor}"{cur}>{html.escape(ch)}</a></li>')
            out.append("</ol></li>")
        out.append("</ol></nav>")
        return "".join(out)

    def shell(title: str, inner: str, active: str) -> str:
        return f"""<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style>
<script src="{MATHJAX}"></script>
<div class="shell">
  <aside class="rail" id="rail"><div class="rail-inner">
    <a class="brand" href="index.html"><b>The ML Interview Book</b>
      <span>{total_chapters} chapters, {len(files)} parts</span></a>
    <input class="filter" id="filter" type="search" placeholder="Filter chapters &nbsp; /" aria-label="Filter chapters">
    <p class="count">Type to filter. Press / from anywhere.</p>
    {sidebar(active)}
  </div></aside>
  <main>
    <div class="bar"><button id="navbtn" type="button">Contents</button></div>
    <article>{inner}</article>
  </main>
</div>
<div class="scrim" id="scrim"></div>
<script>{JS}</script>
"""

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.html"):
        old.unlink()

    def relink(content: str, fname: str) -> str:
        def r(m: re.Match) -> str:
            a = m.group(1)
            tgt = anchor_of.get(a)
            return f"](#{a})" if tgt in (None, fname) else f"]({tgt}#{a})"
        return re.sub(r"\]\(#([a-z0-9-]+)\)", r, content)

    tmp = OUT / "_t.md"
    for fname, title, content in files:
        tmp.write_text(relink(content, fname), encoding="utf-8")
        frag = subprocess.run(
            ["pandoc", str(tmp), "--from",
             "markdown+tex_math_dollars+pipe_tables+fenced_code_blocks+backtick_code_blocks+raw_html",
             "--to", "html", "--standalone",
             "--template", str(ROOT / "scripts" / "frag.template"),
             f"--mathjax={MATHJAX}", "--highlight-style", "tango"],
            check=True, capture_output=True, text=True, cwd=ROOT).stdout
        frag = re.sub(r'src="(?:[^"]*/)?([^"/]+\.(?:png|jpg|jpeg|gif|svg))"', r'src="figures/\1"', frag)
        frag = re.sub(r"(<table[\s\S]*?</table>)", r'<div class="table-wrap">\1</div>', frag)
        page = shell(f"{title} | The ML Interview Book", f"<h1>{html.escape(title)}</h1>\n" + frag, fname)
        (OUT / fname).write_text(page, encoding="utf-8")
        print(f"  {fname}  {(OUT/fname).stat().st_size/1e6:.2f} MB")

    cards = "".join(
        f'<a href="{f}"><b>{html.escape(t)}</b>'
        f'<span>{len(chapters[f])} chapter{"s" if len(chapters[f]) != 1 else ""}</span></a>'
        for f, t, _ in files)
    landing = f"""<h1>The ML Interview Book</h1>
<p>From first principles to the frontier, for staff-level ML interviews.
{total_chapters} chapters across {len(files)} parts: the mathematics, from-scratch
implementations with every tensor shape written down, the systems trade-offs, ML
system design, and what companies actually published about how they built it.</p>
<p>Use the sidebar to jump to any chapter. Press <code>/</code> to filter by name.</p>
<div class="landing-parts">{cards}</div>"""
    (OUT / "index.html").write_text(shell("The ML Interview Book", landing, ""), encoding="utf-8")
    tmp.unlink()

    figs = OUT / "figures"
    figs.mkdir(exist_ok=True)
    for p in FIGS.glob("*.png"):
        shutil.copy(p, figs / p.name)

    big = max((f.stat().st_size, f.name) for f in OUT.glob("*.html"))
    print(f"\n{len(files)+1} pages, {len(list(figs.iterdir()))} figures")
    print(f"largest page {big[0]/1e6:.2f} MB ({big[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
