"""Build the consolidated bibliography from every chapter's References section.

    python scripts/build_references.py

Walks docs/part*/*.md, reads each file's trailing `## References` section, and
emits docs/_references_body.md: one entry per distinct source, grouped by kind,
each listing the chapters that cite it. Entries are de-duplicated by URL where
one exists and by normalised title otherwise, so the same paper cited in six
chapters appears once with six back-links.

Run it after the chapters are written and after the citation passes, since it
reflects whatever the chapters currently say.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "snippets" / "references_body.md"

REF_HEADING = re.compile(r"^##\s+References\s*$", re.I)
NEXT_HEADING = re.compile(r"^##\s+")
LIST_ITEM = re.compile(r"^\s*[-*]\s+(.*)$")
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
# An entry copied out of a chapter may carry a relative cross-link such as
# ](../part12-rl/04-...md). This page lives at the docs root, one level up, so
# that "../" would escape the docs tree. Rewrite it to a root-relative path.
REL_LINK = re.compile(r"\]\(\.\./(part\d\d[^)]*\.md)\)")
# A sibling link such as ](06-dimensionality-reduction.md) is relative to the
# chapter it came from, so it needs that chapter's directory prepended.
SIBLING_LINK = re.compile(r"\]\((?!https?://|\.\./|/|part\d\d)([0-9a-z][0-9a-z-]*\.md)\)")
TITLE_QUOTED = re.compile(r'["“]([^"”]{6,200})["”]|\*([^*]{6,200})\*')

KINDS = [
    ("arXiv preprints and papers", ("arxiv.org",)),
    ("Company engineering and research blogs", (
        "engineering.fb.com", "ai.meta.com", "netflixtechblog", "eng.uber.com",
        "uber.com", "doordash", "medium.engineering", "airbnb.io", "pinterest",
        "blog.google", "research.google", "developers.googleblog", "openai.com",
        "anthropic.com", "deepmind", "nvidia.com", "developer.nvidia.com",
        "apple.com", "aws.amazon.com", "amazon.science", "stripe.com",
        "lyft.com", "linkedin.com", "spotify", "waymo.com", "zoox.com",
        "tesla.com", "aurora.tech", "nuro.ai", "bytedance", "huggingface.co",
        "vllm.ai", "databricks", "anyscale", "transformer-circuits.pub",
        "netflix", "engineering.atspotify", "shopify", "ebay", "snap.com",
        "microsoft.com", "cloudflare", "grab.com", "instacart", "wayve.ai",
        "mistral.ai", "cohere.com", "scale.com", "roblox", "discord.com",
    )),
    ("Documentation and standards", (
        "docs.", "pytorch.org", "iso.org", "ecfr.gov", "mutcd", "nist.gov",
        "readthedocs", "github.io", "triton-lang",
    )),
    ("Code repositories", ("github.com",)),
    ("Other primary sources", ()),
]


def classify(url: str) -> str:
    low = url.lower()
    for name, domains in KINDS[:-1]:
        if any(d in low for d in domains):
            return name
    return KINDS[-1][0]


def chapter_label(path: Path) -> str:
    part = path.parent.name
    return f"{part}/{path.stem}"


def extract(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if REF_HEADING.match(line):
            inside = True
            continue
        if inside and NEXT_HEADING.match(line):
            break
        if inside:
            m = LIST_ITEM.match(line)
            if m and m.group(1).strip():
                out.append(m.group(1).strip())
    return out


def norm_title(entry: str) -> str:
    m = TITLE_QUOTED.search(entry)
    raw = (m.group(1) or m.group(2)) if m else entry
    return re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()[:90]


def main() -> int:
    by_key: dict[str, dict] = {}
    for path in sorted(DOCS.rglob("*.md")):
        if path.name.startswith("_") or "citations-todo" in path.parts:
            continue
        for entry in extract(path):
            links = LINK.findall(entry)
            url = links[0][1] if links else ""
            key = url or ("title:" + norm_title(entry))
            if not key or key == "title:":
                continue
            entry = REL_LINK.sub(r"](\1)", entry)
            entry = SIBLING_LINK.sub(rf"]({path.parent.name}/\1)", entry)
            rec = by_key.setdefault(key, {"entry": entry, "url": url, "cited": set()})
            rec["cited"].add(chapter_label(path))
            # keep the richest phrasing of the entry
            if len(entry) > len(rec["entry"]):
                rec["entry"] = entry

    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in by_key.values():
        grouped[classify(rec["url"]) if rec["url"] else "Cited by title"].append(rec)

    order = [k for k, _ in KINDS] + ["Cited by title"]
    lines: list[str] = []
    total = 0
    for kind in order:
        recs = grouped.get(kind)
        if not recs:
            continue
        recs.sort(key=lambda r: norm_title(r["entry"]))
        lines.append(f"\n## {kind}\n")
        lines.append(f"{len(recs)} sources.\n")
        for rec in recs:
            cites = sorted(rec["cited"])
            shown = ", ".join(f"[{c.split('/')[0][:6]}]({c}.md)" for c in cites[:6])
            more = f" and {len(cites) - 6} more" if len(cites) > 6 else ""
            lines.append(f"* {rec['entry']}  <br/>*Cited in: {shown}{more}*")
            total += 1
        lines.append("")

    header = (
        f"This page is generated from the References section of every chapter by\n"
        f"`scripts/build_references.py`. It lists {total} distinct sources with the\n"
        f"chapters that cite each one. A source appears under \"Cited by title\" when\n"
        f"no URL for it could be verified, in which case search the exact title and venue.\n"
    )
    OUT.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}: {total} sources across {len(grouped)} groups.")
    for kind in order:
        if kind in grouped:
            print(f"  {len(grouped[kind]):>4}  {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
