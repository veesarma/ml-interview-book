"""Generate docs/contents.md: the full table of contents, straight from the mkdocs nav.

Run it after changing the nav so the contents page can never drift from the real
structure:

    python scripts/build_contents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mkdocs.yml"
OUT = ROOT / "docs" / "contents.md"

HEADER = """# Contents

Every part and every chapter, on one page. The same tree lives in the left sidebar,
where each part expands in place.

"""


class _IgnoreUnknownTags(yaml.SafeLoader):
    """mkdocs.yml carries `!!python/name:` tags that SafeLoader refuses; skip them."""


def _ignore(loader, suffix, node):  # noqa: ANN001
    return None


_IgnoreUnknownTags.add_multi_constructor("tag:yaml.org,2002:python/name:", _ignore)
_IgnoreUnknownTags.add_multi_constructor("!", _ignore)


def _entries(nav: list) -> list[tuple[str | None, str | list]]:
    """Normalise nav entries to (title, target) pairs; title is None for a bare path."""
    out = []
    for item in nav:
        if isinstance(item, str):
            out.append((None, item))
        elif isinstance(item, dict):
            for title, target in item.items():
                out.append((str(title), target))
    return out


def _render(nav: list) -> str:
    lines: list[str] = []
    for title, target in _entries(nav):
        if isinstance(target, list):                       # a part, with chapters under it
            children = _entries(target)
            index = next((t for n, t in children if isinstance(t, str)
                          and t.endswith("index.md")), None)
            heading = f"## [{title}]({index})" if index else f"## {title}"
            lines.append(heading)
            lines.append("")
            for child_title, child in children:
                if not isinstance(child, str) or child == index:
                    continue
                label = child_title or Path(child).stem.replace("-", " ")
                lines.append(f"- [{label}]({child})")
            lines.append("")
        elif isinstance(target, str) and title:
            lines.append(f"## [{title}]({target})")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    config = yaml.load(CONFIG.read_text(encoding="utf-8"), Loader=_IgnoreUnknownTags)
    nav = config.get("nav")
    if not nav:
        print("no nav in mkdocs.yml", file=sys.stderr)
        return 1
    OUT.write_text(HEADER + _render(nav), encoding="utf-8")
    n_links = OUT.read_text(encoding="utf-8").count("](")
    print(f"wrote {OUT.relative_to(ROOT)} with {n_links} links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
