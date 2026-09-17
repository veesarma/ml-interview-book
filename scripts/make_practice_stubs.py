"""Generate signature-only practice stubs from the reference implementations.

Usage:
    python scripts/make_practice_stubs.py                 # regenerate every stub
    python scripts/make_practice_stubs.py transformer/multihead   # one module

For each ``src/mlbook/<area>/<module>.py`` this writes ``practice/<area>/<module>.py``
keeping imports, module-level constants, class and function signatures and
docstrings, and replacing every function body with ``raise NotImplementedError``.

Existing practice files are only overwritten when ``--force`` is given or when
they are still untouched stubs (they carry a marker line at the top).
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mlbook"
PRACTICE = ROOT / "practice"
MARKER = "# mlbook-practice-stub: untouched"


class _Stubber(ast.NodeTransformer):
    """Replace function bodies with a docstring (if any) + NotImplementedError."""

    def _stub_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
            getattr(node.body[0], "value", None), ast.Constant
        ) and isinstance(node.body[0].value.value, str):
            body.append(node.body[0])
        msg = f"TODO: implement {node.name} (see the reference in src/mlbook)"
        body.append(
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                    args=[ast.Constant(value=msg)],
                    keywords=[],
                ),
                cause=None,
            )
        )
        return body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._stub_body(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._stub_body(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        return node


def _keep_top_level(node: ast.stmt) -> bool:
    """Keep imports, constants, dataclass-style assignments, classes and functions."""
    return isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Assign,
            ast.AnnAssign,
            ast.Expr,  # module docstring
        ),
    )


def stub_source(source: str, rel: str) -> str:
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if _keep_top_level(n)]
    tree = _Stubber().visit(tree)
    ast.fix_missing_locations(tree)
    header = (
        f"{MARKER}\n"
        f"# Practice stub for src/mlbook/{rel}\n"
        "# Fill in every `raise NotImplementedError`, then run:\n"
        f"#     MLBOOK_IMPL=practice pytest tests/ -k {Path(rel).stem} -q\n"
        "# Regenerate a clean stub with:\n"
        f"#     python scripts/make_practice_stubs.py {rel[:-3]} --force\n\n"
    )
    return header + ast.unparse(tree) + "\n"


def write_stub(src_file: Path, force: bool) -> str:
    rel = src_file.relative_to(SRC).as_posix()
    out = PRACTICE / rel
    if out.exists() and not force:
        first = out.read_text().splitlines()[:1]
        if not first or first[0] != MARKER:
            return f"skip (edited)  {rel}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stub_source(src_file.read_text(), rel))
    return f"wrote          {rel}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("modules", nargs="*", help="e.g. transformer/multihead")
    parser.add_argument("--force", action="store_true", help="overwrite edited stubs")
    args = parser.parse_args()

    if args.modules:
        files = [SRC / (m if m.endswith(".py") else m + ".py") for m in args.modules]
    else:
        files = sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")

    for f in files:
        if not f.exists():
            print(f"missing        {f.relative_to(ROOT)}", file=sys.stderr)
            continue
        print(write_stub(f, args.force))

    # __init__.py files are copied verbatim so `practice.<area>` mirrors `mlbook.<area>`.
    for init in SRC.rglob("__init__.py"):
        rel = init.relative_to(SRC)
        out = PRACTICE / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or out.read_text() != init.read_text():
            out.write_text(init.read_text())


if __name__ == "__main__":
    main()
