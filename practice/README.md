# Practice sandbox

This folder is where you **retype the coding canon by hand**.

Every file here is a signature-only stub of the matching reference module in
`src/mlbook/`: same functions, same classes, same docstrings (with shapes),
bodies replaced by `raise NotImplementedError`.

## Workflow

```bash
make stubs                              # (re)generate stubs for every module
make drill ITEM=transformer/multihead   # opens the stub path + shows the tests to pass
#  ... edit practice/transformer/multihead.py from memory ...
make check ITEM=transformer/multihead   # runs the reference tests against YOUR code
make reset ITEM=transformer/multihead   # throw your attempt away, fresh stub
```

`make check` sets `MLBOOK_IMPL=practice`, which makes every `mlbook.<area>.<module>`
import resolve to `practice/<area>/<module>.py` when that file exists. The tests
are the same ones that guard the reference implementation, so passing them means
your version is behaviourally equivalent.

Practise one module at a time: modules you have not touched still resolve to the
reference implementation, so a test that imports helpers from elsewhere keeps working.

## One-click sandbox

Open the repository in GitHub Codespaces (or any devcontainer-aware editor):
`.devcontainer/devcontainer.json` installs the package and its tests. Locally,
`make install` does the same.

The list of what to retype, in which order, with time targets, is in
Part XVI of the book (`docs/part16-coding-canon/`), and every chapter has a
"Retype by hand" section naming its symbols and the pytest command that checks them.
