.PHONY: install test style fix-style serve build figures references book deploy

install:
	pip install -r requirements.txt && pip install -e .

test:
	python -m pytest -q

style:
	python scripts/check_style.py
	python scripts/check_markdown.py

fix-style:
	@for d in docs/part*; do python scripts/fix_style.py $$d; done
	python scripts/fix_indentation.py

serve:
	mkdocs serve

build:
	mkdocs build --strict

references:
	python scripts/build_references.py

figures:
	@for f in figures/*.py; do echo "→ $$f"; python $$f || exit 1; done

book:
	@command -v pandoc >/dev/null || { echo "pandoc not found: apt-get install pandoc"; exit 1; }
	python scripts/build_book.py --all

book-epub:
	python scripts/build_book.py --epub

book-pdf:
	python scripts/build_book.py --pdf

deploy:
	mkdocs gh-deploy --force

# ---- practice sandbox -------------------------------------------------------
.PHONY: stubs drill check reset
ITEM ?= transformer/multihead

stubs:
	python scripts/make_practice_stubs.py

drill:
	@echo "Edit:   practice/$(ITEM).py"
	@echo "Tests:  $$(ls tests/test_$$(echo $(ITEM) | tr '/' '_')*.py 2>/dev/null || echo 'tests/ -k $(notdir $(ITEM))')"
	@echo "Check:  make check ITEM=$(ITEM)"

check:
	MLBOOK_IMPL=practice python -m pytest tests -k "$(notdir $(ITEM))" -q

reset:
	python scripts/make_practice_stubs.py $(ITEM) --force
