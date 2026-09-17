.PHONY: install test serve build figures deploy

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -q

serve:
	mkdocs serve

build:
	mkdocs build --strict

figures:
	@for f in figures/*.py; do echo "→ $$f"; python $$f || exit 1; done

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
	MLBOOK_IMPL=practice pytest tests -k "$(notdir $(ITEM))" -q

reset:
	python scripts/make_practice_stubs.py $(ITEM) --force
