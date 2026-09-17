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
