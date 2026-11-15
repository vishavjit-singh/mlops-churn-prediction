PY ?= python3
export PYTHONPATH := src

.PHONY: install install-mlops demo pipeline generate prepare train evaluate monitor report summary repro test lint serve docker clean

install:
	$(PY) -m pip install -r requirements-dev.txt

install-mlops:
	$(PY) -m pip install -r requirements-mlops.txt && $(PY) -m pip install -e . --no-deps

## The DVC stages in order (same commands as dvc.yaml), then the README charts and a summary
demo: clean pipeline report summary

pipeline: generate prepare train evaluate monitor

generate prepare train evaluate monitor:
	$(PY) -m churn $@

report:
	$(PY) -m churn report

summary:
	$(PY) -m churn summary

## With DVC installed: run only the stages whose deps or params changed, then show metrics
repro:
	dvc repro && dvc metrics show

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve:
	$(PY) -m churn serve

docker:
	docker compose up --build

clean:
	rm -rf data models metrics plots reports mlruns_local
