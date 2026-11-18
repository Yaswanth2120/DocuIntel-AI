PYTHON ?= .venv/bin/python
export PYTHONPYCACHEPREFIX ?= $(HOME)/.cache/docuintel-ai/pycache

.PHONY: install test test-shared lint run eval eval-public gates up down
install:
	uv sync --frozen --extra dev --python 3.12

test:
	$(PYTHON) -m unittest discover -s tests -t .

test-shared:
	$(PYTHON) -m unittest tests.test_shared_storage

lint:
	uv run ruff check app evals scripts tests
	uv run ruff format --check app evals scripts tests

run:
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

eval:
	$(PYTHON) -m evals.smoke

eval-public:
	$(PYTHON) scripts/run_evaluation.py --dataset scifact --queries 100
	$(PYTHON) scripts/run_evaluation.py --dataset squad --queries 100 --generate 30

gates:
	$(PYTHON) -m evals.check_reports

up:
	docker compose up --build -d

down:
	docker compose down

eval-production:
	$(PYTHON) scripts/run_evaluation.py --dataset squad --queries 100 --generate 30 --generation-backend configured
