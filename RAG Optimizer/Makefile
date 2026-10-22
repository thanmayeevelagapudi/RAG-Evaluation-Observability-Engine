.PHONY: infra-up infra-down seed test lint run benchmark benchmark-ci

DATASET     ?= nfcorpus
MAX_QUERIES ?= 50
BASELINE_FILE ?= benchmarks/baselines/$(DATASET).json
QUALITY_THRESHOLD ?= 0.02
LATENCY_REGRESSION_PCT ?= 20
CI_STRATEGIES ?= dense bm25 hybrid parent_child

infra-up:
	docker compose up -d qdrant elasticsearch postgres redis

infra-down:
	docker compose down

seed:
	python scripts/seed_corpus.py

test:
	pytest tests/ -q --tb=short

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	python -m benchmarks.run_benchmark --dataset $(DATASET) --max-queries $(MAX_QUERIES)

benchmark-ci:
	python -m benchmarks.run_benchmark --dataset $(DATASET) --max-queries $(MAX_QUERIES) --ci --baseline-file $(BASELINE_FILE) --quality-threshold $(QUALITY_THRESHOLD) --latency-regression-pct $(LATENCY_REGRESSION_PCT) --strategies $(CI_STRATEGIES)

lint:
	ruff check . --fix
