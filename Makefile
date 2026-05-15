.PHONY: help verify ingest train serve test clean

help:
	@echo "CerviRisk-MM commands:"
	@echo "  make verify   - check all data sources are reachable"
	@echo "  make ingest   - pull data from all four sources into data/raw/"
	@echo "  make train    - train the risk model (day 2+)"
	@echo "  make serve    - start the FastAPI prediction service (day 3+)"
	@echo "  make test     - run the test suite"
	@echo "  make clean    - remove generated data and models"

verify:
	python verify_data_sources.py

ingest:
	python -m src.ingestion

train:
	python -m src.model.train

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

clean:
	rm -rf data/raw/* data/processed/* models/* mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
