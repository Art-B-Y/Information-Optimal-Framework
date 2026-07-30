.PHONY: install test lint run train-score sample-final

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests
	
run:
	python -m scripts.run_experiment

train-score:
	python -m scripts.train_score_model

sample-final:
	python -m scripts.run_final_sampler
