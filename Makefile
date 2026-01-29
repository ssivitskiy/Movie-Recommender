.PHONY: install install-dev test lint format clean data train evaluate app help

help:
	@echo "Movie Recommender commands:"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make data         Download dataset"
	@echo "  make train        Train model"
	@echo "  make test         Run tests"

install-dev:
	pip install -e ".[dev,web,notebooks]"

data:
	python src/data_loader.py

train:
	python -m src.train --model svd --epochs 20 --evaluate

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf build dist *.egg-info