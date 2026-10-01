.PHONY: install lint test train clean

PYTHON := python3
PIP := pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest tests/ -v

train:
	$(PYTHON) src/train.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf mlruns .coverage