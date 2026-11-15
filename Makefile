.PHONY: venv install lint format test docker-build clean

VENV = .venv
PYTHON = $(VENV)/bin/python
UV = uv

venv:
	$(UV) venv $(VENV)

install: venv
	$(UV) pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m black --check src/ tests/

format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

test:
	$(PYTHON) -m pytest tests/

docker-build:
	docker build -t credex:latest .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
