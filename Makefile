.PHONY: test device clean

test:
	pytest -q

device:
	python -c "from src.utils.device import get_device; print(get_device())"

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache
