test:
	pytest

unit:
	pytest tests/unit

integration:
	pytest tests/integration

coverage-html:
	pytest --cov=CopulaFurtif --cov-report=html && xdg-open htmlcov/index.html || open htmlcov/index.html

clean:
	rm -rf .pytest_cache .coverage htmlcov
