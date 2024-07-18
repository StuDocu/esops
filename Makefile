.PHONY: check-autoformat autoformat lint

check-autoformat:  # Fails if autoformatting is needed, used for CI
	poetry run black --version
	poetry run black --check .

autoformat:
	poetry run black --version
	poetry run black .

lint:
	poetry run flake8 --version
	poetry run flake8 . --max-line-length=120 --ignore=E501,E203,W503
