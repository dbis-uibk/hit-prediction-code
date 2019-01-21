init:
	pipenv install

init-dev:
	pipenv install --dev

tests:
	pipenv run python tests.py

requirements:
	pipenv lock -r > requirements.txt
	pipenv lock -r -d > requirements-dev.txt

.PHONY: init init-dev tests requirements
