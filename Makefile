init:
	pipenv install

init-dev:
	pipenv install --dev

docs:
	pipenv run sphinx-build -b html docs/source/ docs/build/html/

tests:
	pipenv run python tests.py

requirements:
	pipenv lock -r | tail -n +2 > requirements.txt
	pipenv lock -r -d | tail -n +2 > requirements-dev.txt

docker-image: requirements
	docker build -t dbispipeline .

format:
	pipenv run yapf -i -r .

check-format:
	pipenv run flake8

bandit:
	pipenv run bandit -r .

.PHONY: init init-dev docs tests requirements docker-image format check-format bandit
