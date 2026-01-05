install:
	pip install -r requirements.txt

test:
	pytest tests/

lint:
	flake8 src/

format:
	black src/

docker-build:
	docker build -t mlops-app:latest .

docker-run:
	docker run mlops-app:latest

ci-local:
	make lint
	make test
	make docker-build
