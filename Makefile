.PHONY: install data train test lint api dashboard drift docker clean

install:
	pip install -r requirements-dev.txt

data:
	python -m claimsense.data.make_dataset --rows 20000

train:
	python -m claimsense.models.train

test:
	pytest -q

lint:
	ruff check src tests app

api:
	uvicorn claimsense.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run app/streamlit_app.py

drift:
	python -m claimsense.monitoring

docker:
	docker compose up --build

clean:
	rm -rf models/*.joblib data/processed/* reports/metrics.json .pytest_cache
