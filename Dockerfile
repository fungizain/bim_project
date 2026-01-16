FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

RUN pip install poetry

RUN poetry install --no-root --only main

COPY . /app

CMD ["poetry", "run", "bim-app-prod"]