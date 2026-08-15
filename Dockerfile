FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential make curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN make requirements

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "financial_crime.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
