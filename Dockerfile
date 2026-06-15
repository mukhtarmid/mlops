FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./
COPY uv.lock* ./

RUN uv pip install --system .

COPY src src
COPY models models

EXPOSE 8000

CMD [ "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]