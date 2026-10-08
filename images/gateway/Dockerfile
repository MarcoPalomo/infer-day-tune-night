FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[gateway]"
USER 65532
EXPOSE 8080
CMD ["uvicorn", "idtn.gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
