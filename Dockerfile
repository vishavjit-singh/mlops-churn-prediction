# Prediction API image. The model is trained at build time from the seeded synthetic data with the same
# stage commands DVC runs, so the image is self-contained and reproducible.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src CHURN_TRACKER=local
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml ./
COPY src ./src
RUN python -m churn generate && python -m churn prepare && python -m churn train && python -m churn evaluate

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
