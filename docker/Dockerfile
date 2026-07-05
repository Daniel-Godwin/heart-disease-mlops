FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Default: FastAPI service (override in docker-compose for the dashboard)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
