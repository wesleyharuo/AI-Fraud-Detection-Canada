FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default: train model (so API has artifacts); override CMD for prod
RUN python src/train.py --config config.yaml || true

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
