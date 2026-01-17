FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends pandoc \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]

