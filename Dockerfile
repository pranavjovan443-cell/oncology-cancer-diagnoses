FROM python:3.11-slim

WORKDIR /app

# Install essential C build libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Initialize database schema and demonstration dataset
RUN python -c "from app import app; from database.init_db import init_db; init_db(app)"

EXPOSE 5000

ENV PORT=5000
ENV FLASK_ENV=production

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
