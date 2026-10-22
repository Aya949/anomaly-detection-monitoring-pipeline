# Base image — slim Python 3.12
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY api/        ./api/
COPY models/     ./models/

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]