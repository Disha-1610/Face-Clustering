# Use official Python image
FROM python:3.10-slim

# Install system dependencies needed for dlib + OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose port Railway uses
EXPOSE 8000

# Start Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
