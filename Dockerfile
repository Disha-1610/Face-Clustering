# Use official Python image
FROM python:3.10-slim

# Install system dependencies for dlib + OpenCV
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Railway uses PORT dynamically, not fixed 8000
EXPOSE 8000

# Start Flask app using Gunicorn
//CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--timeout", "300", "app:app"]

