# Start with a suitable base image that supports linux/amd64
FROM python:3.9-slim-buster AS build_stage

# Specify the platform explicitly (good practice)
# FROM --platform=linux/amd64 python:3.9-slim-buster AS build_stage # Optional, as specified

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if any (e.g., for PDF parsing libraries)
# RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# Copy your requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# If you have an NLP model, ensure it's copied and accessible
# COPY app/models/your_model_name /app/models/your_model_name

# Final stage for a smaller image (optional, but good for size constraints)
FROM python:3.9-slim-buster
WORKDIR /app
COPY --from=build_stage /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=build_stage /app /app

# Define the entry point for your application
# This command will be executed when the Docker container starts
# It should process PDFs from /app/input and write to /app/output
CMD ["python", "app/main.py"]