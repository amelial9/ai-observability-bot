# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install curl for healthcheck (python:3.11-slim doesn't include it by default)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . /app

# The port your FastAPI app runs on
EXPOSE 8001

# Command to run the application using uvicorn.
# FINAL FIX: We explicitly set PYTHONPATH to include the /app/backend directory.
# This forces the Python interpreter to look there for the 'agent' module.
CMD ["sh", "-c", "PYTHONPATH=$PYTHONPATH:/app/backend uvicorn backend.main:app --host 0.0.0.0 --port 8001"]