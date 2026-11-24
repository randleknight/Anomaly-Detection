# Use a suitable base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Featureform and required dependencies
RUN pip install featureform googleapis-common-protos

# Copy project files
COPY . .

# Command to run Featureform
CMD ["featureform", "deploy"]
