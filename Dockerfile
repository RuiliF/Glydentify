# Use Python 3.10 slim image as the base
FROM python:3.10-slim

# Install system dependencies
# Adding git and git-lfs as they may be required, plus common utilities
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.9 as specified in the README
# We do this step before requirements.txt to cache the heavy PyTorch download
RUN pip install --no-cache-dir \
    torch==2.8.0+cu129 \
    torchvision==0.23.0+cu129 \
    torchaudio==2.8.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# Copy requirements and install remaining python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
# Note: Ensure you download foldseek and place it in bin/ locally before building 
# so it gets copied into the container smoothly.
COPY . .

# Add the local bin directory to the PATH so foldseek can be found globally
ENV PATH="/app/bin:${PATH}"

# Add src to the PYTHONPATH so modules can be successfully imported
ENV PYTHONPATH="/app/src"

# Set a default command to test if inference script shows help
CMD ["python", "scripts/inference.py", "--help"]
