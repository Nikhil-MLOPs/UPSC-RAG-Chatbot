FROM python:3.12-slim

# Prevent Python from writing .pyc files and ensure output is flushed
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies (git for HF models; you can add more if needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Gradio port
EXPOSE 7860

# Default command: run the Gradio app
CMD ["python", "app.py"]
