FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# ffmpeg is crucial for audio processing in LiveKit agents
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy specific requirements file
COPY requirements-agent.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements-agent.txt

# Copy source code
COPY . .

# Run the agent
# Render detects this CMD automatically when using Docker runtime
CMD ["python", "src/agent.py", "dev"]
