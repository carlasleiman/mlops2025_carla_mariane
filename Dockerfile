# NYC Taxi MLOps Project Dockerfile
# Python 3.11 as required by project

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Install the package in development mode
RUN uv pip install -e .

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command (can be overridden)
CMD ["python", "scripts/train.py", "--help"]
