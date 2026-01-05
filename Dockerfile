FROM python:3.9-slim

WORKDIR /app

# Install uv (fast Python package installer)
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv pip install --system -r uv.lock

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
